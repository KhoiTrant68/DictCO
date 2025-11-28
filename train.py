import argparse
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler  # Added for AMP
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim

# Assuming models are defined in a local file named models.py
from models.dcae import DCAE

# --- Global Settings ---
os.environ["TMPDIR"] = "/tmp"
torch.set_num_threads(8)
# Optimization: Benchmark=True is faster for fixed input sizes (enabled by RandomCrop)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, loss_type="mse"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.loss_type = loss_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if self.loss_type == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out["ms_ssim_loss"] = ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * (1 - out["ms_ssim_loss"]) + out["bpp_loss"]

        return out


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer."""
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    clip_max_norm,
    scaler,
    lr_scheduler=None,
    args=None,
):
    model.train()
    device = next(model.parameters()).device

    # Optimization: Only start timer once
    start_time = time.time()

    for i, d in enumerate(train_dataloader):
        d = d.to(device, non_blocking=True)  # Optimization: non_blocking transfer

        optimizer.zero_grad(set_to_none=True)  # Optimization: set_to_none is faster
        aux_optimizer.zero_grad(set_to_none=True)

        # --- Automatic Mixed Precision (AMP) ---
        with autocast(device_type="cpu"):
            out_net = model(d)
            out_criterion = criterion(out_net, d)

        scaler.scale(out_criterion["loss"]).backward()

        if clip_max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        scaler.step(optimizer)
        scaler.update()

        # Aux optimizer usually doesn't need scaling as it acts on quantiles directly,
        # but check CompressAI docs if your specific model differs.
        # Standard flow:
        aux_loss = (
            model.module.aux_loss()
            if isinstance(model, nn.DataParallel)
            or isinstance(model, nn.parallel.DistributedDataParallel)
            else model.aux_loss()
        )
        aux_loss.backward()
        aux_optimizer.step()

        if (i + 1) % 100 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            start_time = current_time  # Reset for next interval

            lr = lr_scheduler.get_last_lr()[0]

            # Use f-string and consolidated printing
            dist_loss_val = (
                out_criterion["mse_loss"].item()
                if criterion.loss_type == "mse"
                else out_criterion["ms_ssim_loss"].item()
            )
            dist_loss_name = "MSE" if criterion.loss_type == "mse" else "MS_SSIM"

            print(
                f"Train epoch {epoch}: [{(i+1)*len(d)}/{len(train_dataloader.dataset)}] "
                f"Time: {elapsed:.2f}s | LR: {lr:.2e} | "
                f"Loss: {out_criterion['loss'].item():.3f} | "
                f"{dist_loss_name}: {dist_loss_val:.3f} | "
                f"Bpp: {out_criterion['bpp_loss'].item():.2f} | "
                f"Aux: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion, args=None):
    model.eval()
    device = next(model.parameters()).device

    loss_meter = AverageMeter()
    bpp_loss_meter = AverageMeter()
    dist_loss_meter = AverageMeter()  # Generic meter for MSE or MS-SSIM

    is_mse = criterion.loss_type == "mse"

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device, non_blocking=True)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            loss_meter.update(out_criterion["loss"].item())
            bpp_loss_meter.update(out_criterion["bpp_loss"].item())

            if is_mse:
                dist_loss_meter.update(out_criterion["mse_loss"].item())
            else:
                dist_loss_meter.update(out_criterion["ms_ssim_loss"].item())

    # Consolidated print
    dist_label = "MSE loss" if is_mse else "MS_SSIM loss"
    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss_meter.avg:.3f} |"
        f"\t{dist_label}: {dist_loss_meter.avg:.3f} |"
        f"\tBpp loss: {bpp_loss_meter.avg:.2f}\n"
    )

    return loss_meter.avg


def save_checkpoint(state, is_best, epoch, save_path):
    torch.save(state, os.path.join(save_path, "checkpoint_latest.pth.tar"))
    if epoch % 5 == 0:
        torch.save(state, os.path.join(save_path, f"{epoch}_checkpoint.pth.tar"))
    if is_best:
        torch.save(state, os.path.join(save_path, "checkpoint_best.pth.tar"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    # Standard args...
    parser.add_argument("--local-rank", default=os.getenv("LOCAL_RANK", -1), type=int)
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument(
        "-n", "--num-workers", type=int, default=8
    )  # Lowered default to safe value
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.0018)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--seed", type=float, default=100)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--type", type=str, default="mse", choices=["mse", "ms-ssim"])
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument(
        "--lr_epoch", nargs="+", type=int, default=[40, 45]
    )  # Added default
    parser.add_argument("--continue_train", action="store_true", default=True)

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # Initialize Distribution
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        global_rank = dist.get_rank()
    else:
        device = torch.device(
            "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
        )
        global_rank = 0

    # Logging setup (only on rank 0)
    if global_rank == 0:
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")

        save_path = os.path.join(args.save_path, str(args.lmbda))
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(os.path.join(save_path, "tensorboard"), exist_ok=True)
        writer = SummaryWriter(os.path.join(save_path, "tensorboard"))
    else:
        save_path = ""  # Placeholder for other ranks
        writer = None

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Transforms
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    # Datasets
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    valid_dataset = ImageFolder(args.dataset, split="valid", transform=test_transforms)

    # Samplers
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        valid_sampler = None
        shuffle_train = True

    # Loaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=shuffle_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # Model
    net = DCAE().to(device)

    # Optimization: find_unused_parameters=False is much faster if model graph is static
    if args.local_rank != -1:
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    optimizer, aux_optimizer = configure_optimizers(net, args)

    # Gradient Scaler for AMP
    scaler = GradScaler()

    milestones = args.lr_epoch if args.lr_epoch else [args.epochs - 10]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=0.1, last_epoch=-1
    )

    criterion = RateDistortionLoss(lmbda=args.lmbda, loss_type=args.type).to(device)

    last_epoch = 0
    if args.checkpoint:
        print("Loading", args.checkpoint)
        # Map location is important for loading on specific GPUs
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Handle state dict loading for DDP vs non-DDP
        state_dict = checkpoint["state_dict"]
        # (Optional) Logic to remove 'module.' prefix if loading DDP checkpoint to non-DDP model could go here

        net.load_state_dict(state_dict)

        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")

    for epoch in range(last_epoch, args.epochs):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            scaler,
            args,
            lr_scheduler,
        )

        loss = test_epoch(epoch, valid_dataloader, net, criterion, args)

        if global_rank == 0:
            writer.add_scalar("test_loss", loss, epoch)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    epoch,
                    save_path,
                )

        lr_scheduler.step()


if __name__ == "__main__":
    main(sys.argv[1:])
