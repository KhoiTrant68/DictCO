import argparse
import math
import os
import random
import sys
import time
import logging
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

# Hugging Face Accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

from compressai.datasets import ImageFolder
from loss.new_loss import AverageMeter, RateDistortionLoss 
from models.dcae import DCAE

# --- Global Settings ---
os.environ["TMPDIR"] = "/tmp"
torch.backends.cudnn.benchmark = True

def setup_logger(log_dir):
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(log_dir, "train_log.txt"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

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
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, accelerator, logger, writer, global_step
):
    model.train()
    
    losses = AverageMeter()
    bpp_losses = AverageMeter()
    dist_losses = AverageMeter()
    aux_losses = AverageMeter()
    
    # Identify distortion key
    if criterion.loss_type == "mse":
        dist_key = "mse_loss"
    elif criterion.loss_type == "charbonnier":
        dist_key = "char_loss"
    else:
        dist_key = "ms_ssim_loss"

    for i, d in enumerate(train_dataloader):
        # Accelerate handles device placement automatically!
        
        optimizer.zero_grad(set_to_none=True)
        aux_optimizer.zero_grad(set_to_none=True)

        # Forward Pass
        out_net = model(d)
        out_criterion = criterion(out_net, d)
        
        # Backward (Main)
        accelerator.backward(out_criterion["loss"])

        if clip_max_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()

        # Aux Backward
        # Need to unwrap model to access .aux_loss() in DDP
        unwrapped_model = accelerator.unwrap_model(model)
        aux_loss = unwrapped_model.aux_loss()
        accelerator.backward(aux_loss)
        aux_optimizer.step()
        
        # --- Metrics & Logging ---
        losses.update(out_criterion["loss"].item())
        bpp_losses.update(out_criterion["bpp_loss"].item())
        aux_losses.update(aux_loss.item())
        
        if dist_key in out_criterion:
            dist_losses.update(out_criterion[dist_key].item())

        # Log only on main process
        if i % 100 == 0 and accelerator.is_main_process:
             if logger is not None:
                 logger.info(
                    f"Epoch: [{epoch}][{i}/{len(train_dataloader)}]\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Bpp {bpp_losses.val:.4f} ({bpp_losses.avg:.4f})\t"
                    f"Dist {dist_losses.val:.5f} ({dist_losses.avg:.5f})\t"
                    f"Aux {aux_losses.val:.3f} ({aux_losses.avg:.3f})"
                )
             
             if writer is not None:
                 writer.add_scalar('Train/Loss_Step', out_criterion["loss"].item(), global_step)
                 writer.add_scalar('Train/Bpp_Step', out_criterion["bpp_loss"].item(), global_step)
                 writer.add_scalar('Train/Distortion_Step', dist_losses.val, global_step)
                 writer.add_scalar('Train/Aux_Step', aux_losses.val, global_step)
                 if "spectral_loss" in out_criterion and out_criterion["spectral_loss"] > 0:
                     writer.add_scalar('Train/Spectral_Loss_Step', out_criterion["spectral_loss"].item(), global_step)
                 if "dict_loss" in out_criterion and out_criterion["dict_loss"] > 0:
                     writer.add_scalar('Train/Dict_Loss_Step', out_criterion["dict_loss"].item(), global_step)

        global_step += 1

    return global_step

def test_epoch(epoch, test_dataloader, model, criterion, accelerator, logger, writer):
    model.eval()
    
    loss_meter = AverageMeter()
    bpp_loss_meter = AverageMeter()
    dist_loss_meter = AverageMeter()
    
    if criterion.loss_type == "mse" or criterion.loss_type == "charbonnier":
        dist_key = "mse_loss"
        metric_name = "MSE"
    else:
        dist_key = "ms_ssim_loss"
        metric_name = "MS_SSIM"

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            # Accelerate handles device
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            loss_meter.update(out_criterion["loss"].item())
            bpp_loss_meter.update(out_criterion["bpp_loss"].item())
            
            if dist_key in out_criterion:
                dist_loss_meter.update(out_criterion[dist_key].item())
            elif "char_loss" in out_criterion:
                 dist_loss_meter.update(out_criterion["char_loss"].item())

            # --- Image Logging (First batch, Main Process Only) ---
            if i == 0 and accelerator.is_main_process and writer is not None:
                n_imgs = min(4, d.size(0))
                inputs = d[:n_imgs].cpu()
                recons = out_net["x_hat"][:n_imgs].cpu().clamp(0, 1)
                combined = torch.cat([inputs, recons], dim=0)
                grid = make_grid(combined, nrow=n_imgs, padding=2, normalize=False)
                writer.add_image('Val_Images/Original_vs_Recon', grid, epoch)

    if accelerator.is_main_process:
        if logger is not None:
            logger.info(
                f"\nTest Epoch: [{epoch}]\t"
                f"Loss: {loss_meter.avg:.4f}\t"
                f"Bpp: {bpp_loss_meter.avg:.4f}\t"
                f"Dist ({metric_name}): {dist_loss_meter.avg:.6f}\n"
            )
        if writer is not None:
            writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
            writer.add_scalar('Val/Bpp', bpp_loss_meter.avg, epoch)
            writer.add_scalar(f'Val/{metric_name}', dist_loss_meter.avg, epoch)

    return loss_meter.avg

def save_checkpoint(state, is_best, epoch, save_path, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path, "checkpoint_best.pth.tar"))

def parse_args(argv):
    parser = argparse.ArgumentParser(description="DCAE Training Script (Accelerate)")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to Dataset")
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("-n", "--num-workers", type=int, default=2)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.0018)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--seed", type=float, default=100)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--type", type=str, default="mse", choices=["mse", "ms-ssim", "charbonnier"])
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--lr_epoch", nargs="+", type=int, default=[40, 45])
    parser.add_argument("--continue_train", action="store_true", default=False)
    # Accelerate handles mixed precision via 'accelerate config'
    # but we can accept a flag if we want manual control, though config is better.
    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    
    # --- 1. Accelerate Init ---
    # Mixed precision is determined by 'accelerate config' or can be passed here
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.save_path)
    set_seed(args.seed)

    # --- 2. Logging Setup (Main Process Only) ---
    logger = None
    writer = None
    save_path = ""
    
    if accelerator.is_main_process:
        save_path = os.path.join(args.save_path, f"lambda_{args.lmbda}")
        os.makedirs(save_path, exist_ok=True)
        # We let Accelerator manage the SummaryWriter via 'log_with', 
        # but to keep your custom logic (images), we access it directly or use our own.
        # Accelerate's 'get_tracker' is the cleanest way, but standard SummaryWriter works fine on rank 0.
        os.makedirs(os.path.join(save_path, "tensorboard"), exist_ok=True)
        writer = SummaryWriter(os.path.join(save_path, "tensorboard"))
        logger = setup_logger(save_path)
        logger.info(f"Training started on {accelerator.device}. Mixed Precision: {accelerator.mixed_precision}")

    # --- 3. Data Loading ---
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    valid_dataset = ImageFolder(args.dataset, split="valid", transform=test_transforms)

    # No need for DistributedSampler, Accelerate handles it!
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        num_workers=args.num_workers, shuffle=True, pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.test_batch_size, 
        num_workers=args.num_workers, shuffle=False, pin_memory=True
    )

    # --- 4. Model & Optimizer ---
    net = DCAE()
    
    # Load Checkpoint (Before prepare)
    start_epoch = 0
    best_loss = float("inf")
    
    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch if args.lr_epoch else [args.epochs - 10]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    
    if args.checkpoint:
        if accelerator.is_main_process: logger.info(f"Loading checkpoint: {args.checkpoint}")
        # Use standard load, accelerate will handle device placement in prepare
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        
        state_dict = checkpoint["state_dict"]
        # Clean 'module.' prefix if it exists
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        net.load_state_dict(new_state_dict)

        if args.continue_train:
            start_epoch = checkpoint["epoch"] + 1
            if "optimizer" in checkpoint: optimizer.load_state_dict(checkpoint["optimizer"])
            # Aux optimizer often not saved/loaded in some pipelines, but good to have
            if "aux_optimizer" in checkpoint: aux_optimizer.load_state_dict(checkpoint["aux_optimizer"]) 
            if "lr_scheduler" in checkpoint: lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if "loss" in checkpoint: best_loss = checkpoint["loss"]

    criterion = RateDistortionLoss(lmbda=args.lmbda, loss_type=args.type)

    # --- 5. Accelerate Prepare ---
    # Magic happens here. 
    # NOTE: Do NOT prepare aux_optimizer if it only works on non-differentiable params in a specific way, 
    # but usually safe to prepare. For CompressAI, aux_optimizer updates quantiles.
    net, optimizer, aux_optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        net, optimizer, aux_optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    global_step = 0
    
    # --- 6. Train Loop ---
    for epoch in range(start_epoch, args.epochs):
        
        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Train/LR', current_lr, epoch)
            logger.info(f"Start Epoch {epoch} | LR: {current_lr}")

        global_step = train_one_epoch(
            net, criterion, train_dataloader, optimizer, aux_optimizer,
            epoch, args.clip_max_norm, accelerator, logger, writer, global_step
        )

        loss = test_epoch(epoch, valid_dataloader, net, criterion, accelerator, logger, writer)

        # Save State
        if accelerator.is_main_process:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                # Unwrap model to save clean state dict
                unwrapped_net = accelerator.unwrap_model(net)
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": unwrapped_net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    epoch,
                    save_path
                )

        lr_scheduler.step()
    
    if accelerator.is_main_process:
        writer.close()
        logger.info("Training Complete.")

if __name__ == "__main__":
    main(sys.argv[1:])