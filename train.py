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
import torch.nn.functional as F
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

# =========================================================
#  LOSS-FREE BALANCER (Algorithm 1 from Paper)
#  OPTIMIZED FOR DDP & PRE-CALCULATED INDICES
# =========================================================
class LossFreeBalancer:
    """
    Manages the expert biases for Loss-Free Balancing.
    Updates biases based on the sign of the load violation error.
    Optimized to handle DDP synchronization and pre-calculated Top-K indices.
    """
    def __init__(self, num_experts, update_rate=0.001):
        self.num_experts = num_experts
        self.update_rate = update_rate

    @torch.no_grad()
    def update_model_biases(self, model, router_data_tuple, accelerator=None):
        """
        Args:
            model: The unwrapped DCAE model.
            router_data_tuple: List of tuples (logits, topk_indices) OR just logits from forward pass.
            accelerator: HuggingFace Accelerator instance (Required for DDP correctness).
        """
        if router_data_tuple is None:
            return

        for i, layer_data in enumerate(router_data_tuple):
            if isinstance(layer_data, (tuple, list)) and len(layer_data) == 2:
                _, topk_indices = layer_data
            else:
                continue

            # 1. Flatten indices to count globally
            flat_indices = topk_indices.contiguous().view(-1)
            
            # 2. Local Counts (on this GPU)
            local_counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
            
            # 3. Global Synchronization (CRITICAL FOR DDP)
            if accelerator and accelerator.num_processes > 1:
                # Sum counts across all GPUs so every GPU sees the global load
                expert_counts = accelerator.reduce(local_counts, reduction="sum")
            else:
                expert_counts = local_counts

            # 4. Calculate Load Error
            avg_count = expert_counts.mean()
            violation_error = expert_counts - avg_count

            # 5. Update Bias
            # Logic: If violation > 0 (Overloaded), we subtract score.
            # So we subtract the sign of the error.
            update_step = self.update_rate * torch.sign(violation_error)

            # Apply to model buffer
            # Check if model has the specific MoE layer structure
            if hasattr(model, 'dt_cross_attention') and i < len(model.dt_cross_attention):
                moe_module = model.dt_cross_attention[i]
                if hasattr(moe_module, "expert_biases"):
                    # In-place update on correct device
                    moe_module.expert_biases.sub_(update_step.to(moe_module.expert_biases.device))

# =========================================================
#  STANDARD TRAINING UTILS
# =========================================================

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
    model, criterion, train_dataloader, optimizer, aux_optimizer, 
    epoch, clip_max_norm, accelerator, logger, writer, global_step,
    balancer
):
    model.train()
    
    losses = AverageMeter()
    bpp_losses = AverageMeter()
    dist_losses = AverageMeter()
    aux_losses = AverageMeter()
    moe_metrics = AverageMeter()
    
    if criterion.loss_type == "mse":
        dist_key = "mse_loss"
    elif criterion.loss_type == "charbonnier":
        dist_key = "char_loss"
    else:
        dist_key = "ms_ssim_loss"

    for i, d in enumerate(train_dataloader):
        optimizer.zero_grad(set_to_none=True)
        aux_optimizer.zero_grad(set_to_none=True)

        # Forward Pass
        out_net = model(d)
        
        # Compute Loss
        # Note: use_loss_free_balancing=True in criterion means 'moe_loss' is NOT added to 'loss'
        out_criterion = criterion(out_net, d)
        
        # Backward (Main)
        accelerator.backward(out_criterion["loss"])
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if p.grad.stride() != p.grad.contiguous().stride():
                print(f"[WARNING] Param: {name}")
                print(f"  grad shape:   {p.grad.shape}")
                print(f"  grad strides: {p.grad.stride()}")
                print(f"  good strides: {p.grad.contiguous().stride()}")

        if clip_max_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()

        # --- LOSS-FREE BALANCING UPDATE ---
        # Update expert biases based on current batch logits
        # We perform this Update AFTER optimizer step (decoupled update)
        if "router_logits" in out_net and out_net["router_logits"] is not None:
            # We must access the underlying model to modify buffers
            unwrapped = accelerator.unwrap_model(model)
            # Pass accelerator to allow global synchronization of expert counts
            balancer.update_model_biases(
                unwrapped, 
                out_net["router_logits"], 
                accelerator=accelerator
            )

        # Aux Backward (Entropy Model)
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
            
        # Log the calculated (but not optimized) MoE loss to monitor balance
        if "moe_loss" in out_criterion:
             moe_val = out_criterion["moe_loss"]
             if isinstance(moe_val, torch.Tensor): moe_val = moe_val.item()
             moe_metrics.update(moe_val)

        if i % 100 == 0 and accelerator.is_main_process:
             if logger is not None:
                 logger.info(
                    f"Epoch: [{epoch}][{i}/{len(train_dataloader)}]\t"
                    f"Loss {losses.val:.4f}\t"
                    f"Bpp {bpp_losses.val:.4f}\t"
                    f"Dist {dist_losses.val:.5f}\t"
                    f"MoE_Metric {moe_metrics.val:.4f}" # Monitoring only
                )
             
             if writer is not None:
                 writer.add_scalar('Train/Loss', out_criterion["loss"].item(), global_step)
                 writer.add_scalar('Train/Bpp', out_criterion["bpp_loss"].item(), global_step)
                 writer.add_scalar('Train/Distortion', dist_losses.val, global_step)
                 writer.add_scalar('Train/MoE_Imbalance_Score', moe_metrics.val, global_step)

        global_step += 1

    return global_step

def test_epoch(epoch, test_dataloader, model, criterion, accelerator, logger, writer):
    model.eval()
    
    loss_meter = AverageMeter()
    bpp_loss_meter = AverageMeter()
    dist_loss_meter = AverageMeter()
    
    metric_name = "MSE" if criterion.loss_type in ["mse", "charbonnier"] else "MS_SSIM"
    dist_key = "mse_loss" if criterion.loss_type == "mse" else "ms_ssim_loss"
    if criterion.loss_type == "charbonnier": dist_key = "char_loss"

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            loss_meter.update(out_criterion["loss"].item())
            bpp_loss_meter.update(out_criterion["bpp_loss"].item())
            
            if dist_key in out_criterion:
                dist_loss_meter.update(out_criterion[dist_key].item())

            if i == 0 and accelerator.is_main_process and writer is not None:
                n_imgs = min(4, d.size(0))
                inputs = d[:n_imgs].cpu()
                recons = out_net["x_hat"][:n_imgs].cpu().clamp(0, 1)
                combined = torch.cat([inputs, recons], dim=0)
                grid = make_grid(combined, nrow=n_imgs, padding=2, normalize=False)
                writer.add_image('Val_Images', grid, epoch)

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
    parser = argparse.ArgumentParser(description="DCAE Loss-Free Training")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to Dataset")
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("-n", "--num-workers", type=int, default=1)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.0018)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--seed", type=float, default=100)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--type", type=str, default="charbonnier", choices=["mse", "ms-ssim", "charbonnier"])
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--lr_epoch", nargs="+", type=int, default=[80, 90])
    parser.add_argument("--continue_train", action="store_true", default=False)
    
    # Loss-Free Balancing Params
    parser.add_argument("--update-rate", type=float, default=0.001, help="Update rate for expert biases")
    
    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.save_path)
    set_seed(args.seed)

    logger = None
    writer = None
    save_path = ""
    
    if accelerator.is_main_process:
        save_path = os.path.join(args.save_path, f"lambda_{args.lmbda}")
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "tensorboard"), exist_ok=True)
        writer = SummaryWriter(os.path.join(save_path, "tensorboard"))
        logger = setup_logger(save_path)
        logger.info(f"Training started on {accelerator.device}")

    # Data
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    valid_dataset = ImageFolder(args.dataset, split="valid", transform=test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # Model
    net = DCAE()
    
    # --- Initialize Balancer ---
    # Assuming standard MoE config (4 experts)
    # If dynamic, get from net config
    balancer = LossFreeBalancer(num_experts=4, update_rate=args.update_rate)

    start_epoch = 0
    best_loss = float("inf")
    
    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch if args.lr_epoch else [args.epochs - 10]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_loss = checkpoint["loss"]

    # --- Loss Config ---
    # ENABLE Loss-Free: use_loss_free_balancing=True
    # This prevents RateDistortionLoss from adding moe_loss to gradients
    criterion = RateDistortionLoss(
        lmbda=args.lmbda, 
        loss_type=args.type, 
        alpha_moe=1.0, 
        use_loss_free_balancing=True 
    )

    net, optimizer, aux_optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        net, optimizer, aux_optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']}")

        # Pass balancer to train loop
        global_step = train_one_epoch(
            net, criterion, train_dataloader, optimizer, aux_optimizer,
            epoch, args.clip_max_norm, accelerator, logger, writer, global_step,
            balancer=balancer
        )

        loss = test_epoch(epoch, valid_dataloader, net, criterion, accelerator, logger, writer)

        if accelerator.is_main_process:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": accelerator.unwrap_model(net).state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best, epoch, save_path
                )

        lr_scheduler.step()
    
    if accelerator.is_main_process:
        writer.close()
if __name__ == "__main__":
    main(sys.argv[1:])