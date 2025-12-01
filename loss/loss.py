import torch
import torch.nn as nn
from torch.nn import functional as F 
import math

# Try importing MS-SSIM
try:
    from pytorch_msssim import ms_ssim
except ImportError:
    ms_ssim = None


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

class CharbonnierLoss(nn.Module):
    """
    Robust L1 Loss (Charbonnier).
    Behaves like L2 near zero (for stability) and L1 elsewhere (for sharpness).
    Used in SOTA restoration and compression (MLIC++, SwinIR).
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = sqrt(diff^2 + eps^2)
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

class RateDistortionLoss(nn.Module):
    """
    Advanced R-D Loss inspired by MLIC++.
    Components:
    1. BPP Loss (Rate)
    2. Charbonnier Loss (Robust Spatial Distortion)
    3. Dictionary Consistency Loss (Auxiliary - New!)
    """
    def __init__(self, lmbda=1e-2, loss_type="charbonnier", alpha_dict=0.1):
        """
        Args:
            lmbda: Lagrangian multiplier (Quality vs Rate).
            loss_type: 'mse', 'charbonnier', or 'ms_ssim'.
            alpha_dict: Weight for the Dictionary Consistency Loss.
        """
        super().__init__()
        self.lmbda = lmbda
        self.loss_type = loss_type
        self.alpha_dict = alpha_dict

        # Metric definitions
        self.mse = nn.MSELoss()
        self.charbonnier = CharbonnierLoss()
        
        if loss_type == "ms_ssim" and ms_ssim is None:
            raise ImportError("Please install pytorch_msssim.")

    def forward(self, output, target):
        """
        output: dict with 'x_hat', 'likelihoods', and optionally 'dict_features'
        target: input image
        """
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # --- 1. Rate Loss (BPP) ---
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        # --- 2. Distortion Loss ---
        if self.loss_type == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            distortion = 255**2 * out["mse_loss"]
            
        elif self.loss_type == "charbonnier":
            # Charbonnier is usually on [0,1] scale, but we scale it 
            # to match the magnitude of 255^2 MSE for consistent lambda usage.
            out["char_loss"] = self.charbonnier(output["x_hat"], target)
            distortion = 255**2 * out["char_loss"]
            
        elif self.loss_type == "ms_ssim":
            out["ms_ssim_loss"] = ms_ssim(output["x_hat"], target, data_range=1.0)
            distortion = 1 - out["ms_ssim_loss"]

        # --- 3. Dictionary Consistency Loss (Novelty) ---
        # We want the dictionary output (prior) to be somewhat close to the actual latent (y).
        # This helps the MoE experts converge faster.
        if "dict_info" in output and "y" in output["para"]:
            # dict_info is the output of the MoE
            # y is the actual latent representation
            # We detach y because we don't want to change the encoder features to match the dictionary,
            # we want the dictionary to match the encoder features.
            
            dict_out = output["dict_info"] 
            target_y = output["para"]["y"].detach() # Stop gradient on target
            
            # Simple MSE between dictionary guess and actual latent
            dict_loss = F.mse_loss(dict_out, target_y)
            out["dict_loss"] = dict_loss
            
            # Add to total distortion with a small weight
            distortion = distortion + (self.alpha_dict * dict_loss * (255**2))
        else:
            out["dict_loss"] = torch.tensor(0.0)

        # --- Final Weighted Sum ---
        out["loss"] = self.lmbda * distortion + out["bpp_loss"]

        return out