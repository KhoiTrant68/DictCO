import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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
    """Robust L1 Loss."""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2, loss_type="mse", alpha_dict=1.0, alpha_spectral=0.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.charbonnier = CharbonnierLoss()
        self.lmbda = lmbda
        self.loss_type = loss_type
        
        # Weights for auxiliary losses
        self.alpha_dict = alpha_dict       # Weight for Latent Consistency
        self.alpha_spectral = alpha_spectral # Weight for Spectral Loss

    def calculate_spectral_loss(self, x_hat, target):
        fft_pred = torch.fft.rfft2(x_hat, norm="ortho")
        fft_target = torch.fft.rfft2(target, norm="ortho")
        return (fft_pred - fft_target).abs().mean()

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # 1. BPP Loss (Rate)
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        # 2. Main Distortion (Pixel Space)
        if self.loss_type == "mse":
            out["mse_loss"] = self.mse(output["x_hat"], target)
            # Standard formulation: lambda * 255^2 * MSE
            main_distortion = 255**2 * out["mse_loss"]
            
        elif self.loss_type == "charbonnier":
            out["char_loss"] = self.charbonnier(output["x_hat"], target)
            # Scale Charbonnier similarly to MSE for lambda consistency
            main_distortion = 255**2 * out["char_loss"]
            
        elif self.loss_type == "ms_ssim":
            out["ms_ssim_loss"] = ms_ssim(output["x_hat"], target, data_range=1.0)
            main_distortion = 1 - out["ms_ssim_loss"]

        # 3. Spectral Loss (Optional)
        spectral_loss = 0
        if self.alpha_spectral > 0:
            spec_val = self.calculate_spectral_loss(output["x_hat"], target)
            out["spectral_loss"] = spec_val
            # Spectral loss is small, scale it up to match pixel magnitude
            spectral_loss = 255**2 * spec_val

        # 4. Dictionary Consistency Loss (Latent Space)
        # FIX: Do NOT scale this by 255^2. Latents are not pixels.
        dict_loss_val = 0
        if "dict_info" in output and "y" in output["para"]:
            dict_out = output["dict_info"]
            target_y = output["para"]["y"].detach()
            
            # Simple MSE in latent space
            loss_d = F.mse_loss(dict_out, target_y)
            out["dict_loss"] = loss_d
            dict_loss_val = loss_d
        
        # --- TOTAL LOSS CALCULATION ---
        # RD Loss = Rate + lambda * (Spatial + Spectral)
        rd_loss = out["bpp_loss"] + self.lmbda * (main_distortion + self.alpha_spectral * spectral_loss)
        
        # Add Dictionary Regularizer separately (unscaled by lambda)
        # We treat this as a helper loss, not part of the R-D trade-off
        total_loss = rd_loss + (self.alpha_dict * dict_loss_val)

        out["loss"] = total_loss
        return out