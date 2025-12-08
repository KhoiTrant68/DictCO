import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Robust L1 Loss (differentiable L1)."""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

class FocalFrequencyLoss(nn.Module):
    """Weights frequency bands based on difficulty (Spectral Loss)."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # Ortho norm ensures energy conservation
        pred_freq = torch.fft.rfft2(pred, norm='ortho')
        target_freq = torch.fft.rfft2(target, norm='ortho')

        # Calculate difference spectrum
        diff = (pred_freq.abs() - target_freq.abs()).pow(2)

        # Dynamic weighting: harder frequencies get higher weight
        # We assume harder frequencies have larger errors relative to the mean
        weight = diff / (diff.mean() + 1e-8)
        weight = torch.clamp(weight, min=0.1, max=10.0) ** self.alpha

        return (diff * weight).mean()

class RateDistortionLoss(nn.Module):
    """
    Main Objective Function: R + lambda * D
    Includes Auxiliary losses for Spectral consistency and MoE balancing.
    """
    def __init__(self, lmbda=1e-2, loss_type="mse", 
                 alpha_spectral=0.1, alpha_moe=1.0):
        super().__init__()
        self.lmbda = lmbda
        self.loss_type = loss_type
        
        # Weights for Aux losses
        self.alpha_spectral = alpha_spectral
        self.alpha_moe = alpha_moe

        # Loss Modules
        self.mse = nn.MSELoss()
        self.charbonnier = CharbonnierLoss()
        self.ffl = FocalFrequencyLoss()

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}

        # --- 1. Rate Loss (BPP) ---
        # Handle list of likelihoods (from slicing) or single tensor
        y_lik = output["likelihoods"]["y"]
        if isinstance(y_lik, list):
            bpp_y = sum(-torch.log(y_l).sum() for y_l in y_lik)
        else:
            bpp_y = -torch.log(y_lik).sum()

        bpp_z = -torch.log(output["likelihoods"]["z"]).sum()
        
        # Total BPP
        out["bpp_loss"] = (bpp_y + bpp_z) / (math.log(2) * num_pixels)

        # --- 2. Main Distortion ---
        x_hat = output["x_hat"]
        
        if self.loss_type == "mse":
            out["mse_loss"] = self.mse(x_hat, target)
            # Scale MSE to make it comparable to typical lambda ranges
            dist_loss = 255**2 * out["mse_loss"]
            
        elif self.loss_type == "charbonnier":
            out["char_loss"] = self.charbonnier(x_hat, target)
            dist_loss = 255**2 * out["char_loss"]
            
        elif self.loss_type == "ms_ssim":
            if ms_ssim is None:
                raise ImportError("pytorch_msssim not installed")
            # MS-SSIM returns 1 for perfect, 0 for bad. We want to minimize.
            out["ms_ssim_loss"] = 1 - ms_ssim(x_hat, target, data_range=1.0)
            dist_loss = 255**2 * out["ms_ssim_loss"] 
            
        else:
            raise ValueError(f"Unknown metric: {self.loss_type}")

        out["dist_loss"] = dist_loss

        # --- 3. Auxiliary Losses ---
        aux_loss = 0

        # A. Spectral Loss (Focal Frequency)
        if self.alpha_spectral > 0:
            ffl_val = self.ffl(x_hat, target)
            out["spectral_loss"] = ffl_val
            aux_loss += self.alpha_spectral * ffl_val

        # B. MoE Load Balancing
        # Checks if the model output contains router weights (from swin_module)
        if self.alpha_moe > 0 and "router_weights" in output:
            router_weights = output["router_weights"]
            if len(router_weights) > 0:
                loss_moe = 0
                for w in router_weights:
                    # w shape: [N_pixels, Experts] or similar
                    # Encourage uniform usage across the batch
                    # Calculate Coefficient of Variation (CV) of expert importance
                    
                    # Mean usage per expert across batch/spatial
                    usage = w.mean(dim=0) 
                    
                    # We want variance of usage to be low (flat distribution)
                    # Loss = Var(usage) / (Mean(usage)^2)
                    loss_moe += usage.var() / (usage.mean().pow(2) + 1e-6)
                
                out["moe_loss"] = loss_moe
                aux_loss += self.alpha_moe * loss_moe

        out["aux_loss"] = aux_loss

        # --- 4. Total Loss ---
        # R + lambda * D + Aux
        out["loss"] = out["bpp_loss"] + self.lmbda * (dist_loss + aux_loss)

        return out