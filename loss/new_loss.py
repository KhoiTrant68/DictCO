import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Union, Tuple

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
        pred_freq = torch.fft.rfft2(pred, norm='ortho')
        target_freq = torch.fft.rfft2(target, norm='ortho')
        diff = (pred_freq.abs() - target_freq.abs()).pow(2)
        weight = diff / (diff.mean() + 1e-8)
        weight = torch.clamp(weight, min=0.1, max=10.0) ** self.alpha
        return (diff * weight).mean()

# ==============================================================================
#  OPTIMIZED LOAD BALANCING (Simplified for Images)
# ==============================================================================
def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]],
    num_experts: int,
    top_k: int = 2
) -> torch.Tensor:
    """
    Computes auxiliary load balancing loss.
    Accepts gate_logits as:
      1. A single Tensor [N, Experts]
      2. A List of Tensors (one per layer)
      3. A List of Tuples (logits, indices) (one per layer, optimized)
    """
    if gate_logits is None:
        return torch.tensor(0.0)

    # 1. Normalize input to a list of items
    # If it's a single tensor, wrap it
    if isinstance(gate_logits, torch.Tensor):
        layer_outputs = [gate_logits]
    # If it's a tuple (logits, indices) representing a single layer, wrap it
    elif isinstance(gate_logits, tuple) and len(gate_logits) == 2 and isinstance(gate_logits[0], torch.Tensor):
         layer_outputs = [gate_logits]
    else:
        # It's already a list (of tensors or tuples)
        layer_outputs = gate_logits

    if len(layer_outputs) == 0:
        return torch.tensor(0.0)

    total_loss = 0.0

    # 2. Iterate over each MoE layer
    for item in layer_outputs:
        # Check if item is (logits, indices) or just logits
        if isinstance(item, (tuple, list)) and len(item) == 2:
            logits, selected_indices = item
        else:
            logits = item
            # Fallback: Calculate Top-K if indices are missing
            _, selected_indices = torch.topk(logits, top_k, dim=-1)

        # Ensure logits are on the correct device for loss calculation
        if logits.numel() == 0:
            continue

        # Flatten logits: [B, H, W, E] -> [N, E]
        if logits.dim() > 2:
            logits = logits.reshape(-1, num_experts)
            
        # 3. Softmax (Probabilities)
        probs = torch.softmax(logits, dim=-1)
        mean_probs = torch.mean(probs, dim=0) # [Experts]

        # 4. Hard Selection (Load) using Indices
        # selected_indices shape: [N, k]
        # Flatten indices: [N*k]
        flat_indices = selected_indices.contiguous().view(-1)
        
        # Count occurrences (Load per expert)
        expert_counts = torch.bincount(flat_indices, minlength=num_experts).float()
        
        # Fraction of tokens per expert
        # Note: We divide by total top-k selections (N * k)
        total_selections = flat_indices.numel()
        if total_selections > 0:
            fraction_tokens = expert_counts / total_selections
        else:
            fraction_tokens = torch.zeros_like(mean_probs)

        # 5. Switch Transformer Loss: N * sum(f_i * P_i)
        # We multiply by num_experts so perfect balance = 1.0
        loss = num_experts * torch.sum(mean_probs * fraction_tokens)
        total_loss += loss

    return total_loss

# ==============================================================================
#  MAIN LOSS MODULE
# ==============================================================================
class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2, loss_type="mse", 
                 alpha_spectral=0.1, alpha_moe=0.01,
                 num_experts=4, top_k=2,
                 use_loss_free_balancing=True): # <--- NEW FLAG
        super().__init__()
        self.lmbda = lmbda
        self.loss_type = loss_type
        self.alpha_spectral = alpha_spectral
        self.alpha_moe = alpha_moe
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_loss_free_balancing = use_loss_free_balancing

        self.mse = nn.MSELoss()
        self.charbonnier = CharbonnierLoss()
        self.ffl = FocalFrequencyLoss()

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}

        # --- 1. Rate Loss (BPP) ---
        y_lik = output["likelihoods"]["y"]
        if isinstance(y_lik, (list, tuple)):
            total_y_bpp = sum(-torch.log(y_l).sum() for y_l in y_lik)
        else:
            total_y_bpp = -torch.log(y_lik).sum()

        bpp_z = -torch.log(output["likelihoods"]["z"]).sum()
        out["bpp_loss"] = (total_y_bpp + bpp_z) / (math.log(2) * num_pixels)

        # --- 2. Main Distortion ---
        x_hat = output["x_hat"]
        
        if self.loss_type == "mse":
            out["mse_loss"] = self.mse(x_hat, target)
            dist_loss = 255**2 * out["mse_loss"]
        elif self.loss_type == "charbonnier":
            out["char_loss"] = self.charbonnier(x_hat, target)
            dist_loss = 255**2 * out["char_loss"]
        elif self.loss_type == "ms_ssim":
            if ms_ssim is None: raise ImportError("pytorch_msssim not installed")
            out["ms_ssim_loss"] = 1 - ms_ssim(x_hat, target, data_range=1.0)
            dist_loss = 255**2 * out["ms_ssim_loss"] 
        else:
            raise ValueError(f"Unknown metric: {self.loss_type}")

        out["dist_loss"] = dist_loss

        # --- 3. Auxiliary Losses ---
        total_aux = 0

        # A. Spectral Loss
        if self.alpha_spectral > 0:
            ffl_val = self.ffl(x_hat, target)
            out["spectral_loss"] = ffl_val
            total_aux += self.alpha_spectral * ffl_val

        # B. MoE Load Balancing
        # We always CALCULATE it for monitoring, but we check if we ADD it.
        if self.alpha_moe > 0:
            moe_loss = 0
            if "router_logits" in output and output["router_logits"] is not None:
                moe_loss = load_balancing_loss_func(
                    output["router_logits"], 
                    num_experts=self.num_experts, 
                    top_k=self.top_k
                )
            
            out["moe_loss"] = moe_loss
            
            # --- CRITICAL LOGIC ---
            if self.use_loss_free_balancing:
                # If using Loss-Free (Bias Updates), do NOT add to gradient.
                # We just log it to see if it remains low.
                pass 
            else:
                # If NOT using Loss-Free, add to gradient to force balance via SGD.
                total_aux += self.alpha_moe * moe_loss

        out["aux_loss"] = total_aux

        # --- 4. Total Loss ---
        out["loss"] = out["bpp_loss"] + self.lmbda * (dist_loss + total_aux)

        return out