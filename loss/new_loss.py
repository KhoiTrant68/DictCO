import torch
import torch.nn as nn
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

# ==============================================================================
#  OPTIMIZED LOAD BALANCING (Supports Tuple inputs)
# ==============================================================================
def load_balancing_loss_func(
    gate_data: Union[torch.Tensor, List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]],
    num_experts: int,
    top_k: int = 2
) -> torch.Tensor:
    """
    Computes auxiliary load balancing loss.
    
    Optimized to handle 'gate_data' in two formats:
      1. Logits only: [N, Experts] -> Calculates TopK internally (Slower).
      2. Tuple: (Logits, Indices) -> Uses pre-calculated indices (Faster).
    """
    if gate_data is None:
        return torch.tensor(0.0)

    # Normalize input to a list
    if isinstance(gate_data, torch.Tensor):
        layer_outputs = [gate_data]
    elif isinstance(gate_data, tuple) and len(gate_data) == 2 and isinstance(gate_data[0], torch.Tensor):
        # Single layer tuple
        layer_outputs = [gate_data]
    else:
        # Already a list
        layer_outputs = gate_data

    if len(layer_outputs) == 0:
        return torch.tensor(0.0)

    total_loss = 0.0

    for item in layer_outputs:
        # Check if item is (logits, indices) or just logits
        if isinstance(item, (tuple, list)) and len(item) == 2:
            logits, selected_indices = item
        else:
            logits = item
            # Fallback: Calculate Top-K if indices are missing
            # This happens if the model didn't return indices
            _, selected_indices = torch.topk(logits, top_k, dim=-1)

        # Skip empty tensors
        if logits.numel() == 0:
            continue

        # Flatten logits: [B, H, W, E] -> [N, E] if needed for softmax
        if logits.dim() > 2:
            logits_flat = logits.reshape(-1, num_experts)
        else:
            logits_flat = logits
            
        # 1. Softmax (Probabilities) - The "Gate"
        probs = torch.softmax(logits_flat, dim=-1)
        mean_probs = torch.mean(probs, dim=0) # [Experts]

        # 2. Hard Selection (Load) using Indices
        flat_indices = selected_indices.contiguous().view(-1)
        
        # Count occurrences (Load per expert)
        expert_counts = torch.bincount(flat_indices, minlength=num_experts).float()
        
        # Fraction of tokens per expert
        total_selections = flat_indices.numel()
        if total_selections > 0:
            fraction_tokens = expert_counts / total_selections
        else:
            fraction_tokens = torch.zeros_like(mean_probs)

        # 3. Switch Transformer Loss: N * sum(f_i * P_i)
        # We multiply by num_experts so perfect balance = 1.0 (minimum is 1.0)
        loss = num_experts * torch.sum(mean_probs * fraction_tokens)
        total_loss += loss

    return total_loss

# ==============================================================================
#  MAIN LOSS MODULE
# ==============================================================================
class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2, loss_type="mse", 
                 alpha_moe=0.01,
                 num_experts=4, top_k=2,
                 use_loss_free_balancing=True):
        super().__init__()
        self.lmbda = lmbda
        self.loss_type = loss_type
        self.alpha_moe = alpha_moe
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_loss_free_balancing = use_loss_free_balancing

        self.mse = nn.MSELoss()
        self.charbonnier = CharbonnierLoss()

    def forward(self, output, target):
        """
        output: Dictionary containing:
            - "x_hat": Reconstructed image
            - "likelihoods": {"y": ..., "z": ...}
            - "router_logits": List of (logits, indices) or logits
        """
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}

        # --- 1. Rate Loss (BPP) ---
        y_lik = output["likelihoods"]["y"]
        z_lik = output["likelihoods"]["z"]
        
        # Sum logs first for numerical stability
        if isinstance(y_lik, (list, tuple)):
            # If y is split into slices
            total_y_log_lik = sum(torch.log(y_l).sum() for y_l in y_lik)
        else:
            total_y_log_lik = torch.log(y_lik).sum()

        total_z_log_lik = torch.log(z_lik).sum()
        
        # BPP = -log2(likelihood) / pixels
        out["bpp_loss"] = (-total_y_log_lik - total_z_log_lik) / (math.log(2) * num_pixels)

        # --- 2. Distortion Loss ---
        x_hat = output["x_hat"]
        
        if self.loss_type == "mse":
            out["mse_loss"] = self.mse(x_hat, target)
            # CompressAI convention: scale MSE by 255^2
            dist_loss = 255**2 * out["mse_loss"]
            
        elif self.loss_type == "charbonnier":
            out["char_loss"] = self.charbonnier(x_hat, target)
            dist_loss = 255**2 * out["char_loss"]
            
        elif self.loss_type == "ms_ssim":
            if ms_ssim is None: 
                raise ImportError("pytorch_msssim not installed. Install it or use 'mse'.")
            # MS-SSIM is max 1.0, so loss is 1 - MS-SSIM
            out["ms_ssim_loss"] = 1 - ms_ssim(x_hat, target, data_range=1.0)
            dist_loss = 255**2 * out["ms_ssim_loss"] 
            
        else:
            raise ValueError(f"Unknown metric: {self.loss_type}")

        out["dist_loss"] = dist_loss

        # --- 3. Auxiliary Loss (MoE Balancing) ---
        total_aux = 0.0
        
        # We calculate this if alpha > 0, either for logging or for gradient
        if self.alpha_moe > 0:
            moe_loss = 0.0
            if "router_logits" in output and output["router_logits"] is not None:
                # This function now handles the Tuple(logits, indices) efficiently
                moe_loss = load_balancing_loss_func(
                    output["router_logits"], 
                    num_experts=self.num_experts, 
                    top_k=self.top_k
                )
            
            out["moe_loss"] = moe_loss
            
            # --- CRITICAL LOGIC FOR LOSS-FREE BALANCING ---
            if self.use_loss_free_balancing:
                # 1. "Loss-Free": We do NOT add this to the gradient.
                #    The balancing happens via Bias Updates in the optimizer loop.
                #    We record it in 'out' only for Tensorboard logging.
                pass 
            else:
                # 2. "Standard": Add to gradient to force Softmax weights to balance.
                total_aux += self.alpha_moe * moe_loss

        out["aux_loss"] = total_aux

        # --- 4. Total Loss ---
        # Loss = Rate + Lambda * (Distortion + Aux)
        out["loss"] = out["bpp_loss"] + self.lmbda * (dist_loss + total_aux)

        return out