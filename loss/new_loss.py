import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple

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
#  YOUR LOAD BALANCING FUNCTION (Adapted for Image Compression)
# ==============================================================================
def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor]], 
    num_experts: int, 
    top_k: int = 2, 
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Computes auxiliary load balancing loss (Switch Transformer style).
    Adapted to handle Image Compression tensors (B, H, W, Experts).
    """
    # If input is a single tensor (one MoE layer), wrap it in a tuple/list
    if isinstance(gate_logits, torch.Tensor):
        gate_logits = [gate_logits]
        
    if gate_logits is None or len(gate_logits) == 0:
        return torch.tensor(0.0)

    # Assume all layers are on the same device
    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
    else: # List
        compute_device = gate_logits[0].device

    layer_aux_loss = []
    
    for layer_gate in gate_logits:
        # layer_gate shape usually: [Batch, H, W, NumExperts] or [Batch, SeqLen, NumExperts]
        # We need to flatten batch and spatial dimensions for calculation
        if layer_gate.dim() == 4: # Image case: (B, H, W, E)
            layer_gate = layer_gate.view(-1, num_experts)
        elif layer_gate.dim() == 3: # Seq case: (B, L, E)
            layer_gate = layer_gate.view(-1, num_experts)
            
        # 1. Softmax to get probabilities
        routing_weights = torch.nn.functional.softmax(layer_gate, dim=-1)

        # 2. Get Top-K selections
        # Note: If top_k is larger than num_experts, clamp it
        k = min(top_k, num_experts)
        _, selected_experts = torch.topk(routing_weights, k, dim=-1)

        # 3. Create Mask (One-hot encoding of selected experts)
        # shape: [TotalTokens, k, NumExperts]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

        if attention_mask is None:
            # Flatten expert mask to [TotalTokens, NumExperts]
            # Since a token can select k experts, we sum the one-hots
            # (or mean depending on specific implementation, Switch Transformer usually sums for load)
            tokens_per_expert = torch.mean(torch.sum(expert_mask.float(), dim=1), dim=0)

            # Average probability routed to each expert
            router_prob_per_expert = torch.mean(routing_weights, dim=0)
        else:
            # Handle attention mask if needed (usually None for Image Compression)
            # Flatten mask to [TotalTokens]
            flat_mask = attention_mask.view(-1).to(compute_device)
            
            # Mask out padding tokens
            valid_tokens = torch.sum(flat_mask)
            
            # Weighted mean for tokens routed
            # Sum over top-k selections
            expert_selection_flat = torch.sum(expert_mask.float(), dim=1) 
            tokens_per_expert = torch.sum(expert_selection_flat * flat_mask.unsqueeze(1), dim=0) / (valid_tokens + 1e-6)

            # Weighted mean for router probs
            router_prob_per_expert = torch.sum(routing_weights * flat_mask.unsqueeze(1), dim=0) / (valid_tokens + 1e-6)

        # 4. Compute Loss: N * sum(f_i * P_i)
        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert) * num_experts
        layer_aux_loss.append(overall_loss)
    
    if len(layer_aux_loss) == 0:
        return torch.tensor(0.0, device=compute_device)
        
    return sum(layer_aux_loss)

# ==============================================================================
#  MAIN LOSS MODULE
# ==============================================================================
class RateDistortionLoss(nn.Module):
    """
    Main Objective: R + lambda * D + Aux_Loss(Spectral + MoE)
    """
    def __init__(self, lmbda=1e-2, loss_type="mse", 
                 alpha_spectral=0.1, alpha_moe=0.01,
                 num_experts=4, top_k=2):
        super().__init__()
        self.lmbda = lmbda
        self.loss_type = loss_type
        self.alpha_spectral = alpha_spectral
        self.alpha_moe = alpha_moe
        
        # MoE specific params
        self.num_experts = num_experts
        self.top_k = top_k

        # Sub-losses
        self.mse = nn.MSELoss()
        self.charbonnier = CharbonnierLoss()
        self.ffl = FocalFrequencyLoss()

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}

        # --- 1. Rate Loss (BPP) ---
        y_lik = output["likelihoods"]["y"]
        # Handle single tensor or list (from uneven slicing)
        if isinstance(y_lik, list) or isinstance(y_lik, tuple):
            # Sum log-likelihoods across all slices
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
        # Check if model returned 'router_logits' (Preferred for this loss func)
        if self.alpha_moe > 0:
            # In dcae_final.py, ensure you return 'router_logits' in the output dict
            # If your model returns 'aux_loss' directly, use that. 
            # If it returns logits, calculate it here.
            
            if "router_logits" in output:
                moe_loss = load_balancing_loss_func(
                    output["router_logits"], 
                    num_experts=self.num_experts, 
                    top_k=self.top_k
                )
                out["moe_loss"] = moe_loss
                total_aux += self.alpha_moe * moe_loss
            elif "aux_loss" in output:
                # Fallback if model calculated it internally
                out["moe_loss"] = output["aux_loss"]
                total_aux += output["aux_loss"] # alpha already applied in model? check logic

        out["aux_loss"] = total_aux

        # --- 4. Total Loss ---
        out["loss"] = out["bpp_loss"] + self.lmbda * (dist_loss + total_aux)

        return out