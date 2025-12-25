import math
import torch
import torch.nn as nn

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

class RateDistortionLoss(nn.Module):
    """
    Custom Rate-Distortion loss module.
    Optimized for Loss-Free Balancing.
    """
    def __init__(
        self,
        lmbda=1e-2,
        loss_type="mse",
        num_experts=4,
        use_loss_free_balancing=True,
        return_type="all", # "all" or "total_only"
    ):
        super().__init__()
        self.lmbda = lmbda
        self.loss_type = loss_type.lower().replace("-", "_")
        self.num_experts = num_experts
        self.use_loss_free_balancing = use_loss_free_balancing
        self.return_type = return_type

        self.mse = nn.MSELoss()
        self.charbonnier = CharbonnierLoss()

    def forward(self, output, target):
        """
        output: Dictionary from the DCAE model
        target: Original ground truth image [0, 1]
        """
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}

        # --- 1. RATE LOSS (BPP) ---
        # Calculate bits based on likelihoods for y and z latents
        total_bits = 0
        
        # Robust handling for list or tensor likelihoods
        for name, likelihood in output["likelihoods"].items():
            if isinstance(likelihood, (list, tuple)):
                # If uneven slicing returns a list of tensors
                for l in likelihood:
                    total_bits += torch.log(l.clamp(min=1e-9)).sum()
            else:
                # Standard case
                total_bits += torch.log(likelihood.clamp(min=1e-9)).sum()

        # BPP = -log2(likelihood) / num_pixels
        out["bpp_loss"] = -total_bits / (math.log(2) * num_pixels)

        # --- 2. DISTORTION LOSS ---
        x_hat = output["x_hat"].clamp(0, 1)
        
        if self.loss_type == "mse":
            mse_val = self.mse(x_hat, target)
            out["mse_loss"] = mse_val
            # Standard CompressAI scaling: 255^2 allows lambda ~ 0.01
            dist_loss = 255**2 * mse_val

        elif self.loss_type == "charbonnier":
            char_val = self.charbonnier(x_hat, target)
            out["char_loss"] = char_val
            # Scaling to match MSE magnitude
            dist_loss = 255**2 * char_val

        elif self.loss_type == "ms_ssim":
            if ms_ssim is None:
                raise ImportError("pytorch_msssim not installed.")
            
            # MS-SSIM ranges 0 to 1. Loss is 1 - MS-SSIM.
            ms_ssim_val = ms_ssim(x_hat, target, data_range=1.0)
            out["ms_ssim_loss"] = 1 - ms_ssim_val
            
            # CRITICAL FIX: Unscaled MS-SSIM is too small for standard lambda.
            # We do NOT auto-scale here to strictly follow literature definition,
            # BUT you must ensure args.lmbda is appropriate (e.g. > 10.0).
            dist_loss = out["ms_ssim_loss"] 

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        out["dist_loss"] = dist_loss

        # --- 3. MOE MONITORING (Loss-Free) ---
        # We calculate this for TensorBoard, but do NOT add to loss
        if "router_logits" in output and output["router_logits"] is not None:
            # Detach to ensure absolutely no gradient leakage from this calculation
            with torch.no_grad():
                out["moe_imbalance"] = self._calculate_imbalance(output["router_logits"])
        else:
            out["moe_imbalance"] = torch.tensor(0.0, device=target.device)

        # --- 4. TOTAL LOSS ---
        # R + Lambda * D
        out["loss"] = out["bpp_loss"] + self.lmbda * dist_loss

        return out

    def _calculate_imbalance(self, router_data):
        """
        Calculates Load Imbalance for monitoring.
        Metric: Normalized Standard Deviation of expert counts.
        """
        total_imbalance = 0.0
        count = 0
        
        for layer_data in router_data:
            # DCAE returns tuple: (logits, indices)
            if isinstance(layer_data, (tuple, list)) and len(layer_data) == 2:
                indices = layer_data[1] # [B, H, W, K]
                
                # Flatten to count
                flat_indices = indices.flatten()
                
                if flat_indices.numel() == 0:
                    continue

                expert_counts = torch.bincount(
                    flat_indices, minlength=self.num_experts
                ).float()
                
                # Ideal load per expert
                target_load = flat_indices.numel() / self.num_experts
                
                # Imbalance metric: Mean Absolute Deviation from Target
                if target_load > 0:
                    imbalance = (expert_counts - target_load).abs().mean() / target_load
                    total_imbalance += imbalance
                    count += 1
                    
        return total_imbalance / count if count > 0 else torch.tensor(0.0)