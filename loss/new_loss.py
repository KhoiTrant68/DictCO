import math
from typing import List, Tuple, Union

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

class RateDistortionLoss(nn.Module):
    """
    Hàm Loss tối ưu hóa Tỷ lệ-Biến dạng (RD Loss).
    Tích hợp cơ chế Loss-Free Balancing theo SOTA của DeepSeek.
    """
    def __init__(
        self,
        lmbda=1e-2,
        loss_type="mse",
        num_experts=4,
        use_loss_free_balancing=True,
    ):
        super().__init__()
        self.lmbda = lmbda
        self.loss_type = loss_type
        self.num_experts = num_experts
        self.use_loss_free_balancing = use_loss_free_balancing
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        """
        output: Dictionary từ model DCAE
        target: Ảnh gốc [0, 1]
        """
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}

        # --- 1. RATE LOSS (BPP) ---
        # Tính toán tổng số bit dựa trên likelihoods (y và z)
        # Sử dụng clamp để tránh log(0) gây ra lỗi NaN
        total_bits = 0
        for name, likelihood in output["likelihoods"].items():
            if isinstance(likelihood, (list, tuple)):
                for l in likelihood:
                    total_bits += torch.log(l.clamp(min=1e-9)).sum()
            else:
                total_bits += torch.log(likelihood.clamp(min=1e-9)).sum()

        # BPP = -log2(likelihood) / pixels
        out["bpp_loss"] = -total_bits / (math.log(2) * num_pixels)

        # --- 2. DISTORTION LOSS ---
        x_hat = output["x_hat"].clamp(0, 1)

        if self.loss_type == "mse":
            out["mse_loss"] = self.mse(x_hat, target)
            # Scaling theo CompressAI (255^2) để cân bằng với BPP
            dist_loss = 255**2 * out["mse_loss"]

        elif self.loss_type == "ms_ssim":
            if ms_ssim is None:
                raise ImportError("Cần cài đặt pytorch_msssim để dùng ms_ssim loss.")
            # MS-SSIM max là 1, nên loss là 1 - MS-SSIM
            out["ms_ssim_loss"] = 1 - ms_ssim(x_hat, target, data_range=1.0)
            dist_loss = out["ms_ssim_loss"] # Thường dùng lambda lớn hơn cho SSIM (ví dụ: 1-100)

        out["dist_loss"] = dist_loss

        # --- 3. MOE MONITORING (Loss-Free) ---
        # Tính toán chỉ số mất cân bằng để theo dõi trên Tensorboard
        # Không tham gia vào quá trình lan truyền ngược (no_grad)
        if "router_logits" in output and output["router_logits"] is not None:
            with torch.no_grad():
                out["moe_imbalance"] = self._calculate_imbalance(output["router_logits"])

        # --- 4. TOTAL LOSS ---
        # Theo đúng paper Loss-Free Balancing: Không cộng auxiliary loss vào đây.
        # Biến dạng (Distortion) được nhân với lambda.
        out["loss"] = out["bpp_loss"] + self.lmbda * dist_loss

        return out

    def _calculate_imbalance(self, router_data):
        """Tính toán độ lệch tải của các Expert (0.0 là hoàn hảo)."""
        total_imbalance = 0.0
        count = 0
        
        # router_data thường là list của (logits, indices) từ các slice
        for item in router_data:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                indices = item[1] # [B, H, W, K]
                
                # Đếm tần suất xuất hiện của mỗi expert
                expert_counts = torch.bincount(
                    indices.flatten(), minlength=self.num_experts
                ).float()
                
                # Tính độ lệch chuẩn so với trung bình
                avg_load = expert_counts.mean()
                if avg_load > 0:
                    # Độ lệch tuyệt đối trung bình
                    imbalance = (expert_counts - avg_load).abs().mean() / avg_load
                    total_imbalance += imbalance
                    count += 1
                    
        return total_imbalance / count if count > 0 else 0.0

class CharbonnierLoss(nn.Module):
    """L1 Loss mượt hóa, tốt cho việc khôi phục chi tiết ảnh."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y)**2 + self.eps**2))