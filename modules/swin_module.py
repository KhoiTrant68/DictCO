import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

# ==========================================
# PART 1: STANDARD SWIN & HELPER BLOCKS
# (Kept unchanged for stability)
# ==========================================


class WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        )
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        trunc_normal_(self.relative_position_params, std=0.02)

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(
            h,
            w,
            p,
            p,
            p,
            p,
            dtype=torch.bool,
            device=self.relative_position_params.device,
        )
        if self.type == "W":
            return attn_mask
        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(
            attn_mask, "w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)"
        )
        return attn_mask

    def forward(self, x):
        if self.type != "W":
            x = torch.roll(
                x,
                shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                dims=(1, 2),
            )
        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c",
            p1=self.window_size,
            p2=self.window_size,
        )
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(
            x,
            "b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )

        qkv = self.embedding_layer(x)
        q, k, v = rearrange(
            qkv, "b nw np (threeh c) -> threeh b nw np c", c=self.head_dim
        ).chunk(3, dim=0)

        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        relative_position_bias = self.relative_position_params[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        sim = sim + relative_position_bias.unsqueeze(1).unsqueeze(1)

        if self.type != "W":
            attn_mask = self.generate_mask(
                h_windows, w_windows, self.window_size, shift=self.window_size // 2
            )
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")
        output = self.linear(output)
        output = rearrange(
            output,
            "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c",
            w1=h_windows,
            p1=self.window_size,
        )

        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )
        return output


class ResScaleConvGateBlock(nn.Module):
    def __init__(
        self, input_dim, output_dim, head_dim, window_size, drop_path, type="W"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = ConvGELU(input_dim, input_dim * 4)
        self.res_scale_1 = Scale(input_dim, init_value=1.0)
        self.res_scale_2 = Scale(input_dim, init_value=1.0)

    def forward(self, x):
        x = self.res_scale_1(x) + self.drop_path(self.msa(self.ln1(x)))
        x = self.res_scale_2(x) + self.drop_path(self.mlp(self.ln2(x)))
        return x


class SwinBlockWithConvMulti(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        head_dim,
        window_size,
        drop_path,
        block=ResScaleConvGateBlock,
        block_num=2,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.block_num = block_num
        for i in range(block_num):
            ty = "W" if i % 2 == 0 else "SW"
            self.layers.append(
                block(input_dim, input_dim, head_dim, window_size, drop_path, type=ty)
            )
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.window_size = window_size

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        trans_x = x.permute(0, 2, 3, 1)
        for layer in self.layers:
            trans_x = layer(trans_x)
        trans_x = trans_x.permute(0, 3, 1, 2)
        trans_x = self.conv(trans_x)
        if pad_b > 0 or pad_r > 0:
            trans_x = trans_x[:, :, :H, :W]
        return trans_x + x[:, :, :H, :W]


class DWConv(nn.Module):
    def __init__(self, dim=128):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class ConvGELU(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = hidden_features // 2
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.dwconv(x)
        x = self.act(x) * v
        x = self.fc2(x)
        return x


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvWithDW(nn.Module):
    def __init__(self, input_dim=320, output_dim=320):
        super(ConvWithDW, self).__init__()
        self.in_trans = nn.Conv2d(input_dim, output_dim, 1)
        self.act1 = nn.GELU()
        self.dw_conv = nn.Conv2d(
            output_dim, output_dim, 3, padding=1, groups=output_dim
        )
        self.act2 = nn.GELU()
        self.out_trans = nn.Conv2d(output_dim, output_dim, 1)

    def forward(self, x):
        x = self.in_trans(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.act2(x)
        x = self.out_trans(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, dim=320):
        super(DenseBlock, self).__init__()
        self.layer_num = 3
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(nn.GELU(), ConvWithDW(dim, dim))
                for _ in range(self.layer_num)
            ]
        )
        self.proj = nn.Conv2d(dim * (self.layer_num + 1), dim, 1)

    def forward(self, x):
        outputs = [x]
        for layer in self.conv_layers:
            outputs.append(layer(outputs[-1]))
        x = self.proj(torch.cat(outputs, dim=1))
        return x


class MultiScaleAggregation(nn.Module):
    def __init__(self, dim):
        super(MultiScaleAggregation, self).__init__()
        self.s = nn.Conv2d(dim, dim, 1)
        self.spatial_atte = SpatialAttentionModule()
        self.dense = DenseBlock(dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        s = self.s(x)
        s_out = self.dense(s)
        x = s_out * self.spatial_atte(s_out)
        x = x.permute(0, 2, 3, 1)
        return x


# ==========================================
# PART 2: OPTIMIZED SPECTRAL MOE CLASSES
# ==========================================


class HaarWaveletTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        ll_filt = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        lh_filt = torch.tensor([[-0.5, -0.5], [0.5, 0.5]]).view(1, 1, 2, 2)
        hl_filt = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)
        hh_filt = torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2)

        filters = torch.cat([ll_filt, lh_filt, hl_filt, hh_filt], dim=0)
        self.register_buffer("filters", filters.repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        # NOTE: Padding is now handled inside the Spectral Attention class
        out = F.conv2d(x, self.filters, stride=2, groups=self.in_channels)
        B, _, H, W = out.shape
        out = out.view(B, self.in_channels, 4, H, W)
        LL = out[:, :, 0, :, :]
        HF = out[:, :, 1:, :, :].reshape(B, self.in_channels * 3, H, W)
        return LL, HF


class InverseHaarWaveletTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        ll_filt = torch.tensor([[1.0, 1.0], [1.0, 1.0]]).view(1, 1, 2, 2)
        lh_filt = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]).view(1, 1, 2, 2)
        hl_filt = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]).view(1, 1, 2, 2)
        hh_filt = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).view(1, 1, 2, 2)
        filters = torch.cat([ll_filt, lh_filt, hl_filt, hh_filt], dim=0)
        self.register_buffer("filters", filters.repeat(in_channels, 1, 1, 1))

    def forward(self, ll, hf):
        B, C, H, W = ll.shape
        hf = hf.view(B, C, 3, H, W)
        stacked = torch.cat([ll.unsqueeze(2), hf], dim=2)
        stacked = stacked.view(B, 4 * C, H, W)
        out = F.conv_transpose2d(
            stacked, self.filters, stride=2, groups=self.in_channels
        )
        return out


class SpectralMoEDictionaryCrossAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        mlp_rate=4,
        head_num=4,
        qkv_bias=True,
        num_experts=4,
        expert_entries=64,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.head_num = head_num
        self.num_experts = num_experts
        self.expert_entries = expert_entries

        # Dimensions
        self.dim_low = 32 * head_num
        self.dim_high = 32 * head_num

        self.dwt = HaarWaveletTransform(self.dim_low)
        self.idwt = InverseHaarWaveletTransform(self.dim_low)

        # === 1. LOW FREQ PATH (Shared Dictionary) ===
        self.dict_low = nn.Parameter(torch.randn(64, self.dim_low))
        self.ln_low = nn.LayerNorm(self.dim_low)
        self.q_low = nn.Linear(self.dim_low, self.dim_low, bias=qkv_bias)
        self.k_low = nn.Linear(self.dim_low, self.dim_low, bias=qkv_bias)
        self.ln_dict_low = nn.LayerNorm(self.dim_low)

        # === 2. HIGH FREQ PATH (Optimized MoE) ===
        # Stack experts into one large tensor: [K * N, C]
        # This allows parallel processing instead of looping
        self.experts_high = nn.Parameter(
            torch.randn(num_experts * expert_entries, self.dim_high)
        )

        self.router = nn.Sequential(
            nn.Linear(self.dim_high, self.dim_high // 4),
            nn.ReLU(),
            nn.Linear(self.dim_high // 4, num_experts),
        )

        self.proj_hf_in = nn.Linear(self.dim_low * 3, self.dim_high)
        self.proj_hf_out = nn.Linear(self.dim_high, self.dim_low * 3)

        self.ln_high = nn.LayerNorm(self.dim_high)
        self.q_high = nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias)
        self.k_high = nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias)
        self.ln_dict_high = nn.LayerNorm(self.dim_high)

        # === Common Utils ===
        self.x_trans = nn.Linear(input_dim, self.dim_low, bias=qkv_bias)
        self.msa = MultiScaleAggregation(self.dim_low)
        self.ln_scale = nn.LayerNorm(self.dim_low)
        self.res_scale_1 = Scale(self.dim_low, init_value=1.0)
        self.scale = nn.Parameter(torch.ones(head_num, 1, 1))

        self.linear = nn.Linear(self.dim_low, self.dim_low, bias=qkv_bias)
        self.ln_mlp = nn.LayerNorm(self.dim_low)
        self.mlp = ConvGELU(self.dim_low, mlp_rate * self.dim_low)
        self.res_scale_2 = Scale(self.dim_low, init_value=1.0)
        self.res_scale_3 = Scale(self.dim_low, init_value=1.0)
        self.output_trans = nn.Linear(self.dim_low, output_dim)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.dict_low, std=0.02)
        trunc_normal_(self.experts_high, std=0.02)
        trunc_normal_(self.scale, std=0.02)

    def process_low_freq(self, x):
        # Standard Cross Attention
        B, H, W, C = x.shape
        x_norm = self.ln_low(x)
        q = self.q_low(x_norm)
        q = rearrange(q, "b h w (e c) -> b e (h w) c", e=self.head_num)

        dict_norm = self.ln_dict_low(self.dict_low)
        k = self.k_low(dict_norm)
        v = dict_norm
        k = rearrange(k, "n (e c) -> 1 e n c", e=self.head_num)
        v = rearrange(v, "n (e c) -> 1 e n c", e=self.head_num)

        sim = torch.einsum("besc,kenc->besn", q, k) * self.scale
        probs = F.softmax(sim, dim=-1)
        out = torch.einsum("besn,kenc->besc", probs, v)
        return rearrange(out, "b e (h w) c -> b h w (e c)", h=H, w=W)

    def process_high_freq_moe_vectorized(self, x):
        """
        OPTIMIZED MoE: No Python Loops.
        """
        B, H, W, C = x.shape

        # 1. Routing [B, H, W, K]
        routing_logits = self.router(x)
        routing_weights = F.softmax(routing_logits, dim=-1)

        # 2. Prepare Query
        x_norm = self.ln_high(x)
        q = self.q_high(x_norm)
        q = rearrange(q, "b h w (e c) -> b e (h w) c", e=self.head_num)

        # 3. Prepare All Experts (Batched)
        # experts_high is [K * N, C]
        all_experts_norm = self.ln_dict_high(self.experts_high)
        k_all = self.k_high(all_experts_norm)
        v_all = all_experts_norm

        # Reshape to [1, E, Total_N, C_head]
        k_all = rearrange(k_all, "n (e c) -> 1 e n c", e=self.head_num)
        v_all = rearrange(v_all, "n (e c) -> 1 e n c", e=self.head_num)

        # 4. Global Attention (Computes all expert interactions at once)
        # sim: [B, E, S, Total_N]
        sim = torch.einsum("besc,kenc->besn", q, k_all) * self.scale
        probs = F.softmax(sim, dim=-1)

        # 5. Apply MoE Masking
        # Reshape probs: [B, E, S, K, N_per_expert]
        probs = rearrange(probs, "b e s (k n) -> b e s k n", k=self.num_experts)

        # Reshape router weights to match: [B, 1, S, K, 1]
        router_mask = rearrange(routing_weights, "b h w k -> b 1 (h w) k 1")

        # Weighting: experts with low router probability get zeroed out
        weighted_probs = probs * router_mask

        # Flatten back for value aggregation: [B, E, S, Total_N]
        weighted_probs = rearrange(weighted_probs, "b e s k n -> b e s (k n)")

        # 6. Value Aggregation
        out = torch.einsum("besn,kenc->besc", weighted_probs, v_all)
        return rearrange(out, "b e (h w) c -> b h w (e c)", h=H, w=W)

    def forward(self, x):
        # Input: [B, C, H, W]
        B, C, H, W = x.size()

        # --- ROBUST PADDING (Fixes crash on odd resolutions) ---
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = x.permute(0, 2, 3, 1)  # [B, H_pad, W_pad, C]

        # 1. Init & MSFA
        x = self.x_trans(x)
        x = self.msa(self.ln_scale(x)) + self.res_scale_1(x)
        shortcut = x

        # 2. Wavelet Split
        x_perm = x.permute(0, 3, 1, 2)
        ll, hf = self.dwt(x_perm)

        # 3. Low Freq Path
        ll = ll.permute(0, 2, 3, 1)
        ll_att = self.process_low_freq(ll)
        ll_out = ll + ll_att
        ll_out = ll_out.permute(0, 3, 1, 2)

        # 4. High Freq Path (Vectorized MoE)
        hf = hf.permute(0, 2, 3, 1)
        hf_proj = self.proj_hf_in(hf)
        hf_att = self.process_high_freq_moe_vectorized(
            hf_proj
        )  # <--- Using Optimized Func
        hf_att = self.proj_hf_out(hf_att)
        hf_out = hf + hf_att
        hf_out = hf_out.permute(0, 3, 1, 2)

        # 5. Inverse Wavelet
        x_recon = self.idwt(ll_out, hf_out)
        x_recon = x_recon.permute(0, 2, 3, 1)

        # 6. Output & Residual
        output = self.linear(x_recon) + self.res_scale_2(shortcut)
        output = self.mlp(self.ln_mlp(output)) + self.res_scale_3(output)
        output = self.output_trans(output)
        output = output.permute(0, 3, 1, 2)

        # --- CROP BACK ---
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :H, :W]

        return output
