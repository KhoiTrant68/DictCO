import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class WMSA(nn.Module):
    """
    Optimized Window Multi-head Self-attention module.
    Key improvements: Pre-computed relative position indices (No CPU-GPU sync in forward).
    """

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
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

    def generate_mask(self, h, w, p, shift):
        """
        Optimized mask generation.
        """
        # Note: In a production setting, you might cache this based on (h, w) keys
        # to avoid reallocation if input size is constant.
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
        """
        Forward pass with optimized relative position lookup.
        """
        B, H, W, C = x.shape

        # Cyclic Shift
        if self.type != "W":
            x = torch.roll(
                x,
                shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                dims=(1, 2),
            )

        # Partition Windows
        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c",
            p1=self.window_size,
            p2=self.window_size,
        )
        h_windows = x.size(1)
        w_windows = x.size(2)

        # Flatten windows
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

        # Attention
        # q: [b, nw, np, c], k: [b, nw, np, c] -> sim: [b, nw, np, np]
        # We need to handle heads 'h'. The original einsum was "hbwpc,hbwqc->hbwpq"
        # Assuming shape coming out of embedding is compatible.

        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale

        # Add Relative Position Bias (Optimized Lookup)
        relative_position_bias = self.relative_position_params[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww

        sim = sim + relative_position_bias.unsqueeze(1).unsqueeze(1)  # Broadcast over batch

        if self.type != "W":
            attn_mask = self.generate_mask(
                h_windows, w_windows, self.window_size, shift=self.window_size // 2
            )
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)

        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")
        output = self.linear(output)

        # Merge windows
        output = rearrange(
            output,
            "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c",
            w1=h_windows,
            p1=self.window_size,
        )

        # Reverse Cyclic Shift
        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )
        return output


class DWConv(nn.Module):
    def __init__(self, dim=768):
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
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
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


class ResScaleConvGateBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        head_dim,
        window_size,
        drop_path,
        type="W",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ["W", "SW"]
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
        **kwargs,
    ) -> None:
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
        # x input is [B, C, H, W]
        # Padding Logic
        H, W = x.size(2), x.size(3)
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size

        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))

        # Permute once for Transformer Blocks: [B, C, H, W] -> [B, H, W, C]
        trans_x = x.permute(0, 2, 3, 1)

        for layer in self.layers:
            trans_x = layer(trans_x)

        # Permute back: [B, H, W, C] -> [B, C, H, W]
        trans_x = trans_x.permute(0, 3, 1, 2)
        trans_x = self.conv(trans_x)

        # Remove Padding
        if pad_b > 0 or pad_r > 0:
            trans_x = trans_x[:, :, :H, :W]

        return trans_x + x[:, :, :H, :W]


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x: [B, C, H, W]
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
    """

    Standard Dense Block logic.
    """

    def __init__(self, dim=320):
        super(DenseBlock, self).__init__()
        self.layer_num = 3
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GELU(),
                    ConvWithDW(dim, dim),
                )
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
        # x in: [B, H, W, C] (Based on usage in CrossAttention)
        x = x.permute(0, 3, 1, 2)  # -> [B, C, H, W]

        s = self.s(x)
        s_out = self.dense(s)
        x = s_out * self.spatial_atte(s_out)

        x = x.permute(0, 2, 3, 1)  # -> [B, H, W, C]
        return x


class MutiScaleDictionaryCrossAttentionGELU(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_rate=4, head_num=20, qkv_bias=True):
        super().__init__()

        dict_dim = 32 * head_num
        self.head_num = head_num

        self.scale = nn.Parameter(torch.ones(head_num, 1, 1))
        self.x_trans = nn.Linear(input_dim, dict_dim, bias=qkv_bias)

        self.ln_scale = nn.LayerNorm(dict_dim)
        self.msa = MultiScaleAggregation(dict_dim)

        self.lnx = nn.LayerNorm(dict_dim)
        self.q_trans = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.dict_ln = nn.LayerNorm(dict_dim)
        self.k = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)

        self.linear = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.ln_mlp = nn.LayerNorm(dict_dim)

        self.mlp = ConvGELU(dict_dim, mlp_rate * dict_dim)
        self.output_trans = nn.Linear(dict_dim, output_dim)

        self.res_scale_1 = Scale(dict_dim, init_value=1.0)
        self.res_scale_2 = Scale(dict_dim, init_value=1.0)
        self.res_scale_3 = Scale(dict_dim, init_value=1.0)

    def forward(self, x, dt):
        # x: [B, C, H, W]
        _, _, H, W = x.size()
        x = x.permute(0, 2, 3, 1)  # -> [B, H, W, C]

        x = self.x_trans(x)

        # MSA and Shortcut
        x = self.msa(self.ln_scale(x)) + self.res_scale_1(x)
        shortcut = x

        x = self.lnx(x)
        x = self.q_trans(x)

        # Prepare Q
        #
        # Reshape to [B, Heads, N, C_head]
        q = rearrange(x, "b h w (e c) -> b e (h w) c", e=self.head_num)

        # Prepare K, V (dt acts as both K and V source here usually, or K specifically)
        dt = self.dict_ln(dt)
        k = self.k(dt)
        k = rearrange(k, "b n (e c) -> b e n c", e=self.head_num)
        dt_val = rearrange(dt, "b n (e c) -> b e n c", e=self.head_num)

        # Cross Attention
        # self.scale is automatically on correct device because it is nn.Parameter
        sim = torch.einsum("benc,bedc->bend", q, k) * self.scale
        probs = F.softmax(sim, dim=-1)

        output = torch.einsum("bend,bedc->benc", probs, dt_val)
        output = rearrange(output, "b e (h w) c -> b h w (e c) ", h=H, w=W)

        # Residuals and MLP
        output = self.linear(output) + self.res_scale_2(shortcut)
        output = self.mlp(self.ln_mlp(output)) + self.res_scale_3(output)

        output = self.output_trans(output)

        # Final permute back to [B, C, H, W]
        output = output.permute(0, 3, 1, 2)
        return output
    
class MoEDictionaryCrossAttention(nn.Module):
    """
    Mixture-of-Dictionaries Cross Attention.
    Replaces the single global dictionary with K specialized expert dictionaries.
    Includes a Spatial Router to dynamically weight experts based on local image content.
    """
    def __init__(self, input_dim, output_dim, mlp_rate=4, head_num=20, qkv_bias=True, 
                 num_experts=4, expert_entries=64):
        super().__init__()

        dict_dim = 32 * head_num
        self.head_num = head_num
        self.num_experts = num_experts
        
        # --- MoE Specifics ---
        # Bank of Dictionaries: [K, N, C]
        # We initialize them with the same distribution as the original paper
        # K = num_experts, N = expert_entries, C = dict_dim
        self.experts = nn.Parameter(torch.randn(num_experts, expert_entries, dict_dim))
        
        # The Router: Predicts which expert to use per spatial location
        # Input: dict_dim (after MSFA), Output: num_experts
        self.router = nn.Sequential(
            nn.Linear(dict_dim, dict_dim // 4),
            nn.ReLU(),
            nn.Linear(dict_dim // 4, num_experts)
        )
        # ---------------------

        self.scale = nn.Parameter(torch.ones(head_num, 1, 1))
        self.x_trans = nn.Linear(input_dim, dict_dim, bias=qkv_bias)

        self.ln_scale = nn.LayerNorm(dict_dim)
        self.msa = MultiScaleAggregation(dict_dim) # Reusing your existing MSFA

        self.lnx = nn.LayerNorm(dict_dim)
        self.q_trans = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        
        # Key Projection (Shared across experts to save params, or separate)
        # Here we use shared projection for stability
        self.dict_ln = nn.LayerNorm(dict_dim)
        self.k_trans = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)

        self.linear = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.ln_mlp = nn.LayerNorm(dict_dim)

        self.mlp = ConvGELU(dict_dim, mlp_rate * dict_dim)
        self.output_trans = nn.Linear(dict_dim, output_dim)

        self.res_scale_1 = Scale(dict_dim, init_value=1.0)
        self.res_scale_2 = Scale(dict_dim, init_value=1.0)
        self.res_scale_3 = Scale(dict_dim, init_value=1.0)

    def forward(self, x, dt=None):
        # Note: 'dt' argument is kept for compatibility with existing training loop calls,
        # but we IGNORE it in favor of self.experts.
        
        # x: [B, C, H, W]
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1)  # -> [B, H, W, C]

        x = self.x_trans(x)

        # 1. Multi-Scale Feature Aggregation
        # This extracts local texture context used for both Querying and Routing
        x = self.msa(self.ln_scale(x)) + self.res_scale_1(x)
        shortcut = x

        # 2. Spatial Routing
        # routing_logits: [B, H, W, K]
        routing_logits = self.router(x) 
        routing_weights = F.softmax(routing_logits, dim=-1)

        # 3. Prepare Query
        x = self.lnx(x)
        x = self.q_trans(x)
        # Q: [B, Heads, (H*W), C_head]
        # We use 's' to represent the spatial dimension (H*W)
        q = rearrange(x, "b h w (e c) -> b e (h w) c", e=self.head_num)

        # 4. Expert Attention Loop
        # We accumulate the weighted results from all experts
        final_context = 0
        
        for k in range(self.num_experts):
            # Get k-th Expert Dictionary: [N, C]
            expert_dict = self.experts[k]
            
            # Prepare Key and Value for this expert
            expert_dict = self.dict_ln(expert_dict)
            k_expert = self.k_trans(expert_dict) # Key
            v_expert = expert_dict               # Value (Dictionary itself)

            # Reshape for Multi-Head Attention
            # K, V: [1, Heads, DictSize, C_head] (Broadcasting over Batch)
            # We use 'd' to represent the Dictionary Size dimension
            k_expert = rearrange(k_expert, "d (e c) -> 1 e d c", e=self.head_num)
            v_expert = rearrange(v_expert, "d (e c) -> 1 e d c", e=self.head_num)

            # Attention: Query [B, E, Spatial, C] vs Key [1, E, Dict, C]
            # Use 's' for Spatial (h*w) and 'd' for Dictionary size
            # sim shape: [B, E, Spatial, Dict]
            # 'k' index handles the broadcasting of the singleton batch dim of the expert
            sim = torch.einsum("besc,kedc->besd", q, k_expert) * self.scale
            probs = F.softmax(sim, dim=-1)

            # Aggregate Values
            # probs: [B, E, Spatial, Dict]
            # v_expert: [1, E, Dict, C]
            # Output: [B, E, Spatial, C]
            expert_out = torch.einsum("besd,kedc->besc", probs, v_expert)
            
            # Reshape back to Spatial: [B, H, W, C_total]
            expert_out = rearrange(expert_out, "b e (h w) c -> b h w (e c)", h=H, w=W)
            
            # 5. Apply Routing Weight
            # Weight: [B, H, W, 1] (Select k-th weight)
            gate = routing_weights[:, :, :, k].unsqueeze(-1)
            
            final_context = final_context + (expert_out * gate)

        # 6. Post-Processing (Residuals + MLP)
        output = self.linear(final_context) + self.res_scale_2(shortcut)
        output = self.mlp(self.ln_mlp(output)) + self.res_scale_3(output)

        output = self.output_trans(output)

        # Permute back to [B, C, H, W]
        output = output.permute(0, 3, 1, 2)
        return output