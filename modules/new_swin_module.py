import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

# ==========================================
# PART 1: STANDARD BLOCKS & HELPERS
# ==========================================

class WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
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
            h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device
        )
        if self.type == "W":
            return attn_mask
        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, "w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)")
        return attn_mask

    def forward(self, x):
        if self.type == "W":
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        
        x = rearrange(x, "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c", p1=self.window_size, p2=self.window_size).contiguous()
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, "b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c", p1=self.window_size, p2=self.window_size).contiguous()

        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, "b nw np (threeh c) -> threeh b nw np c", c=self.head_dim).chunk(3, dim=0)

        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        relative_position_bias = self.relative_position_params[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        # Broadcast bias correctly over Batch (dim 1) and NumWindows (dim 2)
        sim = sim + relative_position_bias.unsqueeze(1).unsqueeze(1)

        if self.type != "W":
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")
        output = self.linear(output)
        output = rearrange(output, "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c", w1=h_windows, p1=self.window_size)

        if self.type == "W":
            output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))
        return output

class ResScaleConvGateBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type="W"):
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
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, block=ResScaleConvGateBlock, block_num=2, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.block_num = block_num
        for i in range(block_num):
            ty = "W" if i % 2 == 0 else "SW"
            self.layers.append(block(input_dim, input_dim, head_dim, window_size, drop_path, type=ty))
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
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        # Expects NHWC input
        x = x.permute(0, 3, 1, 2).contiguous() # NHWC -> NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC
        return x

class ConvGELU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = hidden_features // 2
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # Expects NHWC input
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
        # Expects NHWC (channel last) for correct broadcasting
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
        self.dw_conv = nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim)
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
        self.conv_layers = nn.ModuleList([
            nn.Sequential(nn.GELU(), ConvWithDW(dim, dim)) for _ in range(self.layer_num)
        ])
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
        # Expects NHWC input
        x = x.permute(0, 3, 1, 2) # Permute to NCHW for Conv/Dense layers
        s = self.s(x)
        s_out = self.dense(s)
        x = s_out * self.spatial_atte(s_out)
        x = x.permute(0, 2, 3, 1) # Back to NHWC
        return x

# ==========================================
# PART 2: NEW WAVELET & MOE CLASSES
# ==========================================

class LiftingBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

class LearnableWaveletTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.P_horz = LiftingBlock(in_channels)
        self.U_horz = LiftingBlock(in_channels)
        self.P_vert = LiftingBlock(in_channels)
        self.U_vert = LiftingBlock(in_channels)

    def _lifting_step(self, x, P, U, dim):
        if dim == 2: # Height
            even = x[:, :, 0::2, :]
            odd  = x[:, :, 1::2, :]
        else:        # Width
            even = x[:, :, :, 0::2]
            odd  = x[:, :, :, 1::2]

        high = odd - P(even)
        low = even + U(high)
        return low, high

    def forward(self, x):
        l_horz, h_horz = self._lifting_step(x, self.P_horz, self.U_horz, dim=3)
        ll, lh = self._lifting_step(l_horz, self.P_vert, self.U_vert, dim=2)
        hl, hh = self._lifting_step(h_horz, self.P_vert, self.U_vert, dim=2)
        hf = torch.cat([lh, hl, hh], dim=1)
        return ll, hf

class InverseLearnableWaveletTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.P_horz = LiftingBlock(in_channels)
        self.U_horz = LiftingBlock(in_channels)
        self.P_vert = LiftingBlock(in_channels)
        self.U_vert = LiftingBlock(in_channels)

    def _inverse_lifting_step(self, low, high, P, U, dim):
        even = low - U(high)
        odd = high + P(even)
        
        if dim == 2: # Height
            B, C, H, W = even.shape
            out = torch.zeros(B, C, H*2, W, device=low.device)
            out[:, :, 0::2, :] = even
            out[:, :, 1::2, :] = odd
        else: # Width
            B, C, H, W = even.shape
            out = torch.zeros(B, C, H, W*2, device=low.device)
            out[:, :, :, 0::2] = even
            out[:, :, :, 1::2] = odd
        return out

    def forward(self, ll, hf):
        C = ll.shape[1]
        lh, hl, hh = torch.split(hf, C, dim=1)
        l_horz = self._inverse_lifting_step(ll, lh, self.P_vert, self.U_vert, dim=2)
        h_horz = self._inverse_lifting_step(hl, hh, self.P_vert, self.U_vert, dim=2)
        x = self._inverse_lifting_step(l_horz, h_horz, self.P_horz, self.U_horz, dim=3)
        return x

class InterBandRouter(nn.Module):
    def __init__(self, dim_high, dim_low, num_experts):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim_high + dim_low, dim_high),
            nn.GELU(),
            nn.Linear(dim_high, dim_high // 4),
            nn.GELU(),
            nn.Linear(dim_high // 4, num_experts)
        )

    def forward(self, hf_features, lf_features):
        combined = torch.cat([hf_features, lf_features], dim=-1)
        logits = self.fusion(combined)
        return logits

class SpectralMoEDictionaryCrossAttention(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_rate=4, head_num=4, 
                 qkv_bias=True, num_experts=4, expert_entries=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_num = head_num
        self.num_experts = num_experts
        
        self.dim_low = 32 * head_num
        self.dim_high = 32 * head_num * 3 
        
        self.dwt = LearnableWaveletTransform(32 * head_num)
        self.idwt = InverseLearnableWaveletTransform(32 * head_num)
        
        self.x_trans = nn.Linear(input_dim, 32 * head_num, bias=qkv_bias)
        self.output_trans = nn.Linear(32 * head_num, output_dim, bias=qkv_bias)

        # Low Freq
        c_block = 32 * head_num
        self.ln_low = nn.LayerNorm(c_block)
        self.q_low = nn.Linear(c_block, c_block, bias=qkv_bias)
        self.k_low = nn.Linear(c_block, c_block, bias=qkv_bias)
        self.ln_dict_low = nn.LayerNorm(c_block)
        self.scale = c_block ** -0.5
        self.dict_low = nn.Parameter(torch.randn(64, c_block))

        # High Freq (Inter-Band Guided MoE)
        self.router = InterBandRouter(self.dim_high, c_block, num_experts)
        self.experts_high = nn.Parameter(torch.randn(num_experts * expert_entries, self.dim_high))
        self.ln_dict_high = nn.LayerNorm(self.dim_high)
        self.ln_high = nn.LayerNorm(self.dim_high)
        self.q_high = nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias)
        self.k_high = nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias)
        self.v_all = nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias)

        # Utils - Initialize MSA and MLP with HIDDEN dim (c_block)
        self.msa = MultiScaleAggregation(c_block) 
        self.ln_scale = nn.LayerNorm(c_block)
        self.res_scale_1 = Scale(c_block, init_value=1.0)
        self.ln_mlp = nn.LayerNorm(c_block)
        self.mlp = ConvGELU(c_block, c_block * mlp_rate) 
        self.res_scale_2 = Scale(c_block, init_value=1.0)
        
        # Loss-Free Balancing State (Algo 1)
        # Register as buffer so it's saved in state_dict but not optimized by SGD
        self.register_buffer("expert_biases", torch.zeros(num_experts))
        
        self.last_routing_weights = None
        self.last_routing_logits = None
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.dict_low, std=0.02)
        trunc_normal_(self.experts_high, std=0.02)

    def process_low_freq(self, x):
        x_perm = x.permute(0, 2, 3, 1) 
        x_norm = self.ln_low(x_perm)
        q = self.q_low(x_norm)
        k = self.k_low(self.ln_dict_low(self.dict_low))
        attn = torch.matmul(q, k.transpose(0, 1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, self.dict_low)
        return out.permute(0, 3, 1, 2)

    def process_high_freq_guided(self, hf, lf):
        B, H, W, C_high = hf.shape
        
        # 1. Routing Logits
        routing_logits = self.router(hf, lf) 
        
        # Save RAW logits for external training loop/loss monitoring
        # (Though loss-free algo updates biases manually, monitoring is still useful)
        self.last_routing_logits = routing_logits 
        
        # 2. Apply Loss-Free Balancing Bias (Equation 3 in PDF)
        # "expert bias term is only used to adjust the routing strategy (top-k selection)"
        # Note: We apply bias to logits before Top-K, but typically softmax for gating weights 
        # is calculated on the *original* or *biased* logits depending on specific MoE implementation.
        # The paper says: "use the biased scores to determine the top-K selection".
        biased_logits = routing_logits + self.expert_biases.view(1, 1, 1, -1)
        
        # Get Routing Weights (Probabilities)
        # Paper implies bias affects selection. Standard MoE usually softmaxes the selected logits.
        # Here we follow standard practice: Softmax on biased logits to influence gradients 
        # (if we were using aux loss) or just selection.
        routing_weights = F.softmax(biased_logits, dim=-1)
        self.last_routing_weights = routing_weights # Save for tracking usage
        
        # 3. Expert Attention
        all_keys = self.k_high(self.ln_dict_high(self.experts_high))
        q = self.q_high(self.ln_high(hf))
        q_flat = q.view(B, -1, C_high)
        sim = torch.matmul(q_flat, all_keys.transpose(0, 1)) * (C_high ** -0.5)
        sim = sim.view(B, H*W, self.num_experts, -1)
        attn = F.softmax(sim, dim=-1)
        
        v_all = self.v_all(self.experts_high)
        v_experts = v_all.view(self.num_experts, -1, C_high)
        
        expert_outputs = torch.einsum("bhke,kec->bhkc", attn, v_experts)
        
        # 4. Router Weighting
        router_weights_flat = routing_weights.view(B, H*W, self.num_experts, 1)
        final_out = (expert_outputs * router_weights_flat).sum(dim=2)
        
        return final_out.view(B, H, W, C_high)

    def forward(self, x):
        shortcut = x
        x_emb = self.x_trans(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        ll, hf = self.dwt(x_emb)
        
        ll_processed = self.process_low_freq(ll)
        
        ll_perm = ll_processed.permute(0, 2, 3, 1)
        hf_perm = hf.permute(0, 2, 3, 1)
        
        hf_processed = self.process_high_freq_guided(hf_perm, ll_perm)
        hf_processed = hf_processed.permute(0, 3, 1, 2)
        
        recon = self.idwt(ll_processed, hf_processed) # Output is NCHW
        
        #Apply MSA and MLP properly by respecting channel-last/channel-first conventions
        
        # 1. MSA (MultiScaleAggregation) expects NHWC input
        recon_nhwc = recon.permute(0, 2, 3, 1) 
        msa_in = self.ln_scale(recon_nhwc)
        # self.msa returns NHWC
        msa_out = self.msa(msa_in) 
        # self.res_scale_1 expects NHWC
        scaled_msa = self.res_scale_1(msa_out)
        # Add residual to recon (NCHW) by permuting scaled_msa back to NCHW
        recon = recon + scaled_msa.permute(0, 3, 1, 2)
        
        # 2. MLP (ConvGELU) expects NHWC input
        recon_nhwc = recon.permute(0, 2, 3, 1)
        mlp_in = self.ln_mlp(recon_nhwc)
        # self.mlp returns NHWC
        mlp_out = self.mlp(mlp_in)
        # self.res_scale_2 expects NHWC
        scaled_mlp = self.res_scale_2(mlp_out)
        # Add residual to recon (NCHW)
        recon = recon + scaled_mlp.permute(0, 3, 1, 2)
        
        # Output Projection
        # output_trans expects NHWC
        out = self.output_trans(recon.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        if self.input_dim == self.output_dim:
             out = out + shortcut
             
        return out