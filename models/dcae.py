import torch
import torch.nn as nn
import math

from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models.utils import update_registered_buffers

# Assumes your original modules are present in the path
from modules.new_swin_module import (
    ResScaleConvGateBlock,
    SwinBlockWithConvMulti,
    SpectralMoEDictionaryCrossAttention,
)
from modules.resnet_module import (
    ResidualBottleneckBlockWithStride,
    ResidualBottleneckBlockWithUpsample,
)


def ste_round(x):
    return torch.round(x) - x.detach() + x


def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Replaces standard Convs/ResNets in the Entropy Model.
    """

    def __init__(self, dim, inter_dim=None):
        super().__init__()
        self.dim = inter_dim if inter_dim is not None else dim
        dw_channel = self.dim << 1
        ffn_channel = self.dim << 1

        self.dwconv = nn.Sequential(
            nn.Conv2d(self.dim, dw_channel, 1),
            nn.Conv2d(dw_channel, dw_channel, 3, 1, padding=1, groups=dw_channel),
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )
        self.FFN = nn.Sequential(
            nn.Conv2d(self.dim, ffn_channel, 1),
            SimpleGate(),
            nn.Conv2d(ffn_channel >> 1, self.dim, 1),
        )
        self.norm1 = LayerNorm2d(self.dim)
        self.norm2 = LayerNorm2d(self.dim)
        self.conv1 = nn.Conv2d(dw_channel >> 1, self.dim, 1)

        self.beta = nn.Parameter(torch.zeros((1, self.dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, self.dim, 1, 1)), requires_grad=True)

        self.in_conv = (
            nn.Conv2d(dim, inter_dim, 1) if inter_dim is not None else nn.Identity()
        )
        self.out_conv = (
            nn.Conv2d(inter_dim, dim, 1) if inter_dim is not None else nn.Identity()
        )

    def forward(self, x):
        x_in = self.in_conv(x)
        identity = x_in
        x = self.norm1(x_in)
        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv1(x)
        out = identity + x * self.beta

        identity = out
        out = self.norm2(out)
        out = self.FFN(out)
        out = identity + out * self.gamma

        out = self.out_conv(out)
        return out


class DCAE(CompressionModel):
    def __init__(
        self,
        head_dim=None,
        N=192,
        M=320,
        max_support_slices=5,
    ):
        super().__init__()

        # --- Config & SOTA Slicing ---
        if head_dim is None:
            self.head_dim = [8, 16, 32, 32, 16, 8]
        else:
            self.head_dim = head_dim

        self.N = N
        self.M = M
        self.max_support_slices = max_support_slices

        # Uneven Slicing
        # Sends information in increasing chunk sizes [16, 16, 32, 64, 192]
        self.groups = [0, 16, 16, 32, 64, 192]
        self.num_slices = len(self.groups) - 1

        # ==================================================
        # PART 1: BACKBONE (Preserved Swin/ResNet)
        # ==================================================
        self.window_size = 8
        input_image_channel = 3
        output_image_channel = 3
        feature_dim = [96, 144, 256]

        basic_block = ResScaleConvGateBlock
        swin_block = SwinBlockWithConvMulti
        block_counts = [1, 2, 12]

        # Encoder (Analysis)
        self.m_down1 = nn.Sequential(
            swin_block(
                feature_dim[0],
                feature_dim[0],
                self.head_dim[0],
                self.window_size,
                0,
                basic_block,
                block_num=block_counts[0],
            ),
            ResidualBottleneckBlockWithStride(feature_dim[0], feature_dim[1]),
        )
        self.m_down2 = nn.Sequential(
            swin_block(
                feature_dim[1],
                feature_dim[1],
                self.head_dim[1],
                self.window_size,
                0,
                basic_block,
                block_num=block_counts[1],
            ),
            ResidualBottleneckBlockWithStride(feature_dim[1], feature_dim[2]),
        )
        self.m_down3 = nn.Sequential(
            swin_block(
                feature_dim[2],
                feature_dim[2],
                self.head_dim[2],
                self.window_size,
                0,
                basic_block,
                block_num=block_counts[2],
            ),
            nn.Conv2d(feature_dim[2], M, kernel_size=5, stride=2, padding=2),
        )

        self.g_a = nn.Sequential(
            ResidualBottleneckBlockWithStride(input_image_channel, feature_dim[0]),
            self.m_down1,
            self.m_down2,
            self.m_down3,
        )

        # Decoder (Synthesis)
        self.m_up1 = nn.Sequential(
            swin_block(
                feature_dim[2],
                feature_dim[2],
                self.head_dim[3],
                self.window_size,
                0,
                basic_block,
                block_num=block_counts[2],
            ),
            ResidualBottleneckBlockWithUpsample(feature_dim[2], feature_dim[1]),
        )
        self.m_up2 = nn.Sequential(
            swin_block(
                feature_dim[1],
                feature_dim[1],
                self.head_dim[4],
                self.window_size,
                0,
                basic_block,
                block_num=block_counts[1],
            ),
            ResidualBottleneckBlockWithUpsample(feature_dim[1], feature_dim[0]),
        )
        self.m_up3 = nn.Sequential(
            swin_block(
                feature_dim[0],
                feature_dim[0],
                self.head_dim[5],
                self.window_size,
                0,
                basic_block,
                block_num=block_counts[0],
            ),
            ResidualBottleneckBlockWithUpsample(feature_dim[0], output_image_channel),
        )

        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(
                M, feature_dim[2], kernel_size=5, stride=2, output_padding=1, padding=2
            ),
            self.m_up1,
            self.m_up2,
            self.m_up3,
        )

        # Hyper-Prior
        self.h_a = nn.Sequential(
            ResidualBottleneckBlockWithStride(M, N),
            SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvGateBlock, block_num=1),
            nn.Conv2d(N, 192, kernel_size=3, stride=2, padding=1),
        )

        self.h_z_s1 = nn.Sequential(
            nn.ConvTranspose2d(
                192, N, kernel_size=3, stride=2, output_padding=1, padding=1
            ),
            SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvGateBlock, block_num=1),
            ResidualBottleneckBlockWithUpsample(N, M),
        )

        self.h_z_s2 = nn.Sequential(
            nn.ConvTranspose2d(
                192, N, kernel_size=3, stride=2, output_padding=1, padding=1
            ),
            SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvGateBlock, block_num=1),
            ResidualBottleneckBlockWithUpsample(N, M),
        )

        # ==================================================
        # PART 2: ENTROPY MODULES (NAFNet + MoE + LRP)
        # ==================================================

        # Config for Spectral MoE
        dict_head_num = 10
        mlp_rate = 4
        num_experts = 4
        expert_entries = 32

        self.dt_cross_attention = nn.ModuleList()  # MoE
        self.context_transforms = nn.ModuleList()  # NAF Context
        self.mean_transforms = nn.ModuleList()  # Projector
        self.scale_transforms = nn.ModuleList()  # Projector
        self.lrp_transforms = nn.ModuleList()  # LRP Refinement

        cum_channels = 0

        for i in range(self.num_slices):
            current_dim = self.groups[i + 1]

            # 1. Spectral MoE (Adapts to accumulated channel size)
            # Input: Hyper(M) + Hyper(M) + Accumulated Slices
            moe_input_dim = (M * 2) + cum_channels

            self.dt_cross_attention.append(
                SpectralMoEDictionaryCrossAttention(
                    input_dim=moe_input_dim,
                    output_dim=M,
                    head_num=dict_head_num,
                    mlp_rate=mlp_rate,
                    qkv_bias=True,
                    num_experts=num_experts,
                    expert_entries=expert_entries,
                )
            )

            # 2. NAFBlock for Context Processing (Stronger than simple Conv)
            # Input: MoE Output(M) + Hyper(M*2) + Accumulated Slices
            # We treat this sum as the "Support"
            support_dim = M + (M * 2) + cum_channels
            self.context_transforms.append(NAFBlock(support_dim, inter_dim=128))

            # 3. Projectors for Means and Scales
            self.mean_transforms.append(
                nn.Sequential(
                    nn.Conv2d(support_dim, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, current_dim, 3, 1, 1),
                )
            )

            self.scale_transforms.append(
                nn.Sequential(
                    nn.Conv2d(support_dim, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, current_dim, 3, 1, 1),
                )
            )

            # 4. LRP (Latent Residual Prediction)
            # Refines the quantization error
            lrp_in = support_dim + current_dim
            self.lrp_transforms.append(
                nn.Sequential(
                    nn.Conv2d(lrp_in, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, current_dim, 3, 1, 1),
                )
            )

            cum_channels += current_dim

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x):
        # 1. Transform
        y = self.g_a(x)
        y_shape = y.shape[2:]

        # 2. Hyper-Prior
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)

        # 3. Entropy Modeling (Uneven Slicing)
        y_slices = y.split(self.groups[1:], 1)

        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []

        # New: Collect router logits for auxiliary loss
        all_logits = []

        for i, y_slice in enumerate(y_slices):
            # Construct Query for MoE: [Hyper + Previous Slices]
            if i == 0:
                query = torch.cat([latent_means, latent_scales], dim=1)
            else:
                prev_slices = torch.cat(y_hat_slices, dim=1)
                query = torch.cat([latent_means, latent_scales, prev_slices], dim=1)

            # A. Spectral MoE
            # We need to assume your MoE class in 'new_swin_module.py'
            # exposes the logits or we access them if saved internally.
            # If your SpectralMoEDictionaryCrossAttention forward returns just output,
            # you might need to modify it or access the internal attribute 'last_routing_logits'
            # Assuming here we can get output.
            dict_info = self.dt_cross_attention[i](query)

            # Try to retrieve logits from the module if stored during forward
            # This requires 'SpectralMoEDictionaryCrossAttention' to store 'self.last_routing_logits'
            all_logits.append((self.dt_cross_attention[i].last_routing_logits, self.dt_cross_attention[i].last_routing_indices))

            # B. Construct Support for Context: [MoE_Out + Query]
            support = torch.cat([dict_info, query], dim=1)

            # C. NAFBlock Context Enhancement
            support_feat = self.context_transforms[i](support)

            # D. Predict Parameters
            mu = self.mean_transforms[i](support_feat)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]
            mu_list.append(mu)

            scale = self.scale_transforms[i](support_feat)
            scale = scale[:, :, : y_shape[0], : y_shape[1]]
            scale_list.append(scale)

            # E. Entropy Estimate
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)

            # F. Quantize
            y_hat_slice = ste_round(y_slice - mu) + mu

            # G. LRP Refinement
            lrp_in = torch.cat([support_feat, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[i](lrp_in)

            # Apply Soft Refinement (0.5 * tanh is safer than raw addition)
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        # 4. Reconstruction
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": means, "scales": scales, "y": y},
            "dict_info": dict_info,
            "router_logits": tuple(all_logits) if all_logits else None,
        }

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        try:
            N = state_dict["g_a.0.weight"].size(0)
            M = state_dict["g_a.6.weight"].size(0)
        except KeyError:
            N = 192
            M = 320
        net = cls(N=N, M=M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)

        y_slices = y.split(self.groups[1:], 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        all_symbols = []
        all_indexes = []

        for i, y_slice in enumerate(y_slices):
            if i == 0:
                query = torch.cat([latent_means, latent_scales], dim=1)
            else:
                prev_slices = torch.cat(y_hat_slices, dim=1)
                query = torch.cat([latent_means, latent_scales, prev_slices], dim=1)

            dict_info = self.dt_cross_attention[i](query)
            support = torch.cat([dict_info, query], dim=1)
            support_feat = self.context_transforms[i](support)

            mu = self.mean_transforms[i](support_feat)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]
            scale = self.scale_transforms[i](support_feat)
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            all_symbols.append(y_q_slice.reshape(-1))
            all_indexes.append(index.reshape(-1))

            lrp_in = torch.cat([support_feat, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[i](lrp_in)
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))

            y_hat_slices.append(y_hat_slice)

        symbols_list = torch.cat(all_symbols).tolist()
        indexes_list = torch.cat(all_indexes).tolist()

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )
        y_string = encoder.flush()

        return {"strings": [[y_string], z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for i in range(self.num_slices):
            if i == 0:
                query = torch.cat([latent_means, latent_scales], dim=1)
            else:
                prev_slices = torch.cat(y_hat_slices, dim=1)
                query = torch.cat([latent_means, latent_scales, prev_slices], dim=1)

            dict_info = self.dt_cross_attention[i](query)
            support = torch.cat([dict_info, query], dim=1)
            support_feat = self.context_transforms[i](support)

            mu = self.mean_transforms[i](support_feat)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]
            scale = self.scale_transforms[i](support_feat)
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(
                index.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            device = next(self.parameters()).device
            rv = rv.to(device)

            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_in = torch.cat([support_feat, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[i](lrp_in)
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp(0, 1)

        return {"x_hat": x_hat}
