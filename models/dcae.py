import torch
from torch import nn
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models.utils import update_registered_buffers

from modules.swin_module import (
    ResScaleConvGateBlock,
    SwinBlockWithConvMulti,
    MutiScaleDictionaryCrossAttentionGELU,
    MoEDictionaryCrossAttention
)
from modules.resnet_module import (
    ResidualBottleneckBlockWithStride,
    ResidualBottleneckBlockWithUpsample,
)

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def ste_round(x):
    return torch.round(x) - x.detach() + x


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(
        torch.linspace(
            torch.log(torch.tensor(min)), torch.log(torch.tensor(max)), levels
        )
    )


class DCAE(CompressionModel):
    def __init__(
        self,
        head_dim=None,
        N=192,
        M=320,
        num_slices=5,
        max_support_slices=5,
        **kwargs,
    ):
        super().__init__()
        if head_dim is None:
            self.head_dim = [8, 16, 32, 32, 16, 8]
        else:
            self.head_dim = head_dim

        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.M = M
        self.N = N

        input_image_channel = 3
        output_image_channel = 3
        feature_dim = [96, 144, 256]

        basic_block = ResScaleConvGateBlock
        swin_block = SwinBlockWithConvMulti
        block_counts = [1, 2, 12]

        # dict_num = 128
        # dict_head_num = 20
        # dict_dim = 32 * dict_head_num
        # self.dt = nn.Parameter(torch.randn([dict_num, dict_dim]), requires_grad=True)

        dict_head_num = 20
        num_experts = 4       # K=4 experts
        expert_entries = 64   # N=64 entries per expert (Total capacity = 256)
        
        prior_dim = M
        mlp_rate = 4


        # Cross Attention
        # self.dt_cross_attention = nn.ModuleList(
        #     [
        #         MutiScaleDictionaryCrossAttentionGELU(
        #             input_dim=M * 2 + (M // self.num_slices) * i,
        #             output_dim=M,
        #             head_num=dict_head_num,
        #             mlp_rate=mlp_rate,
        #             qkv_bias=True,
        #         )
        #         for i in range(num_slices)
        #     ]
        # )
        self.dt_cross_attention = nn.ModuleList(
            [
                MoEDictionaryCrossAttention(
                    input_dim=M * 2 + (M // self.num_slices) * i,
                    output_dim=M,
                    head_num=dict_head_num,
                    mlp_rate=mlp_rate,
                    qkv_bias=True,
                    num_experts=num_experts,    # New Param
                    expert_entries=expert_entries # New Param
                )
                for i in range(num_slices)
            ]
        )

        # Analysis Transform (Encoder)
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

        # Synthesis Transform (Decoder)
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

        # Hyper-Analysis
        self.h_a = nn.Sequential(
            ResidualBottleneckBlockWithStride(M, N),
            SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvGateBlock, block_num=1),
            nn.Conv2d(N, 192, kernel_size=3, stride=2, padding=1),
        )

        # Hyper-Synthesis (Split paths)
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

        # Transforms
        slice_dims = 320 // self.num_slices

        self.cc_mean_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        320 * 2 + slice_dims * min(i, 5) + prior_dim,
                        224,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(224, 128, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(128, slice_dims, kernel_size=3, stride=1, padding=1),
                )
                for i in range(self.num_slices)
            ]
        )

        self.cc_scale_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        320 * 2 + slice_dims * min(i, 5) + prior_dim,
                        224,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(224, 128, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(128, slice_dims, kernel_size=3, stride=1, padding=1),
                )
                for i in range(self.num_slices)
            ]
        )

        self.lrp_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        320 * 2 + slice_dims * min(i + 1, 6) + prior_dim,
                        224,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(224, 128, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(128, slice_dims, kernel_size=3, stride=1, padding=1),
                )
                for i in range(self.num_slices)
            ]
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x):
        # b = x.size(0)
        # dt = self.dt.unsqueeze(0).expand(b, -1, -1)

        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )
            query = torch.cat([latent_scales, latent_means] + support_slices, dim=1)

            # dict_info = self.dt_cross_attention[slice_index](query, dt)
            dict_info = self.dt_cross_attention[slice_index](query)

            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]
            mu_list.append(mu)

            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, : y_shape[0], : y_shape[1]]
            scale_list.append(scale)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)

            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)

            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": means, "scales": scales, "y": y},
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
        """Return a new model instance from `state_dict`."""
        # Ensure 'g_a.0' and 'g_a.6' exist in your saved weights.
        try:
            N = state_dict["g_a.0.weight"].size(0)
            M = state_dict["g_a.6.weight"].size(0)
        except KeyError:
            # Fallback or specific logic if keys differ
            N = 192
            M = 320

        net = cls(N=N, M=M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        # b = x.size(0)
        # dt = self.dt.unsqueeze(0).expand(b, -1, -1)

        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()

        all_symbols = []
        all_indexes = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )
            query = torch.cat([latent_scales, latent_means] + support_slices, dim=1)
            # dict_info = self.dt_cross_attention[slice_index](query, dt)
            dict_info = self.dt_cross_attention[slice_index](query)

            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            all_symbols.append(y_q_slice.reshape(-1))
            all_indexes.append(index.reshape(-1))

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
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

        # b = z_hat.size(0)
        # dt = self.dt.unsqueeze(0).expand(b, -1, -1)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )
            query = torch.cat([latent_scales, latent_means] + support_slices, dim=1)
            # dict_info = self.dt_cross_attention[slice_index](query, dt)
            dict_info = self.dt_cross_attention[slice_index](query)

            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(
                index.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )

            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            device = next(self.parameters()).device
            rv = rv.to(device)

            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
