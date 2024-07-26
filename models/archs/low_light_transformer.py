import functools
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import numpy as np
import cv2

from models.archs.transformer.Models import Encoder_patch66
from compressai.models.waseda import Cheng2020Anchor
from compressai.layers import (
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    AttentionBlock
)
import warnings
from compressai.models.utils import update_registered_buffers
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d, conv3x3, subpel_conv3x3

import math
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(
    min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class joint_compression_low_light_v1(nn.Module):
    def __init__(self, N=192, front_RBs=5, local_RBs=1):
        super(joint_compression_low_light_v1, self).__init__()
        self.N = N
        self.trans_nf = 48
        self.g_a = None
        self.g_a_block1 = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
        )
        self.g_a_block2 = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=N)
        self.local_feature_extraction_1 = arch_util.make_layer(ResidualBlock_noBN_f, local_RBs)                # local feature 1
        self.local_feature_extraction_1_CC = conv3x3(N, self.trans_nf)
        self.non_local_feature_extraction_1 = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)             # non local feature 1
        self.non_local_feature_extraction_1_CC = conv3x3(N, self.trans_nf)

        self.local_feature_extraction_2 = arch_util.make_layer(ResidualBlock_noBN_f, local_RBs//2)              # local feature 2
        self.local_feature_extraction_2_CC = conv3x3(N, self.trans_nf)
        self.non_local_feature_extraction_2 = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)             # non local feature 2
        self.non_local_feature_extraction_2_CC = conv3x3(N, self.trans_nf)
        self.transformer_enhance_1 = Encoder_patch66(d_model=4*4*self.trans_nf, d_inner=4*4*2*self.trans_nf, n_layers=3, n_head=4)
        self.transformer_enhance_2 = Encoder_patch66(d_model=4*4*self.trans_nf, d_inner=4*4*2*self.trans_nf, n_layers=3, n_head=4)
        self.feature_fusion_1 = arch_util.Condition_feature_fusion_add(N, prior_nc=self.trans_nf, ks=3, nhidden=N)
        self.feature_fusion_2 = arch_util.Condition_feature_fusion_add(N, prior_nc=self.trans_nf, ks=3, nhidden=N)
        self.enhance1 = AttentionBlock(N)
        self.enhance2 = AttentionBlock(N)

        self.context_prediction = MaskedConv2d(
            N, 2 * N, kernel_size=5, padding=2, stride=1
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 8 // 3, N * 6 // 3, 1),
        )
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _update_entropybottleneck(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self._update_entropybottleneck(force=force)
        return updated


    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)



    def g_a_func(self, x, mask=None, ll_enhance=False):
        x = self.g_a_block1(x)
        if ll_enhance:
            assert mask != None, "Please input mask"
            local_feature_1 = self.local_feature_extraction_1(x)
            local_feature_1 = self.local_feature_extraction_1_CC(local_feature_1)
            non_local_feature_1 = self.non_local_feature_extraction_1(x)
            non_local_feature_1 = self.non_local_feature_extraction_1_CC(non_local_feature_1)
            h_feature_1 = non_local_feature_1.shape[2]
            w_feature_1 = non_local_feature_1.shape[3]
            mask_1 = F.interpolate(mask, size=[h_feature_1, w_feature_1], mode='nearest')
            non_local_feature_1_unfold = F.unfold(non_local_feature_1, kernel_size=4, dilation=1, stride=4, padding=0)
            non_local_feature_1_unfold = non_local_feature_1_unfold.permute(0, 2, 1)
            mask_unfold = F.unfold(mask_1, kernel_size=4, dilation=1, stride=4, padding=0)
            mask_unfold = mask_unfold.permute(0, 2, 1)
            mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
            mask_unfold[mask_unfold <= 0.5] = 0.0
            non_local_feature_1_unfold = self.transformer_enhance_1(non_local_feature_1_unfold, None, src_mask=mask_unfold)
            non_local_feature_1_unfold = non_local_feature_1_unfold.permute(0, 2, 1)
            non_local_feature_1_unfold = nn.Fold(
                output_size=(h_feature_1, w_feature_1),
                kernel_size=(4, 4),
                stride=4,
                padding=0,
                dilation=1)(non_local_feature_1_unfold)
            channel = non_local_feature_1.shape[1]
            mask_1 = mask_1.repeat(1, channel, 1, 1)
            non_local_feature_1_fold = non_local_feature_1_unfold * (1 - mask_1) + local_feature_1 * mask_1
            x = self.feature_fusion_1(x, non_local_feature_1_fold)
            # x = self.enhance1(x)
        y_inter = x

        x = self.g_a_block2(x)
        if ll_enhance:
            assert mask != None, "Please input mask"
            local_feature_2 = self.local_feature_extraction_2(x)
            local_feature_2 = self.local_feature_extraction_2_CC(local_feature_2)
            non_local_feature_2 = self.non_local_feature_extraction_2(x)
            non_local_feature_2 = self.non_local_feature_extraction_2_CC(non_local_feature_2)
            h_feature_2 = non_local_feature_2.shape[2]
            w_feature_2 = non_local_feature_2.shape[3]
            mask_2 = F.interpolate(mask, size=[h_feature_2, w_feature_2], mode='nearest')
            non_local_feature_2_unfold = F.unfold(non_local_feature_2, kernel_size=4, dilation=1, stride=4, padding=0)
            non_local_feature_2_unfold = non_local_feature_2_unfold.permute(0, 2, 1)
            mask_unfold_2 = F.unfold(mask_2, kernel_size=4, dilation=1, stride=4, padding=0)
            mask_unfold_2 = mask_unfold_2.permute(0, 2, 1)
            mask_unfold_2 = torch.mean(mask_unfold_2, dim=2).unsqueeze(dim=-2)
            mask_unfold_2[mask_unfold_2 <= 0.5] = 0.0
            non_local_feature_2_unfold = self.transformer_enhance_2(non_local_feature_2_unfold, None, src_mask=mask_unfold_2)
            non_local_feature_2_unfold = non_local_feature_2_unfold.permute(0, 2, 1)
            non_local_feature_2_unfold = nn.Fold(
                output_size=(h_feature_2, w_feature_2),
                kernel_size=(4, 4),
                stride=4,
                padding=0,
                dilation=1)(non_local_feature_2_unfold)
            channel = non_local_feature_2.shape[1]
            mask_2 = mask_2.repeat(1, channel, 1, 1)
            non_local_feature_2_fold = non_local_feature_2_unfold * (1 - mask_2) + local_feature_2 * mask_2
            x = self.feature_fusion_2(x, non_local_feature_2_fold)
            # x = self.enhance2(x)
        y = x

        return y_inter, y

    def forward(self, x, gt=None, mask=None, ll_enhance=False, compress=False):
        if compress==False:
            y_inter, y = self.g_a_func(x, mask=mask, ll_enhance=ll_enhance)
            # g_a for clean input
            if gt is not None:
                y_inter_gt, y_gt = self.g_a_func(gt, mask=None, ll_enhance=False)
            else:
                y_inter_gt, y_gt = None, None
            x_hat = self.g_s(y)
            y_likelihoods = 0.0
            z_likelihoods = 0.0
        else:
            # g_a for low light input
            y_inter, y = self.g_a_func(x, mask=mask, ll_enhance=ll_enhance)

            # g_a for clean input
            if gt is not None:
                y_inter_gt, y_gt = self.g_a_func(gt, mask=None, ll_enhance=False)
            else:
                y_inter_gt, y_gt = None, None

            # h_a and h_s
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            params = self.h_s(z_hat)

            # g_s
            y_hat = self.gaussian_conditional.quantize(
                y, "noise" if self.training else "dequantize"
            )

            ctx_params = self.context_prediction(y_hat)
            gaussian_params = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
            x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "y_inter": y_inter,
            "y_inter_gt": y_inter_gt,
            "y": y,
            "y_gt": y_gt,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, mask=None, ll_enhance=True):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        _, y = self.g_a_func(x, mask=mask, ll_enhance=ll_enhance)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.N, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv


class joint_compression_low_light(Cheng2020Anchor):
    def __init__(self, N=192, front_RBs=5, local_RBs=1, **kwargs):
        super(joint_compression_low_light, self).__init__(N=N, **kwargs)
        self.g_a = None
        self.g_a_block1 = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
        )
        self.g_a_block2 = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=N)
        self.local_feature_extraction_1 = arch_util.make_layer(ResidualBlock_noBN_f, local_RBs)                 # local feature 1
        self.non_local_feature_extraction_1 = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)     # non local feature 1
        self.local_feature_extraction_2 = arch_util.make_layer(ResidualBlock_noBN_f, local_RBs//2)                 # local feature 2
        self.non_local_feature_extraction_2 = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)  # non local feature 2
        self.transformer_enhance_1 = Encoder_patch66(d_model=3072, d_inner=6144, n_layers=6)
        self.transformer_enhance_2 = Encoder_patch66(d_model=3072, d_inner=6144, n_layers=6)
        self.feature_fusion_1 = arch_util.Condition_feature_fusion(N, prior_nc=N, ks=3, nhidden=N * 2)
        self.feature_fusion_2 = arch_util.Condition_feature_fusion(N, prior_nc=N, ks=3, nhidden=N * 2)

    def g_a_func(self, x, mask=None, ll_enhance=False):
        x = self.g_a_block1(x)
        if ll_enhance:
            assert mask != None, "Please input mask"
            local_feature_1 = self.local_feature_extraction_1(x)
            non_local_feature_1 = self.non_local_feature_extraction_1(x)
            h_feature_1 = non_local_feature_1.shape[2]
            w_feature_1 = non_local_feature_1.shape[3]
            mask_1 = F.interpolate(mask, size=[h_feature_1, w_feature_1], mode='nearest')
            non_local_feature_1_unfold = F.unfold(non_local_feature_1, kernel_size=4, dilation=1, stride=4, padding=0)
            non_local_feature_1_unfold = non_local_feature_1_unfold.permute(0, 2, 1)
            mask_unfold = F.unfold(mask_1, kernel_size=4, dilation=1, stride=4, padding=0)
            mask_unfold = mask_unfold.permute(0, 2, 1)
            mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
            mask_unfold[mask_unfold <= 0.5] = 0.0
            non_local_feature_1_unfold = self.transformer_enhance_1(non_local_feature_1_unfold, None, src_mask=mask_unfold)
            non_local_feature_1_unfold = non_local_feature_1_unfold.permute(0, 2, 1)
            non_local_feature_1_unfold = nn.Fold(
                output_size=(h_feature_1, w_feature_1),
                kernel_size=(4, 4),
                stride=4,
                padding=0,
                dilation=1)(non_local_feature_1_unfold)
            channel = non_local_feature_1.shape[1]
            mask_1 = mask_1.repeat(1, channel, 1, 1)
            non_local_feature_1_fold = non_local_feature_1_unfold * (1 - mask_1) + local_feature_1 * mask_1
            x = self.feature_fusion_1(x, non_local_feature_1_fold)
        y_inter = x

        x = self.g_a_block2(x)
        if ll_enhance:
            assert mask != None, "Please input mask"
            local_feature_2 = self.local_feature_extraction_2(x)
            non_local_feature_2 = self.non_local_feature_extraction_2(x)
            h_feature_2 = non_local_feature_2.shape[2]
            w_feature_2 = non_local_feature_2.shape[3]
            mask_2 = F.interpolate(mask, size=[h_feature_2, w_feature_2], mode='nearest')
            non_local_feature_2_unfold = F.unfold(non_local_feature_2, kernel_size=4, dilation=1, stride=4, padding=0)
            non_local_feature_2_unfold = non_local_feature_2_unfold.permute(0, 2, 1)
            mask_unfold_2 = F.unfold(mask_2, kernel_size=4, dilation=1, stride=4, padding=0)
            mask_unfold_2 = mask_unfold_2.permute(0, 2, 1)
            mask_unfold_2 = torch.mean(mask_unfold_2, dim=2).unsqueeze(dim=-2)
            mask_unfold_2[mask_unfold_2 <= 0.5] = 0.0
            non_local_feature_2_unfold = self.transformer_enhance_2(non_local_feature_2_unfold, None, src_mask=mask_unfold_2)
            non_local_feature_2_unfold = non_local_feature_2_unfold.permute(0, 2, 1)
            non_local_feature_2_unfold = nn.Fold(
                output_size=(h_feature_2, w_feature_2),
                kernel_size=(4, 4),
                stride=4,
                padding=0,
                dilation=1)(non_local_feature_2_unfold)
            channel = non_local_feature_2.shape[1]
            mask_2 = mask_2.repeat(1, channel, 1, 1)
            non_local_feature_2_fold = non_local_feature_2_unfold * (1 - mask_2) + local_feature_2 * mask_2
            x = self.feature_fusion_2(x, non_local_feature_2_fold)
        y = x

        return y_inter, y

    def forward(self, x, gt=None, mask=None, ll_enhance=False):
        # g_a for low light input
        y_inter, y = self.g_a_func(x, mask=mask, ll_enhance=ll_enhance)

        # g_a for clean input
        if gt is not None:
            y_inter_gt, y_gt = self.g_a_func(gt, mask=None, ll_enhance=False)
        else:
            y_inter_gt, y_gt = None, None

        # h_a and h_s
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        # g_s
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "y_inter": y_inter,
            "y_inter_gt": y_inter_gt,
            "y": y,
            "y_gt": y_gt,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, mask=None, ll_enhance=True):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        _, y = self.g_a_func(x, mask=mask, ll_enhance=ll_enhance)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


###############################
class low_light_transformer(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(low_light_transformer, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf*2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64*2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)

    def forward(self, x, mask=None):
        x_center = x

        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        fea = self.feature_extraction(L1_fea_3)
        fea_light = self.recon_trunk_light(fea)

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')

        # xs = np.linspace(-1, 1, fea.size(3) // 4)
        # ys = np.linspace(-1, 1, fea.size(2) // 4)
        # xs = np.meshgrid(xs, ys)
        # xs = np.stack(xs, 2)
        # xs = torch.Tensor(xs).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).cuda()
        # xs = xs.view(fea.size(0), -1, 2)

        height = fea.shape[2]
        width = fea.shape[3]
        fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)
        fea_unfold = fea_unfold.permute(0, 2, 1)

        mask_unfold = F.unfold(mask, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold = mask_unfold.permute(0, 2, 1)
        mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
        mask_unfold[mask_unfold <= 0.5] = 0.0

        # fea_unfold = self.transformer(fea_unfold, xs, src_mask=mask_unfold)
        fea_unfold = self.transformer(fea_unfold, None, src_mask=mask_unfold)
        fea_unfold = fea_unfold.permute(0, 2, 1)
        fea_unfold = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(fea_unfold)

        channel = fea.shape[1]
        mask = mask.repeat(1, channel, 1, 1)
        fea = fea_unfold * (1 - mask) + fea_light * mask

        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x_center

        return out_noise
