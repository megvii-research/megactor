# Copyright 2024 Megvii inc.
#
# Copyright (2024) MegActor Authors.
#
# Megvii Inc. retain all intellectual property and proprietary rights in 
# and to this material, related documentation and any modifications thereto. 
# Any use, reproduction, disclosure or distribution of this material and related 
# documentation without an express license agreement from Megvii Inc. is strictly prohibited.

# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat

import os
import json

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)
from .resnet import InflatedConv3d
from .resampler import Resampler, MLPProjModel
from animate.motion_module import get_motion_module

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor



def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

import torch.nn.functional as F
import numpy as np
from diffusers.models.attention import Attention
from diffusers.models.attention import FeedForward
# from openSora: switch spatial and temp to one dim
# b c f h w -> b n c
class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(1, 2, 2),
        in_chans=4,
        embed_dim=4,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        x = self.proj(x)  # (B C T H W)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

from diffusers.utils.import_utils import is_xformers_available
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

def _initialize_weights(m):
    # Initialize weights with He initialization and zero out the biases
    if isinstance(m, nn.Conv2d) or isinstance(m, InflatedConv3d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        torch.nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class ConvInputLayer(nn.Module):
    # !!!WARNING this parmas is fix and we can adjust in this 
    def __init__(
            self,
            in_channel=3,
            conditioning_embedding_channels=4,
            use_refer_ada=False,
            use_motion_module=False,
            motion_module_type=None,
            motion_module_kwargs={},
    ):
        super().__init__()


        self.use_refer_ada = use_refer_ada

        # Only Spatial layer
        self.spatial_conv_layers = [
            nn.Sequential(
                InflatedConv3d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
                nn.SiLU(),
                InflatedConv3d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
                nn.SiLU()
            ),

            nn.Sequential(
                InflatedConv3d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                nn.SiLU(),
                InflatedConv3d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.SiLU()
            ),

            nn.Sequential(
                InflatedConv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                nn.SiLU(),
                InflatedConv3d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.SiLU()
            ),

            nn.Sequential(
                InflatedConv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.SiLU(),
                InflatedConv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.SiLU()
            ),

        ]
        motion_modules_channels = [16, 32, 64, 128]
        self.motion_modules = []
        for in_channels in motion_modules_channels:
            self.motion_modules.append(
                get_motion_module(
                    in_channels=in_channels,
                    motion_module_type=motion_module_type, 
                    motion_module_kwargs=motion_module_kwargs,
                ) if use_motion_module else None
            )
        
        self.spatial_conv_layers = nn.ModuleList(self.spatial_conv_layers)
        self.motion_modules = nn.ModuleList(self.motion_modules)

        # final projection layer
        self.spatial_conv_out = InflatedConv3d(in_channels=128, out_channels=conditioning_embedding_channels, kernel_size=1)
        self.spatial_conv_layers.apply(_initialize_weights)
        self.spatial_conv_out.apply(_initialize_weights)

        # Temp attention
        if self.use_refer_ada:
            self.refer_convs = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1),
                nn.SiLU(), # 256

                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1),
                nn.SiLU(), # 128

                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.SiLU(), # 64

                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.SiLU(), # 32

                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                nn.SiLU(), # 16

                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.SiLU(), # 8

                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
                nn.SiLU(), # 4

                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4),
                nn.SiLU(), # 1
            )
            self.refer_linears = nn.Sequential(
                nn.Linear(512, 256),
                nn.SiLU(),
            )
            self.refer_linears_out = nn.Sequential(
                nn.Linear(256, 256),
            )
            self.refer_linears_out = zero_module(self.refer_linears_out)
            self.refer_convs.apply(_initialize_weights)
        
    def forward(self, x, x_ref = None):
        x = x * 2. - 1.
        # x: b c f h w: b 3 f 512 512
        # x_ref: b c h w: b 3 512 512
        batch, c, video_length, _, __ = x.shape
        if self.use_refer_ada:
            x_ref = x_ref * 2. - 1.

        for spatial_layer, motion_module in zip(self.spatial_conv_layers, self.motion_modules):
            x = spatial_layer(x)
            x = motion_module(x, temb=None, encoder_hidden_states=None) if motion_module is not None else x


        if self.use_refer_ada:
            x_ref_feat = self.refer_convs(x_ref)
            x_ref_feat = x_ref_feat.view(batch, -1)
            # print(x.shape, x_ref.shape, x_ref_feat.shape)
            # x_ref_feat = self.refer_linears(x_ref_feat).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x_ref_feat = self.refer_linears(x_ref_feat)
            x_ref_feat = self.refer_linears_out(x_ref_feat).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            scale, shift = torch.chunk(x_ref_feat, chunks=2, dim=1)
            x = x * (1.0 + scale) + shift

        x = self.spatial_conv_out(x)
        return x


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size: Optional[int] = None,
            in_channels: int = 4,
            out_channels: int = 4,
            center_input_sample: bool = False,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "DownBlock3D",
            ),
            mid_block_type: str = "UNetMidBlock3DCrossAttn",
            up_block_types: Tuple[str] = (
                    "UpBlock3D",
                    "CrossAttnUpBlock3D",
                    "CrossAttnUpBlock3D",
                    "CrossAttnUpBlock3D"
            ),
            only_cross_attention: Union[bool, Tuple[bool]] = False,
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: int = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            act_fn: str = "silu",
            norm_num_groups: int = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: int = 1280,
            attention_head_dim: Union[int, Tuple[int]] = 8,
            dual_cross_attention: bool = False,
            use_linear_projection: bool = False,
            class_embed_type: Optional[str] = None,
            num_class_embeds: Optional[int] = None,
            upcast_attention: bool = False,
            resnet_time_scale_shift: str = "default",

            # Additional
            use_motion_module=False,
            motion_module_resolutions=(1, 2, 4, 8),
            motion_module_mid_block=False,
            motion_module_decoder_only=False,
            motion_module_type=None,
            motion_module_kwargs={},
            unet_use_cross_frame_attention=None,
            unet_use_temporal_attention=None,

            # Addition for image embeddings
            use_image_condition=False,
            # Additional for dwpose adapter
            use_dwpose_adapter=False,
            # Addition for AdaL Pose Guider
            use_refer_ada=False,
            # Addition for Motion Pose Guider
            use_pose_guider_motion=False,
    ):
        super().__init__()
        
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        # tensor is b c_in f h w -> b 320 f h w
        self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        motion_module_kwargs["norm_num_groups"] = 1
        self.condition_input_layer = ConvInputLayer(use_refer_ada=use_refer_ada, 
                                                    use_motion_module=use_pose_guider_motion,
                                                    motion_module_type=motion_module_type,
                                                    motion_module_kwargs=motion_module_kwargs,)
        del motion_module_kwargs["norm_num_groups"]

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            res = 2 ** i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,

                use_motion_module=use_motion_module and (res in motion_module_resolutions) and (
                    not motion_module_decoder_only),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,

                use_motion_module=use_motion_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            res = 2 ** (3 - i)
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,

                use_motion_module=use_motion_module and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            # for controlnet
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            # for pose_guider
            pose_guide_conditions: Optional[torch.Tensor] = None,
            ref_img_conditions: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2 ** self.num_upsamplers

        # if self.use_image_condition:
        #     # project global image to 16 tokens for cross-attention
        #     encoder_hidden_states = self.image_proj(encoder_hidden_states)
        #     encoder_hidden_states = encoder_hidden_states.reshape(-1, 16, 768)
        #     encoder_hidden_states = self.image_norm(encoder_hidden_states)

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # pre-process
        # add pose guide condition to noise latent
        sample = torch.cat([sample, self.condition_input_layer(pose_guide_conditions, ref_img_conditions)], dim=1)
        sample = self.conv_in(sample)
        
        # down
        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None

        down_block_res_samples = (sample,)


        for i, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb,
                                                        encoder_hidden_states=encoder_hidden_states)
                

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size,
                    encoder_hidden_states=encoder_hidden_states,
                )

        # post-process
        # video_length = sample.shape[2]
        # sample = rearrange(sample, "b c f h w -> (b f) c 1 h w")
        sample = self.conv_norm_out(sample)
        # sample = rearrange(sample, "(b f) c 1 h w -> b c f h w", f=video_length)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, unet_additional_kwargs=None):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded temporal unet's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config["_class_name"] = cls.__name__
        config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ]
        config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ]
        # config["mid_block_type"] = "UNetMidBlock3DCrossAttn"

        from diffusers.utils import WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME
        import safetensors
        model = cls.from_config(config, **unet_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        if not os.path.isfile(model_file):
            SAFETENSORS_WEIGHTS_NAME = 'diffusion_pytorch_model.safetensors'
            model_file = os.path.join(pretrained_model_path, SAFETENSORS_WEIGHTS_NAME)
            if os.path.isfile(model_file):
                state_dict = safetensors.torch.load_file(model_file, device="cpu")
            else:
                raise RuntimeError(f"{model_file} does not exist")
        else:
            state_dict = torch.load(model_file, map_location="cpu")

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        # print(f"### missing keys:\n{m}\n### unexpected keys:\n{u}\n")

        params = [p.numel() if "temporal" in n else 0 for n, p in model.named_parameters()]
        print(f"### Temporal Module Parameters: {sum(params) / 1e6} M")

        return model
