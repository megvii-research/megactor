# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.modeling_utils import ModelMixin

from einops import rearrange, repeat

from diffusers.models.embeddings import PatchEmbed
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.embeddings import ImagePositionalEmbeddings
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings

from .dit_attention import BasicTransformerBlock, AdaLayerNormZero, DiTNetModel
from .dit_appearence import _LoRACompatibleLinear, Identity


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNetOutput(BaseOutput):
    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        height,
        width,
        patch_size,
        in_channels,
        embed_dim,
    ):
        super().__init__()
        self.pos_embed = PatchEmbed(
            height=height,
            width=width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.conv_out = zero_module(
            nn.Conv1d(in_channels=embed_dim, 
                    out_channels=embed_dim, 
                    kernel_size=1)
        )

    def forward(self, conditioning):
        conditioning = self.pos_embed(conditioning)
        embedding = self.conv_out(conditioning.transpose(1, 2)).transpose(1, 2)

        return embedding


class DiTControlNetModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    # 12 4 12 is up mid down
    # 下面from unet 里面的 numlayer 记录了这一信息 
    _controlnet_layer_num = 12

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        attention_type: str = "default",

    ):
        super().__init__()

        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            print("WARNING deprecate: norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif self.is_input_vectorized and self.is_input_patches:
            raise ValueError(
                f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make"
                " sure that either `num_vector_embeds` or `num_patches` is None."
            )
        elif not self.is_input_continuous and not self.is_input_vectorized and not self.is_input_patches:
            raise ValueError(
                f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size:"
                f" {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
            )

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            if use_linear_projection:
                self.proj_in = LoRACompatibleLinear(in_channels, inner_dim)
            else:
                self.proj_in = LoRACompatibleConv(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            assert sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
            assert num_vector_embeds is not None, "Transformer2DModel over discrete input must provide num_embed"

            self.height = sample_size
            self.width = sample_size
            self.num_vector_embeds = num_vector_embeds
            self.num_latent_pixels = self.height * self.width

            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=num_vector_embeds, embed_dim=inner_dim, height=self.height, width=self.width
            )
        elif self.is_input_patches:
            assert sample_size is not None, "Transformer2DModel over patched input must provide sample_size"

            self.height = sample_size
            self.width = sample_size

            self.patch_size = patch_size
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
            )
            # control net conditioning embedding
            # self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            #     height=sample_size,
            #     width=sample_size,
            #     patch_size=patch_size,
            #     in_channels=in_channels,
            #     embed_dim=inner_dim,
            # )
        
        # Define transformers blocks
        transformer_blocks = []
        # controlnet_blocks = []
        for i in range(num_layers):
            transformer_blocks.append(
                    BasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        cross_attention_dim=cross_attention_dim,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                        attention_bias=attention_bias,
                        only_cross_attention=only_cross_attention,
                        double_self_attention=double_self_attention,
                        upcast_attention=upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=norm_elementwise_affine,
                        attention_type=attention_type,
                    )
                )
            # controlnet_blocks.append(nn.Conv1d(in_channels=inner_dim, 
            #                             out_channels=inner_dim, 
            #                             kernel_size=1))

        self.transformer_blocks = nn.ModuleList(transformer_blocks)  
        self.transformer_blocks[-1].attn1.to_q = _LoRACompatibleLinear()
        self.transformer_blocks[-1].attn1.to_k = _LoRACompatibleLinear()
        self.transformer_blocks[-1].attn1.to_v = _LoRACompatibleLinear()
        self.transformer_blocks[-1].attn1.to_out = nn.ModuleList([Identity(), Identity()])
        self.transformer_blocks[-1].norm2 = Identity()
        self.transformer_blocks[-1].attn2 = None
        self.transformer_blocks[-1].norm3 = Identity()
        self.transformer_blocks[-1].ff = Identity()
        self.gradient_checkpointing = False

       

    @classmethod
    def from_dit(
        cls,
        dit_model: DiTNetModel,
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        load_weights_from_unet: bool = True,
    ):
        r"""
        Instantiate Controlnet class from UNet2DConditionModel.

        Parameters:
            unet (`UNet2DConditionModel`):
                UNet model which weights are copied to the ControlNet. Note that all configuration options are also
                copied where applicable.
        """
        num_layers = DiTControlNetModel._controlnet_layer_num
        controlnet = cls(
            num_attention_heads=dit_model.config.num_attention_heads,
            attention_head_dim=dit_model.config.attention_head_dim,
            in_channels=dit_model.config.in_channels,
            num_layers=num_layers,
            dropout=dit_model.config.dropout,
            norm_num_groups=dit_model.config.norm_num_groups,
            cross_attention_dim=dit_model.config.cross_attention_dim,
            attention_bias=dit_model.config.attention_bias,
            sample_size=dit_model.config.sample_size,
            num_vector_embeds=dit_model.config.num_vector_embeds,
            patch_size=dit_model.config.patch_size,
            activation_fn=dit_model.config.activation_fn,
            num_embeds_ada_norm=dit_model.config.num_embeds_ada_norm,
            use_linear_projection=dit_model.config.use_linear_projection,
            only_cross_attention=dit_model.config.only_cross_attention,
            double_self_attention=dit_model.config.double_self_attention,
            upcast_attention=dit_model.config.upcast_attention,
            norm_type=dit_model.config.norm_type,
            norm_elementwise_affine=dit_model.config.norm_elementwise_affine,
            attention_type=dit_model.config.attention_type,
        )

        if load_weights_from_unet:
            controlnet.pos_embed.load_state_dict(dit_model.pos_embed.state_dict())
            controlnet.transformer_blocks[:num_layers-1].load_state_dict(dit_model.transformer_blocks[:num_layers-1].state_dict())

        return controlnet

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

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
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value

    def forward(
        self, 
        hidden_states, 
        controlnet_cond, 
        encoder_hidden_states=None, 
        timestep=None, 
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
        
        hidden_states = controlnet_cond
        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = hidden_states.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=hidden_states.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(hidden_states.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timesteps.expand(hidden_states.shape[0])

        if hidden_states.shape[0] != class_labels.shape[0]:
            class_labels = rearrange(class_labels.unsqueeze(1).repeat(1, hidden_states.shape[0] // class_labels.shape[0]), "b f -> (b f)")

        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states, lora_scale)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = self.proj_in(hidden_states, scale=lora_scale)

        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        
        elif self.is_input_patches:
            hidden_states = self.pos_embed(hidden_states)

        # controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        # hidden_states += controlnet_cond
        # Blocks
        # print(f"before enter transformer_blocks hidden_states shape {hidden_states.shape}")
        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )


        
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module