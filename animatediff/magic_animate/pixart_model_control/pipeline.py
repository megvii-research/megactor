# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************

# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

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
"""
TODO:
1. support multi-controlnet
2. [DONE] support DDIM inversion
3. support Prompt-to-prompt
"""

import inspect
import math
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from .dit_attention import DiTNetModel
from .dit_controlnet import DiTControlNetModel
from .dit_mutual_self_attention import ReferenceAttentionControl
from animatediff.magic_animate.context import (
    get_context_scheduler,
    get_total_steps
)
from animatediff.utils.util import get_tensor_interpolation_method

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]

# two input source_image is (-1, 1)
# control is (0, 1)

class AnimationPipeline(DiffusionPipeline):
    _optional_components = []
    _negative_prompt_embeds_path = "/data/code/yangshurong/cache/PixArt_XL_2_512/negative_prompt_embeds.pt"
    _negative_prompt_attention_mask_path = "/data/code/yangshurong/cache/PixArt_XL_2_512/negative_prompt_attention_mask.pt"
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: DiTNetModel,
            controlnet: DiTControlNetModel,
            scheduler: Union[
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0",
                      deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0",
                      deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(
            unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0",
                      deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        
        self.negative_prompt_embeds = torch.load(self._negative_prompt_embeds_path)
        self.negative_prompt_attention_mask = torch.load(self._negative_prompt_attention_mask_path)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1)
        
        self.negative_prompt_embeds = self.negative_prompt_embeds.to(dtype=torch.float16, device=self._execution_device)
        self.negative_prompt_attention_mask = self.negative_prompt_attention_mask.to(dtype=torch.float16, device=self._execution_device)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError(
                "Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(
                1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def decode_latents(self, latents, rank, decoder_consistency=None):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0]), disable=(rank != 0)):
            if decoder_consistency is not None:
                video.append(decoder_consistency(
                    latents[frame_idx:frame_idx + 1]))
            else:
                video.append(self.vae.decode(
                    latents[frame_idx:frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(
            self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(
                    callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator,
                        latents=None, clip_length=16):
        shape = (
            batch_size, num_channels_latents, clip_length, height // self.vae_scale_factor,
            width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                latents = [
                    torch.randn(
                        shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(
                    shape, generator=generator, device=rand_device, dtype=dtype).to(device)

            latents = latents.repeat(1, 1, video_length // clip_length, 1, 1)
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_condition(self, condition, num_videos_per_prompt, device, dtype, do_classifier_free_guidance):
        # prepare conditions for controlnet
        # condition = torch.from_numpy(condition.copy()).to(
        #     device=device, dtype=dtype) / 255.0
        # condition = torch.stack(
        #     [condition for _ in range(num_videos_per_prompt)], dim=0)
        # condition = rearrange(condition, 'b f h w c -> (b f) c h w').clone()
        # if do_classifier_free_guidance:
        #     condition = torch.cat([condition] * 2)

        
        condition = torch.concat(
            [condition for _ in range(num_videos_per_prompt)], dim=0)
        condition = rearrange(condition, 'b f h w c -> (b f) c h w')
        if do_classifier_free_guidance:
            condition = torch.cat([condition] * 2)
        return condition

    def next_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            eta=0.,
            verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps //
                       self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[
            timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    @torch.no_grad()
    def images2latents(self, images, dtype):
        """
        Convert RGB image to VAE latents
        """
        device = self._execution_device
        images = torch.from_numpy(images).float().to(dtype) / 127.5 - 1
        images = rearrange(images, "f h w c -> f c h w").to(device)
        latents = []
        for frame_idx in range(images.shape[0]):
            latents.append(self.vae.encode(
                images[frame_idx:frame_idx + 1])['latent_dist'].mean * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def invert(
            self,
            image: torch.Tensor,
            prompt,
            num_inference_steps=20,
            num_actual_inference_steps=10,
            eta=0.0,
            return_intermediates=False,
            **kwargs):
        """
        Adapted from: https://github.com/Yujun-Shi/DragDiffusion/blob/main/drag_pipeline.py#L440
        invert a real image into noise map with determinisc DDIM inversion
        """
        device = self._execution_device
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.images2latents(image)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):

            if num_actual_inference_steps is not None and i >= num_actual_inference_steps:
                continue
            model_inputs = latents

            # predict the noise
            # NOTE: the u-net here is UNet3D, therefore the model_inputs need to be of shape (b c f h w)
            model_inputs = rearrange(model_inputs, "f c h w -> 1 c f h w")
            noise_pred = self.unet(
                model_inputs, t, encoder_hidden_states=text_embeddings).sample
            noise_pred = rearrange(noise_pred, "b c f h w -> (b f) c h w")

            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            return latents, latents_list
        return latents

    def interpolate_latents(self, latents: torch.Tensor, interpolation_factor: int, device):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
            (latents.shape[0], latents.shape[1], ((latents.shape[2] - 1) * interpolation_factor) + 1, latents.shape[3],
             latents.shape[4]),
            device=latents.device,
            dtype=latents.dtype,
        )

        org_video_length = latents.shape[2]
        rate = [
            i / interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
            v0 = latents[:, :, i0, :, :]
            v1 = latents[:, :, i1, :, :]

            new_latents[:, :, new_index, :, :] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(
                    v0.to(device=device), v1.to(device=device), f)
                new_latents[:, :, new_index, :, :] = v.to(latents.device)
                new_index += 1

        new_latents[:, :, new_index, :, :] = v1
        new_index += 1

        return new_latents

    def select_controlnet_res_samples(self, controlnet_res_samples_cache_dict, context, do_classifier_free_guidance, b,
                                      f):
        
        # for every item sample, list contain all layer result
        layer_nums = len(controlnet_res_samples_cache_dict[0])
        b = b // 2 if do_classifier_free_guidance else b
        layer_results = []
        for j in range(layer_nums):
            layer_results.append([])
            for i in np.concatenate(np.array(context)):
                layer_results[j].append(controlnet_res_samples_cache_dict[i][j])
            layer_results[j] = torch.cat(layer_results[j])
            # len context, n, c
            if do_classifier_free_guidance:
                layer_results[j] = layer_results[j].unsqueeze(0).repeat(2, 1, 1, 1)
                layer_results[j] = rearrange(layer_results[j], "b f n c -> (b f) n c")
        return layer_results

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            video_length: Optional[int],
            prompt_embeddings: Optional[torch.FloatTensor] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator,
                                      List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[
                int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            controlnet_condition: list = None,
            controlnet_conditioning_scale: float = 1.0,
            context_frames: int = 16,
            context_stride: int = 1,
            context_overlap: int = 4,
            context_batch_size: int = 1,
            context_schedule: str = "uniform",
            init_latents: Optional[torch.FloatTensor] = None,
            num_actual_inference_steps: Optional[int] = None,
            appearance_encoder=None,
            reference_control_writer=None,
            reference_control_reader=None,
            source_image: str = None,
            decoder_consistency=None,
            froce_text_embedding_zero=False,
            ref_concat_image_noises_latents=None,
            do_classifier_free_guidance=True,
            add_noise_image_type="",
            **kwargs,
    ):
        """
        New args:
        - controlnet_condition          : condition map (e.g., depth, canny, keypoints) for controlnet
        - controlnet_conditioning_scale : conditioning scale for controlnet
        - init_latents                  : initial latents to begin with (used along with invert())
        - num_actual_inference_steps    : number of actual inference steps (while total steps is num_inference_steps)
        """

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        # self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        if guidance_scale > 1.0 and do_classifier_free_guidance:
            do_classifier_free_guidance = True
            print('this inference use classifier_free_guidance')
        else:
            do_classifier_free_guidance = False
            print('this inference not use classifier_free_guidance')

        # Encode input prompt
        '''
        source_image0 torch.Size([1, 512, 512, 3])
        control0 torch.Size([1, 40, 512, 512, 3])
        image_prompts torch.Size([2, 257, 1280])
        prompt_embeddings torch.Size([2, 257, 1280])
        text_embeddings torch.Size([2, 64, 768])
        controlnet_text_embeddings torch.Size([80, 64, 768])
        '''
        prompt_embeds = self.negative_prompt_embeds.clone()
        prompt_attention_mask = self.negative_prompt_attention_mask.clone()
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([prompt_attention_mask, prompt_attention_mask], dim=0)

        reference_control_writer = ReferenceAttentionControl(appearance_encoder,
                                                            model_type="appearencenet",
                                                            do_classifier_free_guidance=do_classifier_free_guidance,
                                                            mode='write',)
        # controlnet_writer = ReferenceAttentionControl(self.controlnet,
        #                                                     model_type="controlnet",
        #                                                     do_classifier_free_guidance=do_classifier_free_guidance,
        #                                                     mode='write',)
        reference_control_reader = ReferenceAttentionControl(self.unet,
                                                            model_type="unet",
                                                            do_classifier_free_guidance=do_classifier_free_guidance,
                                                            mode='read',)

        is_dist_initialized = kwargs.get("dist", False)
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)

        # Prepare video
        # FIXME: verify if num_videos_per_prompt > 1 works
        assert num_videos_per_prompt == 1
        assert batch_size == 1  # FIXME: verify if batch_size > 1 works

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        if init_latents is not None:
            latents = rearrange(
                init_latents, "(b f) c h w -> b c f h w", f=video_length)
        else:
            num_channels_latents = self.unet.in_channels
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                height,
                width,
                source_image.dtype,
                device,
                generator,
                latents,
                clip_length=video_length
            )
        latents_dtype = latents.dtype
        latent_channels = latents.shape[1]

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # For img2img setting
        if num_actual_inference_steps is None:
            num_actual_inference_steps = num_inference_steps

        source_image = rearrange(
            source_image, "b h w c -> b c h w").to(device)

        # print('infer controlnet_condition target unique is (0, 1), real is', controlnet_condition.unique())
        # print('infer source_image target unique is (-1, 1), real is', source_image.unique())

        controlnet_cond_images = rearrange(controlnet_condition, "b f h w c -> (b f) c h w") * 2. - 1.
        controlnet_cond_images = self.vae.encode(
            controlnet_cond_images)['latent_dist'].mean * 0.18215
        
        # source_image = source_image.repeat(video_length, 1, 1, 1)
        ref_image_latents = self.vae.encode(
            source_image)['latent_dist'].mean * 0.18215
        
        if add_noise_image_type != "":
            latents = latents * 0.9 + ref_image_latents.unsqueeze(2).repeat(1, 1, video_length, 1, 1) * 0.1 * self.scheduler.init_noise_sigma

        # if batch_size == 1:
        #     ref_image_latents = ref_image_latents[:1]
        context_scheduler = get_context_scheduler(context_schedule)
        
        # Denoising loop
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=(rank != 0)):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue

            noise_pred = torch.zeros(
                (latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                 *latents.shape[1:]),
                device=latents.device,
                dtype=latents.dtype,
            )
            counter = torch.zeros(
                (1, 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents.dtype
            )

            appearance_encoder(
                ref_image_latents.repeat(
                    context_batch_size * (2 if do_classifier_free_guidance else 1), 1, 1, 1),
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=t,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )

            context_queue = list(context_scheduler(
                0, num_inference_steps, latents.shape[2], context_frames, context_stride, context_overlap
            ))

            num_context_batches = math.ceil(
                len(context_queue) / context_batch_size)
            global_context = []
            for i in range(num_context_batches):
                global_context.append(
                    context_queue[i * context_batch_size: (i + 1) * context_batch_size])

            for context in global_context[rank::world_size]:
                
                controlnet_latent_input = (
                    torch.cat([latents[:, :, c] for c in context])
                    .to(device)
                    .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                )
                controlnet_latent_input = self.scheduler.scale_model_input(
                    controlnet_latent_input, t)

                # prepare inputs for controlnet
                b, c, f, h, w = controlnet_latent_input.shape
                controlnet_latent_input = rearrange(
                    controlnet_latent_input, "b c f h w -> (b f) c h w")

                controlnet_cond=torch.cat(
                        [controlnet_cond_images[c] for c in context])

                control_res_samples = self.controlnet(
                    controlnet_latent_input, 
                    controlnet_cond = rearrange(controlnet_cond.unsqueeze(0).repeat(2, 1, 1, 1, 1), "b f c h w -> (b f) c h w"),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=t,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )

                latent_model_input = (
                    torch.cat([latents[:, :, c] for c in context])
                    .to(device)
                    .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                )

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                reference_control_reader.update(reference_control_writer)
                # reference_control_reader.update(controlnet_writer)
        
                if ref_concat_image_noises_latents is not None:
                    ref_back_latent_input = (
                        torch.cat([ref_concat_image_noises_latents[:, :, c] for c in context])
                        .to(device)
                    )

                    latent_model_input = torch.cat([latent_model_input, ref_back_latent_input], dim=1)
                
                pred = self.unet(
                    latent_model_input, 
                    timestep=t, 
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    added_cond_kwargs=added_cond_kwargs,
                    control_res_samples = control_res_samples,
                    return_dict=False,
                )[0]

                reference_control_reader.clear()
                # controlnet_writer.clear()

                if do_classifier_free_guidance:
                    pred_uc, pred_c = pred.chunk(2)
                    pred = torch.cat([pred_uc.unsqueeze(0), pred_c.unsqueeze(0)])
                else:
                    pred = pred.unsqueeze(1)
                
                pred, _ = torch.split(pred, latent_channels, dim=2)

                for j, c in enumerate(context):
                    noise_pred[:, :, c] = noise_pred[:, :, c] + pred[:, j]
                    counter[:, :, c] = counter[:, :, c] + 1

            if is_dist_initialized:
                noise_pred_gathered = [torch.zeros_like(
                    noise_pred) for _ in range(world_size)]
                if rank == 0:
                    dist.gather(tensor=noise_pred,
                                gather_list=noise_pred_gathered, dst=0)
                else:
                    dist.gather(tensor=noise_pred, gather_list=[], dst=0)
                dist.barrier()

                if rank == 0:
                    for k in range(1, world_size):
                        for context in global_context[k::world_size]:
                            for j, c in enumerate(context):
                                noise_pred[:, :, c] = noise_pred[:, :,
                                                                 c] + noise_pred_gathered[k][:, :, c]
                                counter[:, :, c] = counter[:, :, c] + 1

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = (
                    noise_pred / counter).chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)
                # noise_pred = noise_pred_text
                
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs).prev_sample
            

            if is_dist_initialized:
                dist.broadcast(latents, 0)
                dist.barrier()

            if appearance_encoder is not None:
                reference_control_writer.clear()

        interpolation_factor = 1
        latents = self.interpolate_latents(
            latents, interpolation_factor, device)
        # Post-processing
        video = self.decode_latents(
            latents, rank, decoder_consistency=decoder_consistency)
        
        # # INFO vae
        # video = self.decode_latents(
        #     rearrange(controlnet_cond_images, "(b f) c h w -> b c f h w", b=1), rank, decoder_consistency=decoder_consistency)

        if is_dist_initialized:
            dist.barrier()

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)

    def train(
            self,
            prompt: Union[str, List[str]],
            prompt_embeddings: Optional[torch.FloatTensor] = None,
            video_length: Optional[int] = 8,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            timestep: Union[torch.Tensor, float, int] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator,
                                      List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            callback_steps: Optional[int] = 1,
            controlnet_condition: list = None,
            controlnet_conditioning_scale: float = 1.0,
            init_latents: Optional[torch.FloatTensor] = None,
            appearance_encoder=None,
            source_image: str = None,
            decoder_consistency=None,
            context_frames: int = 16,
            context_batch_size: int = 1,
            froce_text_embedding_zero = False,
            add_noise_image_type = "",
            **kwargs,
    ):
        """
        New args:
        - controlnet_condition          : condition map (e.g., depth, canny, keypoints) for controlnet
        - controlnet_conditioning_scale : conditioning scale for controlnet
        - init_latents                  : initial latents to begin with (used along with invert())
        - num_actual_inference_steps    : number of actual inference steps (while total steps is num_inference_steps)
        """

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        # self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)
        if init_latents is not None:
            batch_size = init_latents.shape[0]

        device = self._execution_device
        prompt_embeds = self.negative_prompt_embeds.clone()
        prompt_attention_mask = self.negative_prompt_attention_mask.clone()
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = False

        """
        LLZ TODO
        context_frames: int = 16,
        context_batch_size: int = 1,
        """
        self.reference_control_writer = ReferenceAttentionControl(appearance_encoder,
                                                            model_type="appearencenet",
                                                            do_classifier_free_guidance=do_classifier_free_guidance,
                                                            mode='write',)
        # self.controlnet_writer = ReferenceAttentionControl(self.controlnet,
        #                                                     model_type="controlnet",
        #                                                     do_classifier_free_guidance=do_classifier_free_guidance,
        #                                                     mode='write',)
        self.reference_control_reader = ReferenceAttentionControl(self.unet,
                                                            model_type="unet",
                                                            do_classifier_free_guidance=do_classifier_free_guidance,
                                                            mode='read',)

        # Prepare video
        # FIXME: verify if num_videos_per_prompt > 1 works
        assert num_videos_per_prompt == 1
        # assert batch_size == 1  # FIXME: verify if batch_size > 1 works

        # Prepare latent variables
        if init_latents is None:
            # latents = rearrange(init_latents, "(b f) c h w -> b c f h w", f=video_length)
            num_channels_latents = self.unet.in_channels


            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                height,
                width,
                prompt_attention_mask.dtype,
                device,
                generator,
                latents,
                clip_length=video_length
            )
        else:
            latents = init_latents
        del init_latents

        # print('train controlnet_condition target unique is (0, 1), real is', controlnet_condition.unique())
        # print('train source_image target unique is (-1, 1), real is', source_image.unique())


        # prepare controlnet condition input
        controlnet_cond_images = rearrange(controlnet_condition, "b f h w c -> (b f) c h w") * 2. - 1.
        controlnet_cond_images = self.vae.encode(
            controlnet_cond_images)['latent_dist'].mean * 0.18215
        
        # prepare controlnet condition input
        ref_image_latents = self.vae.encode(
            source_image)['latent_dist'].mean * 0.18215
        
        if add_noise_image_type != "":
            latents[:, :4, ...] = latents[:, :4, ...] * 0.9 + ref_image_latents.unsqueeze(2).repeat(1, 1, video_length, 1, 1) * 0.1 * self.scheduler.init_noise_sigma

        t = timestep

        # print('text_embeddings', text_embeddings.shape)
        
        """
        ref_image_latents torch.Size([2, 4, 64, 64])                                                                                                                                                  │····················
        text_embeddings torch.Size([1, 77, 768]) 
        """
        
        appearance_encoder(
            ref_image_latents,
            timestep=t, 
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )

        # prepare inputs for controlnet
        controlnet_latent_input = rearrange(
            latents, "b c f h w -> (b f) c h w")

        control_res_samples = self.controlnet(
            controlnet_latent_input, 
            controlnet_cond = controlnet_cond_images,
            timestep=t, 
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
            )

        self.reference_control_reader.update(self.reference_control_writer)
        # self.reference_control_reader.update(self.controlnet_writer)

        # predict the noise residual
        # print('timestep is', t)
        noise_pred = self.unet(
            latents, 
            timestep=t, 
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            added_cond_kwargs=added_cond_kwargs,
            control_res_samples=control_res_samples,
            return_dict=False,
        )[0]
        # noise_pred, _ = torch.split(noise_pred, latent_channels, dim=1)

        return noise_pred
    
    def clear_reference_control(self):
        if hasattr(self, "reference_control_reader"):
            self.reference_control_reader.clear()
            self.reference_control_writer.clear()
            # self.controlnet_writer.clear()
            self.reference_control_reader = None
            self.reference_control_writer = None
            # self.controlnet_writer = None
