# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.
import argparse
import datetime
import inspect
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict
import torch.nn.functional as F

import torch

from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from animate.utils.videoreader import VideoReader
from animate.utils.util import get_checkpoint
from einops import rearrange, repeat
from refile import smart_open, smart_load_from
import io

from .resnet import InflatedConv3d
from .unet_controlnet import UNet3DConditionModel
from .appearance_encoder import AppearanceEncoderModel
from .whisper.audio2feature import load_audio_model

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MagicAnimate(torch.nn.Module):
    def __init__(self,
                 config="configs/training/animation.yaml",
                 device=torch.device("cuda"),
                 train_batch_size=1,
                 unet_additional_kwargs=None,
                 is_main_process=True,
                 **kwargs
                 ):
        super().__init__()

        if is_main_process:
            print("Initializing UNet MagicAnimate Pipeline...")
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)

        if config == "configs/training/animation.yaml":
            config = OmegaConf.load(config)
            
        _set_timestep = config.get('appearance_time_step', None)

        inference_config = OmegaConf.load(config['inference_config'])
        self.device = device
        self.train_batch_size = train_batch_size
        self.weight_type = torch.float16
        
        motion_module = config['motion_module']
        self.num_pad_audio_frames = 2

        if unet_additional_kwargs is None:
            unet_additional_kwargs = OmegaConf.to_container(inference_config.unet_additional_kwargs)

        ### >>> create animation pipeline >>> ###
        self.unet = UNet3DConditionModel.from_pretrained_2d(config['pretrained_model_path'], subfolder="unet",
                                                            unet_additional_kwargs=unet_additional_kwargs)
        self.audio_guider = load_audio_model(model_path="./weights/whisper_tiny.pt", device=device)

        if is_main_process:
            print('use appearance_encoder from unet')
        self.appearance_encoder = AppearanceEncoderModel.from_pretrained(config['pretrained_model_path'], subfolder="unet")
        self.appearance_encoder.set_last_layer()

        if _set_timestep is not None:
            if is_main_process:
                print('set appearance_encoder timestep all to zero')
            self.appearance_encoder._set_timestep = _set_timestep

        
        if 'pretrained_vae_path' in config.keys() and config['pretrained_vae_path'] != "":
            self.vae = AutoencoderKL.from_pretrained(config['pretrained_vae_path'])
        else:
            self.vae = AutoencoderKL.from_pretrained(config['pretrained_model_path'], subfolder="vae")

        self.unet.conv_in = InflatedConv3d(4 + 4, 320, kernel_size=3, padding=(1, 1))

        """
        appearance_encoder 662                                                                                                                                                             
        controlnet: 340                                                                                                                                                                     
        motion: 560  
        origin unet: 686
        """
        if torch.cuda.is_available():
            self.unet.enable_xformers_memory_efficient_attention()
            self.appearance_encoder.enable_xformers_memory_efficient_attention()

        if "appearance_controlnet_motion_checkpoint_path" in config.keys() and config['appearance_controlnet_motion_checkpoint_path'] != "":
            appearance_controlnet_motion_checkpoint_path = config['appearance_controlnet_motion_checkpoint_path']
            if is_main_process:
                print(f"load all model from checkpoint: {appearance_controlnet_motion_checkpoint_path}")

            with smart_open(appearance_controlnet_motion_checkpoint_path, 'rb') as f:
                buffer = io.BytesIO(f.read())
                appearance_controlnet_motion_checkpoint_path = torch.load(buffer, map_location="cpu")
            if "global_step" in appearance_controlnet_motion_checkpoint_path and is_main_process:
                print(f"global_step: {appearance_controlnet_motion_checkpoint_path['global_step']}")
            org_state_dict = appearance_controlnet_motion_checkpoint_path["state_dict"] if "state_dict" in appearance_controlnet_motion_checkpoint_path else appearance_controlnet_motion_checkpoint_path        
            
            appearance_encoder_state_dict = {}
            unet_state_dict = {}
            audio_projection_state_dict = {}

            for name, param in org_state_dict.items():
                if "appearance_encoder." in name:
                    if name.startswith('module.appearance_encoder.'):
                        name = name.split('module.appearance_encoder.')[-1]
                    appearance_encoder_state_dict[name] = param
                if "unet." in name:
                    if name.startswith('module.unet.'):
                        name = name.split('module.unet.')[-1]
                    unet_state_dict[name] = param

            if is_main_process:
                print('load checkpoint: appearance_encoder_state_dict', len(list(appearance_encoder_state_dict.keys())))
                print('load checkpoint: unet_state_dict', len(list(unet_state_dict.keys())))


            m, u = self.appearance_encoder.load_state_dict(appearance_encoder_state_dict, strict=False)
            if is_main_process:
                print(f"load checkpoint: appearance_encoder missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0, print("appearance_encoder unexpecting key is", u)

            # unet_tmp_weights = self.unet.conv_in.weight.clone()
            # with torch.no_grad():
            #     self.unet.conv_in.weight[:, :4] = unet_tmp_weights # original weights
            #     self.unet.conv_in.weight[:, 4:] = torch.zeros(self.unet.conv_in.weight[:, 8:12].shape) # new weights initialized to zero

            m, u = self.unet.load_state_dict(unet_state_dict, strict=False)        
            if is_main_process:
                print(f"load checkpoint: unet missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0, print("unet unexpecting key is", u)



        # 1. unet ckpt
        # 1.1 motion module
        if unet_additional_kwargs['use_motion_module'] and motion_module != "":
            if is_main_process:
                print('load motion_module from', motion_module)
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")
            if "global_step" in motion_module_state_dict: func_args.update(
                {"global_step": motion_module_state_dict["global_step"]})
            motion_module_state_dict = motion_module_state_dict[
                'state_dict'] if 'state_dict' in motion_module_state_dict else motion_module_state_dict
            try:
                # extra steps for self-trained models
                state_dict = OrderedDict()
                for key in motion_module_state_dict.keys():
                    if key.startswith("module."):
                        _key = key.split("module.")[-1]
                        state_dict[_key] = motion_module_state_dict[key]
                    else:
                        state_dict[key] = motion_module_state_dict[key]
                motion_module_state_dict = state_dict
                del state_dict
                if is_main_process:
                    print(f'load motion_module params len is {len(motion_module_state_dict)}')
                missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
                if is_main_process:
                    print(f'load motion_module missing {len(missing)}, unexpected {len(unexpected)}')
                assert len(unexpected) == 0
            except:
                _tmp_ = OrderedDict()
                for key in motion_module_state_dict.keys():
                    if "motion_modules" in key:
                        if key.startswith("unet."):
                            _key = key.split('unet.')[-1]
                            _tmp_[_key] = motion_module_state_dict[key]
                        else:
                            _tmp_[key] = motion_module_state_dict[key]
                if is_main_process:            
                    print(f'load motion_module params len is {len(_tmp_)}')            
                missing, unexpected = self.unet.load_state_dict(_tmp_, strict=False)
                if is_main_process:
                    print(f'load motion_module missing {len(missing)}, unexpected {len(unexpected)}')
                assert len(unexpected) == 0
                del _tmp_
            del motion_module_state_dict

        
        self.vae.to(device=self.device, dtype=torch.float16)
        self.appearance_encoder.to(device=self.device, dtype=torch.float16)
        self.unet.to(device=self.device, dtype=torch.float16)
        scheduler = DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs))
        from .pipeline import AnimationPipeline as TrainPipeline
        self.pipeline = TrainPipeline(
            vae=self.vae, unet=self.unet,
            audio_guider=self.audio_guider,
            controlnet=None, 
            # image_proj_model=self.image_proj_model,
            scheduler=scheduler,
            # NOTE: UniPCMultistepScheduler
        ).to(device)
        self.pipeline.load_empty_str_embedding(config["empty_str_embedding"])
        self.scheduler = DDIMScheduler(**OmegaConf.to_container(config['noise_scheduler_kwargs']))

        if 'L' in config.keys():
            self.L = config['L']
        else:
            self.L = config['train_data']['video_length']

        if is_main_process:
            print("Initialization Done!")

    def infer(self, source_image, image_prompts, motion_sequence,
                random_seed, step, guidance_scale, context,
                size=(512, 768),
                audio_signal=None,
                fps=30, **kwargs):
        prompt = n_prompt = ""
        step = int(step)
        guidance_scale = float(guidance_scale)
        samples_per_video = []

        set_seed(random_seed)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(torch.initial_seed())

        if isinstance(motion_sequence, str):
            if motion_sequence.endswith('.mp4'):
                control = VideoReader(motion_sequence).read()
                if control[0].shape[0] != size:
                    control = [np.array(Image.fromarray(c).resize(size)) for c in control]
                control = np.array(control)
            # TODO: 需要过一遍dwpose啊！！！！
                
        else:
            control = motion_sequence

        source_image = rearrange(source_image, 'b c h w -> b h w c')
        # print('source image first is', source_image)
        # print('control first is', control)
        # if source_image.shape[0] != size:
        #     source_image = np.array(Image.fromarray(source_image).resize((size, size)))
        B, H, W, C = source_image.shape

        init_latents = None
        original_length = control.shape[1]
        # offset = control.shape[1] % self.L
        # if offset > 0:
        #     control= control[:,:-offset,...]
            # control = np.pad(control, ((0, self.L - control.shape[0] % self.L), (0, 0), (0, 0), (0, 0)), mode='edge')

        context_frames = context["context_frames"]
        context_stride = context["context_stride"]
        context_overlap = context["context_overlap"]
        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            prompt_embeddings=image_prompts,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            video_length=control.shape[1],
            controlnet_condition=control,
            init_latents=init_latents,
            generator=generator,
            appearance_encoder=self.appearance_encoder,
            source_image=source_image,
            context_frames = context_frames,
            context_stride = context_stride,
            context_overlap = context_overlap,
            num_pad_audio_frames = self.num_pad_audio_frames,
            audio_signal = audio_signal,
            fps = fps,
        ).videos

        source_images = rearrange(source_image[:1,...].repeat(original_length,1,1,1), "t h w c -> 1 c t h w") 
        source_images = (source_images + 1.0) / 2.0
        samples_per_video.append(source_images.cpu())

        # control = (control+1.0)/2.0
        control = rearrange(control[0], "t h w c -> 1 c t h w")
        samples_per_video.append(control[:, :, :original_length].cpu())

        samples_per_video.append(sample[:, :, :original_length])

        # samples_per_video = torch.cat(samples_per_video)

        return samples_per_video

    def forward(self, 
                target,
                init_latents, 
                image_prompts, 
                timestep, 
                source_image, 
                motion_sequence, 
                guidance_scale, 
                audio_signal=None,
                frame_stride = None,
                clip_start = None,
                clip_end = None,
                start_idx = None,
                fps = None,
                **kwargs
                ):
        """
        :param init_latents: the most important input during training
        :param timestep: another important input during training
        :param source_image: an image in np.array (b, c, h, w)
        :param motion_sequence: np array, (b, f, h, w, c) (0, 255)
        :param random_seed:
        :param size: width=512, height=768 by default
        :return:
        """
        prompt = n_prompt = ""
        control = motion_sequence
        B, C, H, W  = source_image.shape

        generator = torch.Generator(device=self.device)
        generator.manual_seed(torch.initial_seed())


        noise_pred = self.pipeline.train(
            prompt,
            prompt_embeddings=image_prompts,
            negative_prompt=n_prompt,
            timestep=timestep,
            width=W,
            height=H,
            video_length=control.shape[1],
            controlnet_condition=control,
            init_latents=init_latents,  # add noise to latents
            generator=generator,
            appearance_encoder=self.appearance_encoder,
            source_image=source_image,
            context_frames = control.shape[1],
            context_batch_size = 1,
            guidance_scale = guidance_scale,
            audio_signal = audio_signal,
            frame_stride = frame_stride,
            clip_start = clip_start,
            clip_end = clip_end,
            start_idx = start_idx,
            fps = fps
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    def clear_reference_control(self):
        self.pipeline.clear_reference_control()


    def preprocess_train(self, batch, use_both_ratio=0.5, use_audio_ratio=0.5):
            use_both = np.random.rand() < use_both_ratio
            use_only_audio = np.random.rand() < use_audio_ratio
            frame_stride = batch['frame_stride']
            clip_start = batch["clip_start"]
            clip_end = batch["clip_end"]
            start_idx = batch["start_idx"]
            fps = batch["fps"]
            pixel_values = batch["video"].to(self.device, dtype=self.weight_type)
            ref_img_conditions = batch["reference"].clone() / 255.
            ref_img_conditions = ref_img_conditions.to(self.device, dtype=self.weight_type)
            pixel_values_ref_img = batch["reference"].byte().to(self.device, dtype=self.weight_type)
            pixel_values_ref_img = pixel_values_ref_img / 127.5 - 1.

            audio_signal = batch["audio_signal"].to(self.device, dtype=self.weight_type)
            
            if use_both:
                pixel_values_pose = batch['frames_eye'].to(self.device, dtype=self.weight_type) # b c h w
            elif use_only_audio:
                pixel_values_pose = batch['frames_eye_mouth'].to(self.device, dtype=self.weight_type) # b c h w
                pixel_values_pose[...] = 0
            else:
                pixel_values_pose = batch['frames_eye_mouth'].to(self.device, dtype=self.weight_type) # b c h w
                audio_signal[...] = 0
                
            pixel_values_pose = rearrange(pixel_values_pose  / 255., "b f c h w -> b f h w c") 
            batch_size = pixel_values_pose.shape[0]

            pixel_values = pixel_values / 127.5 - 1
            pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")

            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist
                latents = latents.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", b=batch_size)
                latents = latents * 0.18215

            noise = torch.randn_like(latents) + 0.1 * torch.randn(latents.shape[0], latents.shape[1], 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            
            return dict(
                image_prompts=None,
                target=noise.float(),
                init_latents=noisy_latents,
                timestep=timesteps,
                guidance_scale=1.0,
                source_image=pixel_values_ref_img, 
                motion_sequence=pixel_values_pose,
                audio_signal=audio_signal,
                # mask_mouth=mask_mouth,
                frame_stride = frame_stride,
                clip_start = clip_start,
                clip_end = clip_end,
                start_idx = start_idx,
                fps = fps
            )
            
    def preprocess_eval(self, face_mask, pixel_values_ref_img, pixel_values_pose, audio_signal, guidance_scale=1.0, do_classifier_free_guidance=True, **kwargs):
            ref_img_conditions = pixel_values_ref_img.clone() / 255.
            ref_img_conditions = ref_img_conditions.to(self.device, dtype=self.weight_type)
            pixel_values_ref_img = pixel_values_ref_img.byte().to(self.device, dtype=self.weight_type)
            pixel_values_ref_img = (pixel_values_ref_img / 255 - 0.5) * 2

            video_length = pixel_values_pose.shape[1]
            pixel_values_pose = pixel_values_pose.to(self.device, dtype=self.weight_type) # b c h w
            pixel_values_pose = rearrange(pixel_values_pose  / 255., "b f c h w -> b f h w c") 

            return dict(
                source_image=pixel_values_ref_img,
                motion_sequence=pixel_values_pose,
                audio_signal=audio_signal,
                image_prompts=None,
            )
