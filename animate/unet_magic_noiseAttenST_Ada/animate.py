# Copyright 2024 Megvii inc.
#
# Copyright (2024) MegActor Authors.
#
# Megvii Inc. retain all intellectual property and proprietary rights in 
# and to this material, related documentation and any modifications thereto. 
# Any use, reproduction, disclosure or distribution of this material and related 
# documentation without an express license agreement from Megvii Inc. is strictly prohibited.

import argparse
import argparse
import datetime
import inspect
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import torch

from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate.utils import set_seed
from animate.utils.videoreader import VideoReader
from animate.utils.util import get_checkpoint
from einops import rearrange, repeat
import io
import torch.nn.functional as F

from .resnet import InflatedConv3d
from .resampler import Resampler, MLPProjModel
from .unet_controlnet import UNet3DConditionModel
from .controlnet import ControlNetModel
from .appearance_encoder import AppearanceEncoderModel


class MagicAnimate(torch.nn.Module):
    def __init__(self,
                 config="configs/training/animation.yaml",
                 device=torch.device("cuda"),
                 train_batch_size=1,
                 unet_additional_kwargs=None,
                 mixed_precision_training=False,
                 trainable_modules=[],
                 is_main_process=True,
                 weight_type=torch.float16,
                 image_finetune=True
                 ):
        super().__init__()

        if is_main_process:
            print("Initializing UNet MagicAnimate Pipeline...")
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)

        if config == "configs/training/animation.yaml":
            config = OmegaConf.load(config)
            
        controlnet_type = config.get('controlnet_type', '2d')
        controlnet_config = config.get('controlnet_config', '')
        add_noise_image_type = config.get('add_noise_image_type', "")
        concat_noise_image_type = config.get('concat_noise_image_type', "")
        appearance_controlnet_motion_checkpoint_ignore = config.get("appearance_controlnet_motion_checkpoint_ignore", {})
        concat_more_convin_channels_dict = {
            "": 0,
            "origin": 4 + 1,
            "background": 4 + 1,
            "foreground": 4 + 1,
            "origin_control": 4 + 1 + 4,
        }
        concat_more_convin_channels = concat_more_convin_channels_dict[concat_noise_image_type]
        
        remove_referencenet = config.get('remove_referencenet', False)
        _set_timestep = config.get('appearance_time_step', None)

        inference_config = OmegaConf.load(config['inference_config'])
        self.device = device
        self.weight_type = weight_type
        self.train_batch_size = train_batch_size
        self.image_finetune = image_finetune
        
        motion_module = config['motion_module']

        if unet_additional_kwargs is None:
            unet_additional_kwargs = OmegaConf.to_container(inference_config.unet_additional_kwargs)

        ### >>> create animation pipeline >>> ###
        self.tokenizer = CLIPTokenizer.from_pretrained(config['pretrained_model_path'], subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(config['pretrained_model_path'], subfolder="text_encoder")
        if config['pretrained_unet_path'] != "":
            self.unet = UNet3DConditionModel.from_pretrained_2d(config['pretrained_unet_path'],
                                                                unet_additional_kwargs=unet_additional_kwargs)
        else:
            self.unet = UNet3DConditionModel.from_pretrained_2d(config['pretrained_model_path'], subfolder="unet",
                                                                unet_additional_kwargs=unet_additional_kwargs)


        # self.image_proj_model = None
        # if self.use_image_encoder:
        #     print('init image_encoder as text clip embedding')
        #     self.image_proj_model = MLPProjModel(
        #         cross_attention_dim=768,
        #         clip_embeddings_dim=1024 + 512, # foreground and acrface
        #     )

        ########################LLZ TODO#############################
        # if "ip_ckpt" in config.keys() and config['ip_ckpt'] != "":
        #     image_proj_state_dict = torch.load(config["ip_ckpt"], map_location="cpu")["image_proj"]
        #     image_proj_state_dict = {f'image_proj_model.{k}':v for k,v in image_proj_state_dict.items()}
        #     m, u = self.unet.load_state_dict(image_proj_state_dict, strict=False)
        #     print('image_proj_state_dict keys', len(list(image_proj_state_dict.keys())))
        #     print('load pretrained image_proj',len(m),len(u))

        if 'pretrained_appearance_encoder_path' in config.keys() and config['pretrained_appearance_encoder_path'] != '':
            if is_main_process:
                print('use appearance_encoder from', config['pretrained_appearance_encoder_path'])
            self.appearance_encoder = AppearanceEncoderModel.from_pretrained(config['pretrained_appearance_encoder_path'],
                                                                            subfolder="appearance_encoder")
        else:
            if is_main_process:
                print('use appearance_encoder from unet')
            self.appearance_encoder = AppearanceEncoderModel.from_unet(self.unet)

        if _set_timestep is not None:
            if is_main_process:
                print('set appearance_encoder timestep all to zero')
            self.appearance_encoder._set_timestep = _set_timestep

        
        if 'pretrained_vae_path' in config.keys() and config['pretrained_vae_path'] != "":
            self.vae = AutoencoderKL.from_pretrained(config['pretrained_vae_path'])
        else:
            self.vae = AutoencoderKL.from_pretrained(config['pretrained_model_path'], subfolder="vae")

        if is_main_process:
            print(f'Concat some Reference: concat_noise_image_type is {concat_noise_image_type}, reconstruct unet.conv_in')
        unet_tmp_weights = self.unet.conv_in.weight.clone()
        self.unet.conv_in = InflatedConv3d(4 + 4 + concat_more_convin_channels, unet_tmp_weights.shape[0], kernel_size=3, padding=(1, 1))
        with torch.no_grad():
            self.unet.conv_in.weight[:, :4] = unet_tmp_weights # original weights
            self.unet.conv_in.weight[:, 4:] = torch.zeros(self.unet.conv_in.weight[:, 4:].shape) # new weights initialized to zero

        # if 'pretrained_controlnet_path' in config.keys() and config['pretrained_controlnet_path'] != '':
        #     self.controlnet = ControlNetModel.from_pretrained(config['pretrained_controlnet_path'])
        # else:
        #     self.controlnet = ControlNetModel()
            
        ###########################################
        # load stage1 and stage2 trained apperance_encoder, controlnet and motion module
        """
        appearance_encoder 662                                                                                                                                                             
        controlnet: 340                                                                                                                                                                     
        motion: 560  
        origin unet: 686
        """
        self.unet.enable_xformers_memory_efficient_attention()
        self.appearance_encoder.enable_xformers_memory_efficient_attention()

        if "appearance_controlnet_motion_checkpoint_path" in config.keys() and config['appearance_controlnet_motion_checkpoint_path'] != "":
            appearance_controlnet_motion_checkpoint_path = config['appearance_controlnet_motion_checkpoint_path']
            if is_main_process:
                print(f"load all model from checkpoint: {appearance_controlnet_motion_checkpoint_path}")

            with open(appearance_controlnet_motion_checkpoint_path, 'rb') as f:
                buffer = io.BytesIO(f.read())
                appearance_controlnet_motion_checkpoint_path = torch.load(buffer, map_location="cpu")
            if "global_step" in appearance_controlnet_motion_checkpoint_path and is_main_process:
                print(f"global_step: {appearance_controlnet_motion_checkpoint_path['global_step']}")
            org_state_dict = appearance_controlnet_motion_checkpoint_path["state_dict"] if "state_dict" in appearance_controlnet_motion_checkpoint_path else appearance_controlnet_motion_checkpoint_path        
            
            appearance_encoder_state_dict = {}
            unet_state_dict = {}

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
            assert len(u) == 0, print("appearance_encoder miss key is", u)

            m, u = self.unet.load_state_dict(unet_state_dict, strict=False)        
            if is_main_process:
                print(f"load checkpoint: unet missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0, print("unet missing key is", u)

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
        self.text_encoder.to(device=self.device, dtype=torch.float16)
        self.appearance_encoder.to(device=self.device, dtype=torch.float16)
        self.unet.to(device=self.device, dtype=torch.float16)
            
        scheduler = DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs))
        from .pipeline import AnimationPipeline as TrainPipeline
        self.pipeline = TrainPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            controlnet=None, 
            # image_proj_model=self.image_proj_model,
            scheduler=scheduler,
            # NOTE: UniPCMultistepScheduler
        ).to(device)
        self.scheduler = DDIMScheduler(**OmegaConf.to_container(config['noise_scheduler_kwargs']))

        if 'L' in config.keys():
            self.L = config['L']
        else:
            self.L = config['train_data']['video_length']

        if is_main_process:
            print("Initialization Done!")


    def infer(self, source_image, image_prompts, motion_sequence, random_seed, step, guidance_scale, context, size=(512, 768),froce_text_embedding_zero=False, ref_concat_image_noises_latents=None, do_classifier_free_guidance=True, add_noise_image_type="", ref_img_condition=None, visualization=False, show_progressbar=False):
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
            froce_text_embedding_zero = froce_text_embedding_zero,
            ref_concat_image_noises_latents = ref_concat_image_noises_latents,
            do_classifier_free_guidance = do_classifier_free_guidance,
            add_noise_image_type = add_noise_image_type,
            ref_img_condition=ref_img_condition,
            show_progressbar=show_progressbar,
        ).videos
        if visualization == False:
            return sample[:, :, :original_length]

        # TODO: save batch个视频
        # source_images = np.array([source_image[0].cpu()] * original_length)
        # source_images = rearrange(torch.from_numpy(source_images), "t h w c -> 1 c t h w") / 255.0


        source_images = rearrange(source_image[:1,...].repeat(original_length,1,1,1), "t h w c -> 1 c t h w") 
        source_images = (source_images+1.0)/2.0
        samples_per_video.append(source_images.cpu())

        # control = (control+1.0)/2.0
        control = rearrange(control[0], "t h w c -> 1 c t h w")
        samples_per_video.append(control[:, :, :original_length].cpu())

        samples_per_video.append(sample[:, :, :original_length])

        # samples_per_video = torch.cat(samples_per_video)

        return samples_per_video

    def forward(self, 
                init_latents, 
                image_prompts, 
                timestep, 
                source_image, 
                motion_sequence, 
                guidance_scale, 
                random_seed, 
                froce_text_embedding_zero=False, 
                ref_img_conditions=None,
                add_noise_image_type=""):
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

        samples_per_video = []

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
            context_batch_size = control.shape[0],
            guidance_scale = guidance_scale,
            froce_text_embedding_zero = froce_text_embedding_zero,
            add_noise_image_type = add_noise_image_type,
            ref_img_conditions = ref_img_conditions,
        )
        return noise_pred

    def clear_reference_control(self):
        self.pipeline.clear_reference_control()

    def preprocess_train(self, batch, image_processor, image_encoder):
            pixel_values = batch["video"].to(self.device, dtype=self.weight_type)
            ref_img_conditions = batch["reference"].clone() / 255.
            ref_img_conditions = ref_img_conditions.to(self.device, dtype=self.weight_type)
            pixel_values_ref_img = batch["reference"].byte().to(self.device, dtype=self.weight_type)
            pixel_values_ref_img = pixel_values_ref_img / 127.5 - 1.

            pixel_values_pose = batch['swapped'].to(self.device, dtype=self.weight_type) # b c h w
            pixel_values_pose = rearrange(pixel_values_pose  / 255., "b f c h w -> b f h w c") 
            batch_size = pixel_values_pose.shape[0]
            concat_poses = batch['concat_poses'].to(self.device, dtype=self.weight_type)
            concat_background = batch['concat_background'].to(self.device, dtype=self.weight_type)
            clip_conditions = batch['clip_conditions'].to(self.device, dtype=self.weight_type)

            clip_images = image_processor(
                images=rearrange(clip_conditions, "b f c h w -> (b f) c h w"), return_tensors="pt").pixel_values.to(self.device, dtype=self.weight_type)
            image_emb = image_encoder(clip_images, output_hidden_states=True).last_hidden_state
            image_emb = image_encoder.vision_model.post_layernorm(image_emb)
            image_emb = image_encoder.visual_projection(image_emb)

            pixel_values = pixel_values / 127.5 - 1
            pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist
                latents = latents.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", b=batch_size)
                latents = latents * 0.18215

                ref_concat_image_noises = concat_poses
                # Image.fromarray(ref_concat_image_noises[0].astype('uint8')).save('ref_concat_image_noise.png')
                one_img_have_more = False
                if len(ref_concat_image_noises.shape) == 5:
                    ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b f c h w -> (b f) c h w')
                    one_img_have_more = True
                ref_concat_image_noises = ref_concat_image_noises / 127.5 - 1
                # print('ref_img_backgrounds unique is', ref_img_backgrounds.unique())
                ref_concat_image_noises_latents = self.vae.encode(ref_concat_image_noises).latent_dist
                ref_concat_image_noises_latents = ref_concat_image_noises_latents.sample().unsqueeze(2)
                ref_concat_image_noises_latents = ref_concat_image_noises_latents * 0.18215
                # b c 1 h w b c f h w

                ref_img_back_mask_latents = concat_background
                H, W = ref_concat_image_noises_latents.shape[3:]
                ref_img_back_mask_latents = F.interpolate(ref_img_back_mask_latents, size=(H, W), mode='nearest').unsqueeze(2)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents) + 0.1 * torch.randn(latents.shape[0], latents.shape[1], 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            ref_concat_image_noises_latents = ref_concat_image_noises_latents.repeat(1, 1, noisy_latents.shape[2], 1, 1)
            ref_img_back_mask_latents = ref_img_back_mask_latents.repeat(1, 1, noisy_latents.shape[2], 1, 1)
            noisy_latents = torch.cat([noisy_latents, ref_concat_image_noises_latents, ref_img_back_mask_latents], dim=1)
            
            return noisy_latents, image_emb, timesteps, pixel_values_ref_img, pixel_values_pose, ref_img_conditions, noise

    def preprocess_eval(self, pixel_values_ref_img, pixel_values_pose, concat_poses, concat_background, clip_conditions, image_processor, image_encoder, guidance_scale=1.0, do_classifier_free_guidance=True):
            ref_img_conditions = pixel_values_ref_img.clone() / 255.
            ref_img_conditions = ref_img_conditions.to(self.device, dtype=self.weight_type)
            pixel_values_ref_img = pixel_values_ref_img.byte().to(self.device, dtype=self.weight_type)
            pixel_values_ref_img = (pixel_values_ref_img / 255 - 0.5) * 2

            video_length = pixel_values_pose.shape[1]
            pixel_values_pose = pixel_values_pose.to(self.device, dtype=self.weight_type) # b c h w
            pixel_values_pose = rearrange(pixel_values_pose  / 255., "b f c h w -> b f h w c") 
            concat_poses = concat_poses.to(self.device, dtype=self.weight_type)
            concat_background = concat_background.to(self.device, dtype=self.weight_type)
            clip_conditions = clip_conditions.to(self.device, dtype=self.weight_type)

            clip_images = image_processor(
                images=rearrange(clip_conditions, "b f c h w -> (b f) c h w"), return_tensors="pt").pixel_values.to(self.device, dtype=self.weight_type)
            image_emb = image_encoder(clip_images, output_hidden_states=True).last_hidden_state
            image_emb = image_encoder.vision_model.post_layernorm(image_emb)
            image_emb = image_encoder.visual_projection(image_emb)
            if guidance_scale > 1.0 and do_classifier_free_guidance:
                image_emb = torch.cat([image_emb, image_emb])

            with torch.no_grad():
                ref_concat_image_noises = concat_poses
                # Image.fromarray(ref_concat_image_noises[0].astype('uint8')).save('ref_concat_image_noise.png')
                one_img_have_more = False
                if len(ref_concat_image_noises.shape) == 5:
                    ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b f c h w -> (b f) c h w')
                    one_img_have_more = True
                ref_concat_image_noises = ref_concat_image_noises / 127.5 - 1
                # print('ref_img_backgrounds unique is', ref_img_backgrounds.unique())
                ref_concat_image_noises_latents = self.vae.encode(ref_concat_image_noises).latent_dist
                ref_concat_image_noises_latents = ref_concat_image_noises_latents.sample().unsqueeze(2)
                ref_concat_image_noises_latents = ref_concat_image_noises_latents * 0.18215
                # b c 1 h w b c f h w

                ref_img_back_mask_latents = concat_background
                # print(concat_background.shape, concat_poses.shape, clip_conditions.shape)
                H, W = ref_concat_image_noises_latents.shape[3:]
                ref_img_back_mask_latents = F.interpolate(ref_img_back_mask_latents, size=(H, W), mode='nearest').unsqueeze(2)

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)

            ref_concat_image_noises_latents = ref_concat_image_noises_latents.repeat(1, 1, video_length, 1, 1)
            ref_img_back_mask_latents = ref_img_back_mask_latents.repeat(1, 1, video_length, 1, 1)
            ref_concat_image_noises_latents = torch.cat([ref_concat_image_noises_latents, ref_img_back_mask_latents], dim=1)
            if guidance_scale > 1.0 and do_classifier_free_guidance:
                ref_concat_image_noises_latents = torch.cat([ref_concat_image_noises_latents,
                    ref_concat_image_noises_latents])
            return pixel_values_ref_img, image_emb, pixel_values_pose, ref_concat_image_noises_latents, ref_img_conditions