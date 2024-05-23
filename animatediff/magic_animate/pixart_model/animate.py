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

import torch

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from accelerate.utils import set_seed
from animatediff.utils.videoreader import VideoReader
from einops import rearrange, repeat
from megfile import smart_open
import io

from .dit_attention import DiTNetModel
from .dit_controlnet import DiTControlNetModel
from .dit_appearence import DiTAppearenceEncoder
from .pipeline import AnimationPipeline as TrainPipeline



class MagicAnimate(torch.nn.Module):
    def __init__(self,
                 config="configs/training/animation.yaml",
                 device=torch.device("cuda"),
                 train_batch_size=1,
                 unet_additional_kwargs=None,
                 mixed_precision_training=False,
                 trainable_modules=[],
                 is_main_process=True,
                 ):
        super().__init__()

        if is_main_process:
            print("Initializing DiT Transformer MagicAnimate Pipeline...")
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
        assert concat_noise_image_type == "", "For DiT Transformer, not support any concat"
        assert config['control_aux_type'] != "densepose_dwpose_concat", \
            "For DiT Transformer, not support concat for control"
        remove_referencenet = config.get('remove_referencenet', False)

        self.device = device
        self.train_batch_size = train_batch_size

        motion_module = config['motion_module']
        noise_scheduler_kwargs = config['noise_scheduler_kwargs']

        assert unet_additional_kwargs is not None, "unet_additional_kwargs is None, we need this"
        
        
        ### >>> create animation pipeline >>> ###
        self.tokenizer = None
        self.text_encoder = None
        # self.tokenizer = CLIPTokenizer.from_pretrained(config['pretrained_model_path'], subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(config['pretrained_model_path'], subfolder="text_encoder")
        if config['pretrained_unet_path'] != "":
            self.unet = DiTNetModel.from_pretrained_2d(config['pretrained_unet_path'],
                                                                unet_additional_kwargs=unet_additional_kwargs)
        else:
            self.unet = DiTNetModel.from_pretrained_2d(config['pretrained_model_path'], subfolder="transformer",
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

        if not remove_referencenet:
            if 'pretrained_appearance_encoder_path' in config.keys() and config['pretrained_appearance_encoder_path'] != '':
                self.appearance_encoder = DiTAppearenceEncoder.from_pretrained(config['pretrained_appearance_encoder_path'],
                                                                                subfolder="appearance_encoder")
                
            else:
                if is_main_process:
                    print('use appearance_encoder from unet')
                self.appearance_encoder = DiTAppearenceEncoder.from_dit(self.unet)

        else:
            self.appearance_encoder = None
        
        if 'pretrained_vae_path' in config.keys() and config['pretrained_vae_path'] != "":
            self.vae = AutoencoderKL.from_pretrained(config['pretrained_vae_path'])
        else:
            self.vae = AutoencoderKL.from_pretrained(config['pretrained_model_path'], subfolder="vae")

        if is_main_process:
            print('load controlnet type is', controlnet_type)
        ### Load controlnet

        if 'pretrained_controlnet_path' in config.keys() and config['pretrained_controlnet_path'] != '':
            self.controlnet = DiTControlNetModel.from_pretrained(config['pretrained_controlnet_path'])
        else:
            if is_main_process:
                print('use controlnet from unet')
            self.controlnet = DiTControlNetModel.from_dit(self.unet)
            
        if config['control_aux_type'] == "densepose_dwpose_concat":
            if is_main_process:
                print('for using concat control video, we addjust control net conv_in')
            cond_embedding_conv_in_weight = self.controlnet.controlnet_cond_embedding.conv_in.weight.clone()
            self.controlnet.controlnet_cond_embedding.conv_in = torch.nn.Conv2d(4,
                self.controlnet.controlnet_cond_embedding.adjust_block_out_channels_0,
                kernel_size=3,
                padding=1)
            with torch.no_grad():
                self.controlnet.controlnet_cond_embedding.conv_in.weight[:, :3] = cond_embedding_conv_in_weight
                self.controlnet.controlnet_cond_embedding.conv_in.weight[:, 3] = \
                    torch.zeros_like(self.controlnet.controlnet_cond_embedding.conv_in.weight[:, 3])
            
        if concat_noise_image_type != "":
            if is_main_process:
                print(f'concat_noise_image_type is {concat_noise_image_type}, reconstruct unet.conv_in')
            unet_tmp_weights = self.unet.conv_in.weight.clone()
            self.unet.conv_in = InflatedConv3d(4 + concat_more_convin_channels, unet_tmp_weights.shape[0], kernel_size=3, padding=(1, 1))
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
        if not remove_referencenet:
            self.appearance_encoder.enable_xformers_memory_efficient_attention()
        self.controlnet.enable_xformers_memory_efficient_attention()

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
            controlnet_state_dict = {}
            unet_state_dict = {}
            # image_proj_state_dict = {}

            for name, param in org_state_dict.items():
                if "appearance_encoder." in name:
                    if name.startswith('module.appearance_encoder.'):
                        name = name.split('module.appearance_encoder.')[-1]
                    appearance_encoder_state_dict[name] = param
                if "controlnet." in name:
                    if name.startswith('module.controlnet.'):
                        name = name.split('module.controlnet.')[-1]
                    controlnet_state_dict[name] = param
                if "unet." in name:
                    if name.startswith('module.unet.'):
                        name = name.split('module.unet.')[-1]
                    unet_state_dict[name] = param
                # if "image_proj_model." in name:
                #     if name.startswith('module.image_proj_model.'):
                #         name = name.split('module.image_proj_model.')[-1]
                #     image_proj_state_dict[name] = param

            if is_main_process:
                print('load checkpoint: appearance_encoder_state_dict', len(list(appearance_encoder_state_dict.keys())))
                print('load checkpoint: controlnet_state_dict', len(list(controlnet_state_dict.keys())))
                print('load checkpoint: unet_state_dict', len(list(unet_state_dict.keys())))
            # print('image_proj_model', len(list(image_proj_state_dict.keys())))    

            if not remove_referencenet:
                if "appearance_encoder" not in appearance_controlnet_motion_checkpoint_ignore:
                    m, u = self.appearance_encoder.load_state_dict(appearance_encoder_state_dict, strict=False)
                else:
                    if is_main_process:
                        print(f"ignore appearance_encoder checkpoint")
                
                if is_main_process:
                    print(f"load checkpoint: appearance_encoder missing keys: {len(m)}, unexpected keys: {len(u)}")
                assert len(u) == 0
            if "controlnet" not in appearance_controlnet_motion_checkpoint_ignore:
                m, u = self.controlnet.load_state_dict(controlnet_state_dict, strict=False)
            else:
                if is_main_process:
                    print(f"ignore controlnet checkpoint")
            if is_main_process:
                print(f"load checkpoint: controlnet missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0
            if "unet" not in appearance_controlnet_motion_checkpoint_ignore:
                m, u = self.unet.load_state_dict(unet_state_dict, strict=False)
            else:
                if is_main_process:
                    print(f"ignore unet checkpoint")            
            if is_main_process:
                print(f"load checkpoint: unet missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0

            # if self.use_image_encoder:
            #     m, u = self.image_proj_model.load_state_dict(image_proj_state_dict, strict=False)
            #     print(f"image_proj_model missing keys: {len(m)}, unexpected keys: {len(u)}")
            #     assert len(u) == 0
        #     image_proj_state_dict = {f'image_proj_model.{k}':v for k,v in image_proj_state_dict.items()}

        ###########################################

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
        # self.text_encoder.to(device=self.device, dtype=torch.float16)
        self.appearance_encoder.to(device=self.device, dtype=torch.float16)
        self.controlnet.to(device=self.device, dtype=torch.float16)
        self.unet.to(device=self.device, dtype=torch.float16)

        # if mixed_precision_training and self.use_image_encoder:
        #     self.image_proj_model.to(device=self.device, dtype=torch.float32)
        # elif self.use_image_encoder:
        #     self.image_proj_model.to(device=self.device, dtype=torch.float16)
        
        # if not remove_referencenet:
        #     if mixed_precision_training and 'appearance_encoder' in trainable_modules:
        #         self.appearance_encoder.to(device=self.device, dtype=torch.float32)
        #     else:
        #         self.appearance_encoder.to(device=self.device, dtype=torch.float16)

        # if mixed_precision_training and 'controlnet' in trainable_modules:
        #     self.controlnet.to(device=self.device, dtype=torch.float32)
        # else:
        #     self.controlnet.to(device=self.device, dtype=torch.float16)

        # if mixed_precision_training and \
        #     ('motion_module' in trainable_modules or 'unet' in trainable_modules):
        #     self.unet.to(device=self.device, dtype=torch.float32)
        # else:
        #     self.unet.to(device=self.device, dtype=torch.float16)
        
        # if mixed_precision_training and 'unet.conv_in' in trainable_modules:
        #     self.unet.conv_in.to(device=self.device, dtype=torch.float32)
            

        self.pipeline = TrainPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            controlnet=self.controlnet, 
            # image_proj_model=self.image_proj_model,
            scheduler=DPMSolverMultistepScheduler(**OmegaConf.to_container(noise_scheduler_kwargs)),
            # NOTE: UniPCMultistepScheduler
        ).to(device)
        self.pipeline.enable_xformers_memory_efficient_attention()

        if 'L' in config.keys():
            self.L = config['L']
        else:
            self.L = config['train_data']['video_length']

        if is_main_process:
            print("Initialization Done!")

    def infer(self, source_image, image_prompts, motion_sequence, random_seed, step, guidance_scale, context, size=(512, 768),froce_text_embedding_zero=False, ref_concat_image_noises_latents=None, do_classifier_free_guidance=True, add_noise_image_type=""):
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
        ).videos

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

    def forward(self, init_latents, image_prompts, timestep, source_image, motion_sequence, guidance_scale, random_seed,froce_text_embedding_zero=False, add_noise_image_type=""):
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
            context_batch_size = 1,
            guidance_scale = guidance_scale,
            froce_text_embedding_zero = froce_text_embedding_zero,
            add_noise_image_type = add_noise_image_type,
        )

        return noise_pred

    def clear_reference_control(self):
        self.pipeline.clear_reference_control()