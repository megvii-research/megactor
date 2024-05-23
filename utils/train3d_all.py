import os
import math
import random
import logging
import inspect
import argparse
import datetime
import threading
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
# from ip_adapter import IPAdapterFull

# from animatediff.data.dataset import WebVid10M, PexelsDataset
from animatediff.utils.util import save_videos_grid, pad_image, generate_random_params, apply_transforms
from animatediff.utils.util import get_condition, crop_and_resize_tensor_face, crop_and_resize_tensor
from accelerate import Accelerator
from einops import repeat
from accelerate.utils import set_seed
import webdataset as wds

from face_dataset import S3VideosIterableDataset
import webdataset as wds
import facer

def main(
        image_finetune: bool,

        origin_config,
        name: str,
        use_wandb: bool,
        launcher: str,

        output_dir: str,

        data_module: str,
        data_class: str,
        train_data: Dict,
        validation_data: Dict,
        context: Dict,
        cfg_random_null_text: bool = True,
        cfg_random_null_text_ratio: float = 0.1,

        pretrained_model_path: str = "",
        pretrained_appearance_encoder_path: str = "",
        pretrained_controlnet_path: str = "",
        pretrained_vae_path: str = "",
        motion_module: str = "",
        appearance_controlnet_motion_checkpoint_path: str = "",
        pretrained_unet_path: str = "",
        inference_config: str = "",
        unet_checkpoint_path: str = "",
        unet_additional_kwargs: Dict = {},
        ema_decay: float = 0.9999,
        noise_scheduler_kwargs=None,

        max_train_epoch: int = -1,
        max_train_steps: int = 100,
        validation_steps: int = 100,
        validation_steps_tuple: Tuple = (-1,),

        learning_rate: float = 3e-5,
        scale_lr: bool = False,
        lr_warmup_steps: int = 0,
        lr_scheduler: str = "constant",

        trainable_modules: Tuple[str] = (None,),
        num_workers: int = 8,
        train_batch_size: int = 1,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        checkpointing_epochs: int = 5,
        checkpointing_steps: int = -1,

        mixed_precision_training: bool = True,
        enable_xformers_memory_efficient_attention: bool = True,

        valid_seed: int = 42,
        is_debug: bool = False,

        dwpose_only_face = False,
        froce_text_embedding_zero = False,

        ip_ckpt=None,
        train_cross_rate_step: Dict = {
            0: 0.2
        },
        train_frame_number_step: Dict = {
            0: 16
        },
        train_warp_rate_step: Dict = {
            0: 0.
        },
        control_aux_type: str = 'dwpose',
        controlnet_type: str = '2d',
        controlnet_config: str = '',

        model_type: str = "unet",
        warp_condition_use: bool = False,
        clip_image_type: str = '',
        remove_background: bool = False,
        concat_noise_image_type: str = '',
        ref_image_type: str = "origin",
        do_classifier_free_guidance: bool = True,
        add_noise_image_type: str = '',
        special_ref_index: int = -1,
        remove_referencenet: bool = False,
        appearance_controlnet_motion_checkpoint_ignore: Dict = {},
        resume_step_offset: int = 0,
        appearance_time_step = None,
        recycle_seed: int = 100000000
):

    # check params is true to run
    assert train_batch_size == 1, "For crop face to center, this code only support train_batch_size == 1"

    weight_type = torch.float16
    # Accelerate
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)



    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        # OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    if accelerator.state.deepspeed_plugin is not None and \
            accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto":
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = train_batch_size

    # Load tokenizer and models.
    
    local_rank = accelerator.device

    # load dwpose detector, see controlnet_aux: https://github.com/patrickvonplaten/controlnet_aux
    # specify configs, ckpts and device, or it will be downloaded automatically and use cpu by default
    
    from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor
    dwpose_model = DenseDWposePredictor(local_rank)
    if model_type in ["unet_condition", "unet", 
                        "unet_guide_inAttenST", "unet_guide_noiseAttenST", 
                        "unet_magic_noiseAttenST"]:
        if model_type in ["unet"]:
            from animatediff.magic_animate.unet_model.animate import MagicAnimate
        if model_type in ["unet_condition"]:
            from animatediff.magic_animate.unet_condition.animate import MagicAnimate
        if model_type in ["unet_guide_inAttenST"]:
            from animatediff.magic_animate.unet_guide_inAttenST.animate import MagicAnimate
        if model_type in ["unet_guide_noiseAttenST"]:
            from animatediff.magic_animate.unet_guide_noiseAttenST.animate import MagicAnimate
        if model_type in ["unet_magic_noiseAttenST"]:
            from animatediff.magic_animate.unet_magic_noiseAttenST.animate import MagicAnimate

        if accelerator.is_main_process:
            print("using mse_loss")
        def train_loss_func(model_output,
                            x_t,
                            x_start,
                            target,
                            timestep,
                            ):
            return F.mse_loss(model_output.float(), target.float(), reduction="mean")
    elif model_type in ["dit", "dit_back_control", "dit_back_inputme", "dit_inputme", 
                        "pixart", "pixart_control", "dit_refer"]:
        if model_type in ["dit_back_control"]:
            from animatediff.magic_animate.dit_model_back_control.animate import MagicAnimate
        if model_type in ["dit"]:
            from animatediff.magic_animate.dit_model.animate import MagicAnimate
        if model_type in ["dit_back_inputme"]:
            from animatediff.magic_animate.dit_model_back_inputme.animate import MagicAnimate
        if model_type in ["dit_inputme"]:
            from animatediff.magic_animate.dit_model_inputme.animate import MagicAnimate
        if model_type in ["dit_refer"]:
            from animatediff.magic_animate.dit_model_refer.animate import MagicAnimate
        if model_type in ["pixart"]:
            from animatediff.magic_animate.pixart_model.animate import MagicAnimate
        if model_type in ["pixart_control"]:
            from animatediff.magic_animate.pixart_model_control.animate import MagicAnimate
        from animatediff.utils.variational_lower_bound_loss import VariationalLowerBoundLoss
        num_train_timesteps = config["noise_scheduler_kwargs"]["num_train_timesteps"] if "num_train_timesteps" in config["noise_scheduler_kwargs"] else 1000
        vlb_loss_func = VariationalLowerBoundLoss(
            num_train_timesteps = num_train_timesteps,
            beta_start = config["noise_scheduler_kwargs"]["beta_start"],
            beta_end = config["noise_scheduler_kwargs"]["beta_end"],
            beta_schedule = config["noise_scheduler_kwargs"]["beta_schedule"],
        )
        if accelerator.is_main_process:
            print("using VariationalLowerBoundLoss and mse_loss")
        def train_loss_func(model_output,
                            x_t,
                            x_start,
                            target,
                            timestep,
                            ):
            mse_loss = F.mse_loss(model_output[:, :4, ...], target, reduction="mean")
            vlb_loss = vlb_loss_func(model_output, x_t, x_start, timestep).mean()
            return mse_loss * 0.5 + vlb_loss * 0.5

    model = MagicAnimate(config=config,
                         train_batch_size=train_batch_size,
                         device=local_rank,
                         unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs),
                         mixed_precision_training=True,
                         trainable_modules=trainable_modules,
                         is_main_process=accelerator.is_main_process,)
    # Load noise_scheduler
    noise_scheduler = model.scheduler
    # ----- load image encoder ----- #
    """
    使用IP-adapter，主要包含image_encoder，clip_image_processor和image_proj_model
    image_proj_model在Resampler里定义
    """
    arcface_encoder = None
    image_processor = None
    image_encoder = None
    face_detector = None
    if clip_image_type != "":
        if accelerator.is_main_process:
            print(f"use clip code image, image type is {clip_image_type}")
        image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_path, subfolder="feature_extractor")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="image_encoder")
        
        # from controlnet_resource.arcface_backbones import get_model
        # arcface_encoder = get_model('r100', fp16=False)
        # arcface_weight_path = '/root/.cache/yangshurong/magic_pretrain/arcface_backbone.pth'
        # arcface_encoder.load_state_dict(torch.load(arcface_weight_path))
        # arcface_encoder.to(local_rank, weight_type)
        # arcface_encoder.requires_grad_(False)
        # arcface_encoder.eval()

        image_encoder.to(local_rank, weight_type)
        image_encoder.requires_grad_(False)
        face_detector = facer.face_detector('retinaface/mobilenet', device=local_rank)
        face_detector.requires_grad_(False)

    # Set trainable parameters
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if accelerator.is_main_process:
        print('trainable_params', len(trainable_params))
    # with open('model.txt', 'w') as fp:
    #     for item in list(model.state_dict().keys()):
    #         fp.write("%s\n" % item)
    # fp.close()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if accelerator.is_main_process:
        accelerator.print(f"trainable params number: {len(trainable_params)}")
        accelerator.print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
        model.appearance_encoder.enable_gradient_checkpointing()
        model.controlnet.enable_gradient_checkpointing()

    
    model.to(local_rank)
    
    def worker_init_fn(_):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        worker_id = worker_info.id
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

    video_length = train_data["video_length"]
    special_ref_index = video_length // 2
    resolution     = train_data['resolution']
    train_dataset = S3VideosIterableDataset(
        data_dirs = train_data['data_dirs'],
        video_length   = train_data['frame_stride'] * train_data["video_length"],
        resolution     = resolution,
        frame_stride   = train_data['frame_stride'],
        dataset_length = 1000000,
        shuffle        = train_data['shuffle'],
        resampled      = train_data['resampled'],
        image_processor = None,
    )

    train_dataloader = wds.WebLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        # this must be zeros since in mul GPU
        collate_fn = None,
    ).with_length(len(train_dataset))

    # save train config
    if accelerator.is_main_process:
        OmegaConf.save(origin_config, f"{output_dir}/config.yaml")
        subprocess.run(['aws',
                        f'--endpoint-url={train_dataset.endpoint_url}',
                        's3',
                        'cp',
                        f"{output_dir}/config.yaml",
                        f"s3://radar/yangshurong/{folder_name}/"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    num_processes = torch.cuda.device_count()
    if accelerator.is_main_process:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {num_train_epochs}")
        print(f"  Instantaneous batch size per device = {train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_train_steps}")
        print(f"  num_processes = {num_processes}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    seed = 0
    set_seed((seed + resume_step_offset) % recycle_seed)

    def find_lower_bound(d, x):
        lower_keys = [k for k in d.keys() if k <= x]
        assert lower_keys is not None, f"Can not find any one for step {x}"
        closest_key = max(lower_keys)
        return d[closest_key]

    warp_params = generate_random_params(resolution[0], resolution[1])

    for epoch in range(first_epoch, num_train_epochs):
        # TODO: check webdataset在多卡的随机性问题
        # train_dataloader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            video_length = find_lower_bound(train_frame_number_step, step + resume_step_offset)
            use_2d_cross_rate = find_lower_bound(train_cross_rate_step, step + resume_step_offset)
            warp_rate = find_lower_bound(train_warp_rate_step, step + resume_step_offset)
            with accelerator.accumulate(model):
                # Data batch sanity check
                if global_step % 1000 == 0:
                    # pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                    pixel_values = batch['pixel_values'].cpu()
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, pixel_value in enumerate(pixel_values):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value,
                                        f"{output_dir}/sanity_check/global_{global_step}_pixel_value.gif")

                ### >>>> Training >>>> ###

                # Get video data as origin       
                pixel_values = batch["pixel_values"].to(local_rank, dtype=weight_type)
                pixel_values_pose = batch["dwpose_all"].to(local_rank, dtype=weight_type)

                # Standard: Detect face, make face to the center, and chose whether use warp
                pixel_values_pose = rearrange(pixel_values_pose, "b f c h w -> b f h w c")
                if warp_rate > np.random.rand() and warp_condition_use:
                    pixel_values_pose = rearrange(pixel_values_pose, "b f c h w -> (b f) c h w")
                    pixel_values_pose = apply_transforms(pixel_values_pose, warp_params)
                    pixel_values_pose = rearrange(pixel_values_pose, "(b f) c h w -> b f h w c", b=train_batch_size).to(local_rank, dtype=weight_type)
                pixel_values_pose = pixel_values_pose / 255.
                # Chose ref-image
                pixel_values_ref_img = pixel_values[:, special_ref_index, ...]
                train_cross_rate = np.random.rand()

                # NOTE: not use_2d_cross_rate, pixel_values all set to pixel_values_ref_img

                if video_length == 1:
                    pixel_index = special_ref_index if train_cross_rate > use_2d_cross_rate else special_ref_index + 4
                    pixel_values = pixel_values[:, pixel_index: pixel_index+1, ...]
                    pixel_values_pose = pixel_values_pose[:, pixel_index: pixel_index+1, ...]
                elif train_cross_rate > use_2d_cross_rate or video_length == pixel_values.shape[1]:
                    pixel_values = pixel_values[:, :video_length, ...]
                    pixel_values_pose = pixel_values_pose[:, :video_length, ...]
                else:
                    rng = np.random.default_rng(seed=(((seed + resume_step_offset) % recycle_seed) * num_processes + accelerator.process_index))
                    frames_num = pixel_values.shape[1]
                    segment_num = frames_num // video_length 
                    segment_index = rng.integers(1, segment_num)
                    start_frame = segment_index * video_length 
                    end_frame = start_frame + video_length 
                    pixel_values = pixel_values[:, start_frame:end_frame, ...]
                    pixel_values_pose = pixel_values_pose[:, start_frame:end_frame, ...]

                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")

                with torch.no_grad():
                    ref_img_foregrounds = []
                    ref_concat_image_noises = []
                    ref_add_image_noises = []
                    ref_img_background_masks = []
                    ref_img_clips = []
                    ref_img_converts = []
                    ref_img_conditions = []
                    image_np = rearrange(pixel_values_ref_img, "b c h w -> b h w c")
                    image_np = image_np.cpu().numpy().astype(np.uint8)
                    for i, ref_img in enumerate(image_np):
                        ref_img = Image.fromarray(ref_img)
                        dwpose_model_result_dict = dwpose_model(ref_img)
                        # NOTE: foreground used for remove background
                        ref_img_foreground = dwpose_model_result_dict['foreground']
                        ref_img_foregrounds.append(ref_img_foreground)
                        ref_img_convert = dwpose_model_result_dict[ref_image_type]
                        ref_img_converts.append(ref_img_convert)
                        ref_img_condition = dwpose_model_result_dict[control_aux_type]
                        ref_img_conditions.append(ref_img_condition)

                        # NOTE: background_mask used for concat to noise
                        if concat_noise_image_type != "":
                            ref_concat_image_noise = dwpose_model_result_dict[concat_noise_image_type]
                            ref_concat_image_noises.append(ref_concat_image_noise)
                            ref_img_background_mask = dwpose_model_result_dict['background_mask']
                            ref_img_background_masks.append(ref_img_background_mask)                                

                        if add_noise_image_type != "":
                            print(f'WARNING it is use add_noise_image_type is {add_noise_image_type}')
                            ref_add_image_noise = dwpose_model_result_dict[add_noise_image_type]
                            ref_add_image_noises.append(ref_add_image_noise)

                        if clip_image_type != "":
                            ref_img_clip = dwpose_model_result_dict[clip_image_type]
                            ref_img_clips.append(ref_img_clip)

                ref_img_conditions = torch.Tensor(np.array(ref_img_conditions)).to(local_rank, dtype=weight_type)
                ref_img_conditions = rearrange(ref_img_conditions, 'b h w c -> b c h w')
                ref_img_conditions = ref_img_conditions / 255.

                # work for imageencoder   
                with torch.no_grad():
                    image_prompt_embeddings = None
                    if clip_image_type != "":
                        clip_images = []
                        for i, ref_image_clip in enumerate(ref_img_clips):
                            ref_image_clip = Image.fromarray(ref_image_clip)
                                                    
                            clip_image = image_processor(
                                images=ref_image_clip, return_tensors="pt").pixel_values
                            clip_images.append(clip_image)

                        clip_images = torch.cat(clip_images)

                        image_emb = image_encoder(clip_images.to(
                            local_rank, dtype=weight_type), output_hidden_states=True).last_hidden_state
                        image_emb = image_encoder.vision_model.post_layernorm(image_emb)
                        image_emb = image_encoder.visual_projection(image_emb)
                        image_prompt_embeddings = image_emb

                # NOTE: convert pixel_values(origin video) to latent by vae
                pixel_values = pixel_values / 127.5 - 1
                # print('train pixel_values unique is', pixel_values.unique())
                with torch.no_grad():
                    if not image_finetune:
                        
                        latents = model.module.vae.encode(pixel_values).latent_dist
                        latents = latents.sample()
                        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                    else:
                        latents = model.module.vae.encode(pixel_values).latent_dist
                        latents = latents.sample()

                    latents = latents * 0.18215

                # NOTE: concat_noise_image_type: turn background to latent by concat
                with torch.no_grad():
                    if concat_noise_image_type != "":
                        # Image.fromarray(ref_concat_image_noises[0].astype('uint8')).save('ref_concat_image_noise.png')
                        one_img_have_more = False
                        ref_concat_image_noises = torch.Tensor(np.array(ref_concat_image_noises)).to(local_rank, dtype=weight_type)
                        if len(ref_concat_image_noises.shape) == 5:
                            ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b f h w c -> (b f) h w c')
                            one_img_have_more = True
                        ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b h w c -> b c h w')
                        ref_concat_image_noises = ref_concat_image_noises / 127.5 - 1
                        # print('ref_img_backgrounds unique is', ref_img_backgrounds.unique())
                        ref_concat_image_noises_latents = model.module.vae.encode(ref_concat_image_noises).latent_dist
                        ref_concat_image_noises_latents = ref_concat_image_noises_latents.sample().unsqueeze(2)
                        ref_concat_image_noises_latents = ref_concat_image_noises_latents * 0.18215
                        # b c 1 h w b c f h w

                        if one_img_have_more == True:
                            B, C, _, H, W = ref_concat_image_noises_latents.shape
                            ref_concat_image_noises_latents = ref_concat_image_noises_latents.reshape(B//2, C*2, _, H, W)

                        ref_img_back_mask_latents = torch.tensor(np.array(ref_img_background_masks).transpose(0, 3, 1, 2)).to(local_rank, dtype=weight_type)
                        H, W = ref_concat_image_noises_latents.shape[3:]
                        ref_img_back_mask_latents = F.interpolate(ref_img_back_mask_latents, size=(H, W), mode='nearest').unsqueeze(2)
                        # print('train ref_img_back_mask_latents unique is', ref_img_back_mask_latents.unique())
                        # ref_img_backgrounds_latents = ref_img_backgrounds_latents.repeat(1, 1, latents.shape[2], 1, 1)
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents) + 0.1 * torch.randn(latents.shape[0], latents.shape[1], 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
                bsz = latents.shape[0]

                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if concat_noise_image_type != "":
                    ref_concat_image_noises_latents = ref_concat_image_noises_latents.repeat(1, 1, noisy_latents.shape[2], 1, 1)
                    ref_img_back_mask_latents = ref_img_back_mask_latents.repeat(1, 1, noisy_latents.shape[2], 1, 1)
                    noisy_latents = torch.cat([noisy_latents, ref_concat_image_noises_latents, ref_img_back_mask_latents], dim=1)
                
                if ref_image_type != "origin":
                    ref_img_converts = torch.Tensor(np.array(ref_img_converts)).to(local_rank, dtype=weight_type)
                    ref_img_converts = rearrange(ref_img_converts, 'b h w c -> b c h w')
                    pixel_values_ref_img = ref_img_converts

                # show_ref_img = pixel_values_ref_img.cpu().numpy().astype('uint8')[0]
                # Image.fromarray(show_ref_img.transpose(1, 2, 0)).save('show_ref_img.png')
                pixel_values_ref_img = pixel_values_ref_img / 127.5 - 1.
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                # import pdb;pdb.set_trace()
                # for x in [noisy_latents, encoder_hidden_states, np.array(ref_pil_images[0]), control_conditions]:
                #     print(x.shape)
                """
                noisy_latents: torch.Size([1, 4, 8, 64, 64])
                encoder_hidden_states: torch.Size([1, 257, 1280])
                np.array(ref_pil_images[0]): (512, 512, 3)
                control_conditions: (8, 512, 512, 3)
                """

                """
                TODO：pose改成b f h w c格式 待会适配下， ref_pil_images改成了b h w c格式
                """
                
                with accelerator.autocast():
                    model_pred = model(init_latents=noisy_latents,
                                    image_prompts=image_prompt_embeddings,
                                    timestep=timesteps,
                                    guidance_scale=1.0,
                                    source_image=pixel_values_ref_img, 
                                    motion_sequence=pixel_values_pose,
                                    random_seed=seed,
                                    froce_text_embedding_zero=froce_text_embedding_zero,
                                    ref_img_conditions=None
                                    )
                    loss = train_loss_func(
                        model_output = model_pred.float(),
                        x_t = noisy_latents.float(),
                        x_start = latents.float(),
                        target = target.float(),
                        timestep = timesteps.to(device=local_rank),
                    )
                    avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                    train_loss += avg_loss.item() / gradient_accumulation_steps

                    # use accelerator
                accelerator.backward(loss)
                model.module.clear_reference_control()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                    seed = global_step
                    set_seed((seed + resume_step_offset) % recycle_seed)
                    warp_params = generate_random_params(resolution[0], resolution[1])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                    
                progress_bar.update(1)
                global_step += 1

                ### <<<< Training <<<< ###
                is_main_process = accelerator.is_main_process

                # Save checkpoint
                if is_main_process and (global_step % checkpointing_steps == 0 or global_step in validation_steps_tuple or global_step % validation_steps == 0):
                    if global_step >= checkpointing_steps and global_step % checkpointing_steps == 0:
                        save_path = os.path.join(output_dir, f"checkpoints")
                        state_dict = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "state_dict": model.state_dict(),
                        }
                        if step == len(train_dataloader) - 1:
                            torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch + 1}.ckpt"))
                        else:
                            def execute_aws_and_rm_commands(model_path, save_dir):
                                subprocess.run(['aws',
                                                f'--endpoint-url={train_dataset.endpoint_url}',
                                                's3',
                                                'cp',
                                                model_path,
                                                f"s3://radar/yangshurong/{save_dir}/"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                                
                                subprocess.run(['rm', '-rf', model_path], check=True)
                            model_save_path = os.path.join(save_path, f"checkpoint-steps{global_step}.ckpt")
                            torch.save(state_dict, model_save_path)
                            thread = threading.Thread(target=execute_aws_and_rm_commands, args=(model_save_path, folder_name))
                            # 启动子线程
                            thread.start()
                        logging.info(f"Saved state to {save_path} (global_step: {global_step})")

                    eval_model(validation_data, 
                                model, 
                                local_rank, 
                                weight_type, 
                                context, 
                                output_dir, 
                                global_step,
                                accelerator,
                                valid_seed,
                                dwpose_model,
                                arcface_encoder,
                                image_processor,
                                image_encoder,
                                face_detector,
                                control_aux_type,
                                clip_image_type,
                                ref_image_type,
                                concat_noise_image_type,
                                do_classifier_free_guidance,
                                add_noise_image_type,
                                )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()


from animatediff.utils.videoreader import VideoReader


def eval_model(validation_data,
                 model, 
                 local_rank, 
                 weight_type, 
                 context,
                 output_dir,
                 global_step,
                 accelerator,
                 valid_seed,
                 dwpose_model,
                 arcface_encoder=None,
                 image_processor=None,
                 image_encoder=None,
                 face_detector=None,
                 control_aux_type='',
                 clip_image_type="",
                 ref_image_type="",
                 concat_noise_image_type="",
                 do_classifier_free_guidance=True,
                 add_noise_image_type="",
                 ):
    sample_size = validation_data['sample_size']
    guidance_scale = validation_data['guidance_scale']

    # input test videos (either source video/ conditions)

    test_videos = validation_data['video_path']
    source_images = validation_data['source_image']

    # read size, step from yaml file
    sizes = [sample_size] * len(test_videos)
    steps = [validation_data['S']] * len(test_videos)

    for idx, (source_image, test_video, size, step) in tqdm(
        enumerate(zip(source_images, test_videos, sizes, steps)),
        total=len(test_videos)
    ):

        # Load control and source image
        if test_video.endswith('.mp4') or test_video.endswith('.gif'):
            print('Control Condition', test_video)
            control = VideoReader(test_video).read()
            video_length = control.shape[0]
            print('control', control.shape)
        else:
            print("!!!WARNING: SKIP this case since it is not a video")
        
        print('Reference Image', source_image)
        if source_image.endswith(".mp4") or source_image.endswith(".gif"):
            source_image = VideoReader(source_image).read()[0]
        else:
            source_image = Image.open(source_image)
            if np.array(source_image).shape[2] == 4:
                source_image = source_image.convert("RGB")

        source_image = torch.tensor(np.array(source_image)).unsqueeze(0)
        source_image = rearrange(source_image, "b h w c -> b c h w") # b c h w
        control = torch.tensor(control)
        control = rearrange(control, "b h w c -> b c h w") # b c h w

        control = crop_and_resize_tensor_face(control, size, crop_face_center=True, face_detector=face_detector)
        source_image = crop_and_resize_tensor_face(source_image, size, crop_face_center=True, face_detector=face_detector)
        
            # print("source image shape is", np.array(source_image).shape, np.unique(np.array(source_image)))

        control_condition, control = get_condition(control, source_image, dwpose_model, control_aux_type, switch_control_to_source = True)

        pixel_values_pose = torch.Tensor(np.array(control_condition))
        pixel_values_pose = rearrange(
            pixel_values_pose, "(b f) h w c -> b f h w c", b=1)
        pixel_values_pose = pixel_values_pose.to(local_rank, dtype=weight_type)
        pixel_values_pose = pixel_values_pose / 255.
        # img_for_face_det = torch.tensor(np.array(source_image)).to(local_rank, torch.uint8).unsqueeze(0).permute(0, 3, 1, 2)
        # if image_encoder is not None:
            # with torch.inference_mode():
            #     # img_for_face_det is B C H W
            #     faces = face_detector(img_for_face_det)
            #     if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
            #         face_rect = None
            #     else:
            #         face_rect = faces['rects'][0].cpu().numpy()

            # face_image_pil = crop_and_resize(source_image, size, crop_rect=face_rect, is_arcface=True)
            # face_image = np.array(face_image_pil)
            # face_image = ((torch.Tensor(face_image).unsqueeze(
            #             0).to(local_rank, dtype=weight_type) / 255.0) - 0.5) * 2

        # if ref_image_type != "origin" or concat_noise_image_type != "" or add_noise_image_type != "" or clip_image_type != "":
        with torch.inference_mode():
            source_image_pil = Image.fromarray(source_image[0].permute(1, 2, 0).numpy().astype("uint8"))
            dwpose_model_result_dict = dwpose_model(source_image_pil)
            # Image.fromarray(ref_image_control).save('ref_image_control.png')
            ref_img_foreground = dwpose_model_result_dict['foreground']
            ref_img_convert = dwpose_model_result_dict[ref_image_type]
            ref_img_condition = dwpose_model_result_dict[control_aux_type]
            if concat_noise_image_type != "":
                ref_concat_image_noise = dwpose_model_result_dict[concat_noise_image_type]
                ref_img_background_mask = dwpose_model_result_dict['background_mask']
            if add_noise_image_type != "":
                ref_add_image_noise = dwpose_model_result_dict[add_noise_image_type]
            if clip_image_type != "":
                ref_img_clip = dwpose_model_result_dict[clip_image_type]  
                ref_img_clip = Image.fromarray(ref_img_clip)

        ref_img_condition = torch.Tensor(np.array(ref_img_condition)).unsqueeze(0).to(local_rank, dtype=weight_type)
        ref_img_condition = rearrange(ref_img_condition, 'b h w c -> b c h w')
        ref_img_condition = ref_img_condition / 255.

        # Image.fromarray(ref_image_background.astype('uint8')).save('backtest.png')

        source_image = np.array(source_image_pil)
        if ref_image_type != "origin":
            source_image = ref_img_convert
        source_image = ((torch.Tensor(source_image).unsqueeze(
            0).to(local_rank, dtype=weight_type) / 255.0) - 0.5) * 2

        B, H, W, C = source_image.shape
        
        # concat noise with background latents
        ref_concat_image_noises_latents = None
        if concat_noise_image_type != "":
            ref_concat_image_noises = torch.Tensor(np.array(ref_concat_image_noise)).unsqueeze(0).to(local_rank, dtype=weight_type)
            one_img_have_more = False
            if len(ref_concat_image_noises.shape) == 5:
                ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b f h w c -> (b f) h w c')
                one_img_have_more = True
            ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b h w c -> b c h w')
            ref_concat_image_noises = ref_concat_image_noises / 127.5 - 1
            # print('ref_img_backgrounds unique is', ref_img_backgrounds.unique())
            ref_concat_image_noises_latents = model.module.vae.encode(ref_concat_image_noises).latent_dist
            ref_concat_image_noises_latents = ref_concat_image_noises_latents.sample().unsqueeze(2)
            ref_concat_image_noises_latents = ref_concat_image_noises_latents * 0.18215

            if one_img_have_more == True:
                B, C, _, H, W = ref_concat_image_noises_latents.shape
                ref_concat_image_noises_latents = ref_concat_image_noises_latents.reshape(B//2, C*2, _, H, W)

            
            ref_img_back_mask_latents = torch.tensor(np.array(ref_img_background_mask)[None, ...].transpose(0, 3, 1, 2)).to(local_rank, dtype=weight_type)
            H, W = ref_concat_image_noises_latents.shape[3:]
            ref_img_back_mask_latents = F.interpolate(ref_img_back_mask_latents, size=(H, W), mode='nearest').unsqueeze(2)
            # print('infer ref_img_back_mask_latents unique is', ref_image_back_mask_latents.unique())
            ref_concat_image_noises_latents = torch.cat([
                ref_concat_image_noises_latents, ref_img_back_mask_latents
            ], dim=1).repeat(1, 1, video_length, 1, 1)

            if guidance_scale > 1.0 and do_classifier_free_guidance:
                ref_concat_image_noises_latents = torch.cat([ref_concat_image_noises_latents,
                 ref_concat_image_noises_latents])

            # ref_img_backgrounds_latents = ref_img_backgrounds_latents.repeat(1, 1, latents.shape[2], 1, 1)


        ######################### image encoder#########################
        image_prompt_embeddings = None
        if clip_image_type != "":
            with torch.inference_mode():
                clip_image = image_processor(
                    images=ref_img_clip, return_tensors="pt").pixel_values
                image_emb = image_encoder(clip_image.to(
                    local_rank, dtype=weight_type), output_hidden_states=True).last_hidden_state
                image_emb = image_encoder.vision_model.post_layernorm(image_emb)
                image_emb = image_encoder.visual_projection(image_emb)# image_emb = image_encoder.vision_model.post_layernorm(image_emb)

                # face_image = rearrange(face_image, 'b h w c -> b c h w')
                # face_image_emb = arcface_encoder(face_image) # (B, 512)
                # face_image_emb = face_image_emb.unsqueeze(1).repeat(1, image_emb.shape[1], 1)

                # image_emb = torch.cat([image_emb, face_image_emb], dim=2)
                # negative image embeddings
                # image_np_neg = np.zeros_like(source_image_pil)
                # ref_pil_image_neg = Image.fromarray(
                #     image_np_neg.astype(np.uint8))
                # ref_pil_image_pad = pad_image(ref_pil_image_neg)
                # clip_image_neg = image_processor(
                #     images=ref_pil_image_pad, return_tensors="pt").pixel_values
                # image_emb_neg = image_encoder(clip_image_neg.to(
                #     device, dtype=weight_type), output_hidden_states=True).hidden_states[-2]

                # image_prompt_embeddings = torch.cat([image_emb_neg, image_emb])
                image_prompt_embeddings = image_emb
                if guidance_scale > 1.0 and do_classifier_free_guidance:
                    # guidance free
                    image_prompt_embeddings = torch.cat([image_emb, image_emb])


        with torch.no_grad():
            with accelerator.autocast():
                source_image = rearrange(source_image, 'b h w c -> b c h w')
                samples = model.module.infer(
                    source_image=source_image,
                    image_prompts=image_prompt_embeddings,
                    motion_sequence=pixel_values_pose,
                    step=validation_data['num_inference_steps'],
                    guidance_scale=guidance_scale,
                    context=context,
                    size=sample_size,
                    random_seed=valid_seed,
                    froce_text_embedding_zero=False,
                    ref_concat_image_noises_latents=ref_concat_image_noises_latents,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    add_noise_image_type=add_noise_image_type,
                    ref_img_condition=None
                )
            if control_aux_type == "densepose_dwpose_concat":
                control = torch.tensor(control).unsqueeze(0)
                control = rearrange(control, 'b t h w c -> b c t h w') / 255.
                samples[1] = control
            
            # shape need to be 1 c t h w
            source_image = np.array(source_image_pil) # h w c
            source_image = torch.Tensor(source_image).unsqueeze(
                        0) / 255.
            source_image = source_image.repeat(video_length, 1, 1, 1)
            samples[0] = rearrange(source_image, "t h w c -> 1 c t h w") 
            control = torch.tensor(control).unsqueeze(0)
            control = rearrange(control, 'b t h w c -> b c t h w') / 255.
            samples.insert(0, control)
            samples = torch.cat(samples)

            
            # print('eval save samples shape is', samples.shape)
            
        os.makedirs(f"{output_dir}/samples/sample_{global_step}", exist_ok=True)
        video_name = os.path.basename(test_video)[:-4]
        source_name = os.path.basename(validation_data['source_image'][idx]).split(".")[0]

        save_path = f"{output_dir}/samples/sample_{global_step}/{source_name}_{video_name}.gif"
        save_videos_grid(samples, save_path, save_every_image=False)
        accelerator.print(f"Saved samples to {save_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, origin_config=config, **config)
