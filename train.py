# Copyright 2024 Megvii inc.
#
# Copyright (2024) MegActor Authors.
#
# Megvii Inc. retain all intellectual property and proprietary rights in 
# and to this material, related documentation and any modifications thereto. 
# Any use, reproduction, disclosure or distribution of this material and related 
# documentation without an express license agreement from Megvii Inc. is strictly prohibited.

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
from animate.utils.util import save_videos_grid, pad_image, generate_random_params, apply_transforms
from animate.utils.util import crop_move_face
from animate.utils.util import crop_and_resize_tensor, get_condition_face
from animate.unet_magic_noiseAttenST_Ada.animate import MagicAnimate    
from accelerate import Accelerator
from einops import repeat
from accelerate.utils import set_seed
import webdataset as wds
from eval import eval
from face_dataset import VideosIterableDataset
import facer
from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor


def main(
        origin_config,
        name: str,
        launcher: str,

        output_dir: str,

        size: list,
        train_data: Dict,
        validation_data: Dict,
        context: Dict,

        pretrained_model_path: str = "",
        pretrained_appearance_encoder_path: str = "",
        pretrained_controlnet_path: str = "",
        pretrained_vae_path: str = "",
        motion_module: str = "",
        appearance_controlnet_motion_checkpoint_path: str = "",
        pretrained_unet_path: str = "",
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

        ip_ckpt=None,
        control_aux_type: str = 'dwpose',
        controlnet_type: str = '2d',
        controlnet_config: str = '',

        model_type: str = "unet",

        clip_image_type: str = '',
        concat_noise_image_type: str = '',
        do_classifier_free_guidance: bool = True,
        inference_config: str = "",
):

    assert model_type in ["unet", 
                        "unet_magic_noiseAttenST",
                        "unet_magic_noiseAttenST_Ada"]

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
    video_length = train_data["video_length"]
    resolution     = size
    # load dwpose detector, see controlnet_aux: https://github.com/patrickvonplaten/controlnet_aux
    # specify configs, ckpts and device, or it will be downloaded automatically and use cpu by default
    
    if accelerator.is_main_process:
        print("using mse_loss")

    model = MagicAnimate(config=config,
                         train_batch_size=train_batch_size,
                         device=local_rank,
                         unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs),
                         mixed_precision_training=True,
                         trainable_modules=trainable_modules,
                         is_main_process=accelerator.is_main_process,
                         weight_type=weight_type,)
    # Load noise_scheduler
    noise_scheduler = model.scheduler
    # ----- load image encoder ----- #
    """
    使用IP-adapter，主要包含image_encoder，clip_image_processor和image_proj_model
    image_proj_model在Resampler里定义
    """
    image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="image_encoder")
    image_encoder.to(local_rank, weight_type)
    image_encoder.requires_grad_(False)

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

    train_dataset = VideosIterableDataset(
        data_dirs = train_data['data_dirs'],
        batch_size=train_batch_size,
        video_length   = video_length,
        resolution     = size,
        frame_stride   = train_data['frame_stride'],
        dataset_length = 1000000,
        shuffle        = True,
        resampled      = True,
        return_origin  = True,
        warp_rate=train_data['warp_rate'],
        color_jit_rate=train_data['color_jit_rate'],
        use_swap_rate=train_data['use_swap_rate'],
    )

    train_dataloader = wds.WebLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        # this must be zeros since in mul GPU
        collate_fn = None,
    ).with_length(len(train_dataset))

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
    set_seed(seed)

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                noisy_latents, image_emb, timesteps, pixel_values_ref_img, pixel_values_pose, ref_img_conditions, target = model.preprocess_train(batch, image_processor, image_encoder)
                with accelerator.autocast():
                    model_pred = model(init_latents=noisy_latents,
                                    image_prompts=image_emb,
                                    timestep=timesteps,
                                    guidance_scale=1.0,
                                    source_image=pixel_values_ref_img, 
                                    motion_sequence=pixel_values_pose,
                                    random_seed=seed,
                                    ref_img_conditions=ref_img_conditions,
                                    )
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                model.module.clear_reference_control()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                    seed = global_step
                    set_seed(seed)
                    warp_params = generate_random_params(size[0], size[1])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                    
                progress_bar.update(1)
                global_step += 1

                is_main_process = accelerator.is_main_process
                if is_main_process and (global_step % checkpointing_steps == 0 or global_step in validation_steps_tuple or global_step % validation_steps == 0):
                    cur_save_path = f"{output_dir}/samples/sample_{global_step}"      
                    os.makedirs(cur_save_path, exist_ok=True)
                    for source, driver in tqdm(zip(validation_data['source_image'], validation_data['video_path'])):
                        eval(source, driver, 
                            config=None,
                            config_path=origin_config,
                            output_path=cur_save_path, 
                            random_seed=valid_seed,
                            guidance_scale=validation_data['guidance_scale'],
                            weight_type=torch.float16, 
                            num_steps=validation_data['num_inference_steps'],
                            device=local_rank, 
                            model=model,
                            image_processor=image_processor,
                            image_encoder=image_encoder,
                            clip_image_type="background",
                            concat_noise_image_type="origin",
                            do_classifier_free_guidance=True,
                            show_progressbar=False
                        )
                    save_path = os.path.join(output_dir, f"checkpoints")
                    state_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "state_dict": model.state_dict(),
                    }
                    model_save_path = os.path.join(save_path, f"checkpoint-steps{global_step}.ckpt")
                    torch.save(state_dict, model_save_path)

                    logging.info(f"Saved state to {save_path} (global_step: {global_step})")
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, origin_config=args.config, **config)