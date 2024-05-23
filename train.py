import os
import math
import random
import logging
import inspect
import importlib
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

from animatediff.utils.util import save_videos_grid, pad_image, generate_random_params, apply_transforms
from animatediff.utils.util import get_condition, crop_and_resize_tensor_face, crop_and_resize_tensor
from animatediff import losses
from animatediff import preprocess
from animatediff.utils.videoreader import VideoReader
from accelerate import Accelerator
from einops import repeat
from accelerate.utils import set_seed
import webdataset as wds
from libs.controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor

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
        train_loss: str = "mse",
        warp_condition_use: bool = False,
        do_classifier_free_guidance: bool = True,
        special_ref_index: int = -1,
        remove_referencenet: bool = False,
        appearance_controlnet_motion_checkpoint_ignore: Dict = {},
        resume_step_offset: int = 0,
        appearance_time_step = None,
        recycle_seed: int = 100000000,
        condition_config: dict = {}
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

    if accelerator.state.deepspeed_plugin is not None and \
            accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto":
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = train_batch_size

    local_rank = accelerator.device

    # load dwpose detector, see controlnet_aux: https://github.com/patrickvonplaten/controlnet_aux
    # specify configs, ckpts and device, or it will be downloaded automatically and use cpu by default
    dwpose_model = DenseDWposePredictor(local_rank, resolution=train_data['resolution'])
    MagicAnimate = getattr(importlib.import_module(f'animatediff.magic_animate.{model_type}.animate'), 'MagicAnimate')
    train_loss_func = getattr(losses, train_loss)
    if accelerator.is_main_process:
        print(f"using {train_loss}_loss")

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

    for epoch in range(first_epoch, num_train_epochs):
        # TODO: check webdataset在多卡的随机性问题
        # train_dataloader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # 根据配置参数选择reference frame并提取关键点
            # 根据resume进度重新判断video length
            # 获取所有帧、pose帧、reference帧
            video_length, pixel_values, pixel_values_pose, pixel_values_ref_img = preprocess.preprocess_with_video_length(
                accelerator, batch, special_ref_index, train_batch_size,
                global_step, step, train_frame_number_step, train_cross_rate_step, warp_condition_use,  train_warp_rate_step,
                output_dir=output_dir, seed=seed, resume_step_offset=resume_step_offset, recycle_seed=recycle_seed, resolution=resolution, weight_type=weight_type
                )
            conditions = preprocess.condition_extraction(pixel_values_ref_img, dwpose_model, condition_config)
            with accelerator.accumulate(model):
                inputs, noise, noisy_latents, latents, timesteps = preprocess.prepare_input(accelerator, pretrained_model_path, model, video_length, pixel_values, pixel_values_ref_img, pixel_values_pose, seed, conditions,image_finetune=image_finetune, weight_type=weight_type)
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
                    model_pred = model(**inputs)
                    loss = train_loss_func(
                        model_output = model_pred.float(),
                        x_t = noisy_latents.float(),
                        x_start = latents.float(),
                        target = target.float(),
                        timestep = timesteps.to(device=local_rank),
                        config=config
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
                                condition_config,
                                pretrained_model_path,
                                do_classifier_free_guidance=do_classifier_free_guidance
                    )
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()


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
                 condition_config,
                 clip_path,
                 do_classifier_free_guidance=False,
                 crop_face_center=True, 
                 switch_control_to_source=True,
                 ):
        test_videos = validation_data['video_path']
        source_images = validation_data['source_image']
        with torch.no_grad():
            with accelerator.autocast():
                inputs, vis_grids = preprocess.prepare_input_for_eval(accelerator,
                    validation_data, dwpose_model, clip_path, condition_config, 
                    valid_seed, validation_data['guidance_scale'], context, do_classifier_free_guidance, 
                    weight_type = weight_type)

                for idx, (ipt, vis_grid, test_video) in enumerate(zip(inputs, vis_grids, test_videos)):
                    samples  = model.infer(**ipt)
                    samples = torch.cat( [vis_grid] + samples)
                    
                    os.makedirs(f"{output_dir}/samples/sample_{global_step}", exist_ok=True)
                    video_name = os.path.basename(test_video)[:-4]
                    source_name = os.path.basename(validation_data['source_image'][idx]).split(".")[0]
                    save_path = f"{output_dir}/samples/sample_{global_step}/{source_name}_{video_name}.gif"
                    save_videos_grid(samples, save_path, save_every_image=False)

                    if accelerator:
                        accelerator.print(f"Saved samples to {save_path}")
                    else:
                        print(f"Saved samples to {save_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, origin_config=config, **config)
