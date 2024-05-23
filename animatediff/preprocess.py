import argparse
import datetime
import inspect
import os
import random
import numpy as np

from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm

from animatediff.utils.util import save_videos_grid, get_condition
from animatediff.utils.util import crop_and_resize_tensor_face
from animatediff.utils.dist_tools import distributed_init
from accelerate.utils import set_seed

from animatediff.utils.videoreader import VideoReader

from einops import rearrange

from pathlib import Path
from megfile import smart_open
import io
from animatediff.utils.util import save_videos_grid, pad_image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from animatediff.utils.util import save_videos_grid, pad_image, generate_random_params, apply_transforms
import facer


arcface_encoder = None
image_processor = None
image_encoder = None
face_detector = None


def find_lower_bound(d, x):
    lower_keys = [k for k in d.keys() if k <= x]
    assert lower_keys is not None, f"Can not find any one for step {x}"
    closest_key = max(lower_keys)
    return d[closest_key]


def condition_extraction(pixel_values_ref_img, dwpose_model, condition_config):
    # ref_image_type, concat_noise_image_type, add_noise_image_type, clip_image_type, control_aux_type
    with torch.no_grad():
        results = {}
        ref_img_background_masks = []
        image_np = rearrange(pixel_values_ref_img, "b c h w -> b h w c")
        image_np = image_np.cpu().numpy().astype(np.uint8)
        for i, ref_img in enumerate(image_np):
            ref_img = Image.fromarray(ref_img)
            dwpose_model_result_dict = dwpose_model(ref_img)
            for name, types in condition_config.items():
                if types == '':
                    continue
                cur_condition = []
                for ret_type in types.split(','):
                    cur_condition.append(dwpose_model_result_dict[ret_type])
                if name not in results:
                    results[name] = []
                results[name].append(cur_condition)
        return results


def prepare_input(accelerator, clip_path, model, video_length,
        pixel_values, pixel_values_ref_img, pixel_values_pose, seed, conditions,
        image_finetune=False, weight_type=torch.float16):
        inputs = {
            'random_seed': seed,
            'ref_img_conditions': None,
            'timestep': None,
            'image_prompts': None,
            'guidance_scale': 1.0,
            'init_latents': None,
            'source_image': None, 
            'motion_sequence': pixel_values_pose,
            'froce_text_embedding_zero': False,
        }
        local_rank = accelerator.device
        # NOTE: convert pixel_values(origin video) to latent by vae
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

        if 'control_aux' in conditions:
            ref_img_conditions = conditions.get('control_aux', None)
            ref_img_conditions = torch.Tensor(np.array(ref_img_conditions)).to(local_rank, dtype=weight_type)
            ref_img_conditions = rearrange(ref_img_conditions, 'b n h w c -> b (n c) h w')
            ref_img_conditions = ref_img_conditions / 255.
            inputs['ref_img_conditions'] = ref_img_conditions

        bsz = latents.shape[0]
        timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        inputs['timestep'] = timesteps

        noise = torch.randn_like(latents) + 0.1 * torch.randn(latents.shape[0], latents.shape[1], 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)
        inputs['init_latents'] = noisy_latents

        if 'clip_image' in conditions:
            global  image_processor, image_encoder, face_detector
            if image_encoder is None:
                image_processor = CLIPImageProcessor.from_pretrained(clip_path, subfolder="feature_extractor")
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_path, subfolder="image_encoder")

                image_encoder.to(local_rank, weight_type)
                image_encoder.requires_grad_(False)
            if face_detector is None:
                face_detector = facer.face_detector('retinaface/mobilenet', device=local_rank)
                face_detector.requires_grad_(False)

            clip_images = []
            ref_img_clips = conditions.get('clip_image', None)
            for i, ref_image_clips in enumerate(ref_img_clips):
                for ref_image_clip in ref_image_clips:
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
            inputs['image_prompts'] = image_emb

        # NOTE: concat_noise_image_type: turn background to latent by concat
        with torch.no_grad():
            if 'concat_noise' in conditions:
                ref_concat_image_noises = conditions.get('concat_noise', None)
                one_img_have_more = False
                ref_concat_image_noises = torch.Tensor(np.array(ref_concat_image_noises)).to(local_rank, dtype=weight_type)
                if len(ref_concat_image_noises.shape) == 5:
                    ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b f h w c -> (b f) h w c')
                    one_img_have_more = True
                ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b h w c -> b c h w')
                ref_concat_image_noises = ref_concat_image_noises / 127.5 - 1
                ref_concat_image_noises_latents = model.module.vae.encode(ref_concat_image_noises).latent_dist.sample().unsqueeze(2) * 0.18215
                # b c 1 h w b c f h w

                if one_img_have_more == True:
                    B, C, _, H, W = ref_concat_image_noises_latents.shape
                    ref_concat_image_noises_latents = ref_concat_image_noises_latents.reshape(B//2, C*2, _, H, W)

                ref_img_back_mask_latents = torch.tensor(np.array(ref_img_background_masks).transpose(0, 3, 1, 2)).to(local_rank, dtype=weight_type)
                H, W = ref_concat_image_noises_latents.shape[3:]
                ref_img_back_mask_latents = F.interpolate(ref_img_back_mask_latents, size=(H, W), mode='nearest').unsqueeze(2)
                ref_concat_image_noises_latents = ref_concat_image_noises_latents.repeat(1, 1, noisy_latents.shape[2], 1, 1)
                ref_img_back_mask_latents = ref_img_back_mask_latents.repeat(1, 1, noisy_latents.shape[2], 1, 1)
                noisy_latents = torch.cat([noisy_latents, ref_concat_image_noises_latents, ref_img_back_mask_latents], dim=1)
            if 'add_noise' in conditions:
                ref_concat_image_noises = conditions.get('concat_noise', None)
                one_img_have_more = False
                ref_concat_image_noises = torch.Tensor(np.array(ref_concat_image_noises)).to(local_rank, dtype=weight_type)
                if len(ref_concat_image_noises.shape) == 5:
                    ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b f h w c -> (b f) h w c')
                    one_img_have_more = True
                ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b h w c -> b c h w')
                ref_concat_image_noises = ref_concat_image_noises / 127.5 - 1
                ref_concat_image_noises_latents = model.module.vae.encode(ref_concat_image_noises).latent_dist.sample().unsqueeze(2) * 0.18215
                # b c 1 h w b c f h w

                if one_img_have_more == True:
                    B, C, _, H, W = ref_concat_image_noises_latents.shape
                    ref_concat_image_noises_latents = ref_concat_image_noises_latents.reshape(B//2, C*2, _, H, W)

                ref_img_back_mask_latents = torch.tensor(np.array(ref_img_background_masks).transpose(0, 3, 1, 2)).to(local_rank, dtype=weight_type)
                H, W = ref_concat_image_noises_latents.shape[3:]
                ref_img_back_mask_latents = F.interpolate(ref_img_back_mask_latents, size=(H, W), mode='nearest').unsqueeze(2)
                ref_concat_image_noises_latents = ref_concat_image_noises_latents.repeat(1, 1, noisy_latents.shape[2], 1, 1)
                ref_img_back_mask_latents = ref_img_back_mask_latents.repeat(1, 1, noisy_latents.shape[2], 1, 1)
                noisy_latents = noisy_latents + ref_concat_image_noises_latents + ref_img_back_mask_latents
        inputs['init_latents'] = noisy_latents
        
        if 'ref_image' in conditions:
            ref_img_converts = conditions.get('ref_image', None)
            ref_img_converts = torch.Tensor(np.array(ref_img_converts)).to(local_rank, dtype=weight_type)
            ref_img_converts = rearrange(ref_img_converts, 'b h w c -> b c h w')
            pixel_values_ref_img = ref_img_converts
        pixel_values_ref_img = pixel_values_ref_img / 127.5 - 1.
        inputs['source_image'] = pixel_values_ref_img
        return inputs, noise, noisy_latents, latents, timesteps

def preprocess_with_video_length(accelerator, batch, special_ref_index, train_batch_size, global_step, step, train_frame_number_step, train_cross_rate_step, warp_condition_use, train_warp_rate_step,  output_dir='./', seed=0, resume_step_offset=0, recycle_seed=10000000, resolution=(512, 512), weight_type=torch.float16):
    local_rank = accelerator.device
    video_length = find_lower_bound(train_frame_number_step, step + resume_step_offset)
    use_2d_cross_rate = find_lower_bound(train_cross_rate_step, step + resume_step_offset)
    warp_rate = find_lower_bound(train_warp_rate_step, step + resume_step_offset)

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
        warp_params = generate_random_params(resolution[0], resolution[1])
        pixel_values_pose = rearrange(pixel_values_pose, "b f h w c -> (b f) c h w")
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
        num_processes = torch.cuda.device_count()
        rng = np.random.default_rng(seed=(((seed + resume_step_offset) % recycle_seed) * num_processes + accelerator.process_index))
        frames_num = pixel_values.shape[1]
        segment_num = frames_num // video_length 
        segment_index = rng.integers(1, segment_num)
        start_frame = segment_index * video_length 
        end_frame = start_frame + video_length 
        pixel_values = pixel_values[:, start_frame:end_frame, ...]
        pixel_values_pose = pixel_values_pose[:, start_frame:end_frame, ...]

    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
    pixel_values = pixel_values / 127.5 - 1
    return video_length, pixel_values, pixel_values_pose, pixel_values_ref_img


def prepare_input_for_eval(accelerator, validation_data, dwpose_model, clip_path, condition_config, valid_seed, guidance_scale, context, do_classifier_free_guidance, weight_type=torch.float16, crop_face_center=True, switch_control_to_source=True):
    local_rank = accelerator.device
    input_sequences = []
    with torch.inference_mode():
        sample_size = validation_data['sample_size']
        guidance_scale = validation_data['guidance_scale']

        # input test videos (either source video/ conditions)
        test_videos = validation_data['video_path']
        source_images = validation_data['source_image']

        # read size, step from yaml file
        sizes = [sample_size] * len(test_videos)
        vis_grids= []

        for idx, (source_image, test_video, size) in tqdm(
            enumerate(zip(source_images, test_videos, sizes)),
            total=len(test_videos)
        ):
            inputs = {
                'random_seed': valid_seed,
                'guidance_scale': guidance_scale,
                'source_image': None, 
                'motion_sequence': None,
                'froce_text_embedding_zero': False,
                'step': validation_data['num_inference_steps'],
                'context': context,
                'size': validation_data['sample_size'],
                'ref_concat_image_noises_latents': None, # ref_concat_image_noises_latents,
                'do_classifier_free_guidance': do_classifier_free_guidance, #do_classifier_free_guidance,
                'add_noise_image_type': condition_config['add_noise'], #add_noise_image_type,
                'ref_img_condition': None,
                'image_prompts': None,
            }
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

            global face_detector
            if face_detector is None:
                face_detector = facer.face_detector('retinaface/mobilenet', device=local_rank)
                face_detector.requires_grad_(False)
            control = crop_and_resize_tensor_face(control, size, crop_face_center=crop_face_center, face_detector=face_detector)
            source_image = crop_and_resize_tensor_face(source_image, size, crop_face_center=crop_face_center, face_detector=face_detector)
            
            control_condition, control = get_condition(control, source_image, dwpose_model, condition_config['control_aux'], switch_control_to_source = switch_control_to_source)
            pixel_values_pose = torch.Tensor(np.array(control_condition))
            pixel_values_pose = rearrange(
                # pixel_values_pose, "(b f) h w c -> b f c h w", b=1)
                pixel_values_pose, "(b f) h w c -> b f h w c", b=1)
            pixel_values_pose = pixel_values_pose.to(local_rank, dtype=weight_type)
            pixel_values_pose = pixel_values_pose / 255.
            inputs['motion_sequence'] = pixel_values_pose

            conditions =  condition_extraction(source_image[0].to(torch.uint8)[None, ...], dwpose_model, condition_config)
            if 'ref_image' in conditions:
                ref_img_converts = conditions.get('ref_image', None)
                ref_img_converts = torch.Tensor(np.array(ref_img_converts)).to(local_rank, dtype=weight_type)
                ref_img_converts = rearrange(ref_img_converts, 'b h w c -> b c h w')
                source_image = ref_img_converts
            source_image = (source_image / 127.5 - 1.).to(local_rank, dtype=weight_type)
            inputs['source_image'] = source_image

        B, H, W, C = source_image.shape
        # concat noise with background latents
        ref_concat_image_noises_latents = None
        if 'concat_noise' in conditions:
            ref_concat_image_noises = conditions.get('concat_noise', None)
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
            inputs['ref_concat_image_noises_latents'] = ref_concat_image_noises_latents

            if guidance_scale > 1.0 and do_classifier_free_guidance:
                ref_concat_image_noises_latents = torch.cat([ref_concat_image_noises_latents,
                 ref_concat_image_noises_latents])

        ######################### image encoder#########################
        image_prompt_embeddings = None
        if 'clip_image' in conditions:
            global  image_processor, image_encoder
            if image_encoder is None:
                image_processor = CLIPImageProcessor.from_pretrained(clip_path, subfolder="feature_extractor")
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_path, subfolder="image_encoder")
                image_encoder.to(local_rank, weight_type)
                image_encoder.requires_grad_(False)

            ref_img_clips = conditions.get('clip_image', None)
            clip_images = []
            with torch.inference_mode():
                for i, ref_image_clips in enumerate(ref_img_clips):
                    for ref_image_clip in ref_image_clips:
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
                if guidance_scale > 1.0 and do_classifier_free_guidance:
                    image_prompt_embeddings = torch.cat([image_emb, image_emb])
            inputs['image_prompts'] = image_prompt_embeddings
        input_sequences.append(inputs)
        
        vis_source = source_image + 1 / 2
        vis_source = source_image.repeat(video_length, 1, 1, 1)
        vis_source = rearrange(vis_source, "t c h w  -> 1 c t h w") 

        control = torch.tensor(control).unsqueeze(0).to(local_rank)
        control = rearrange(control, 'b t h w c -> b c t h w') / 255.

        vis = torch.cat([vis_source,  control]).cpu()
        vis_grids.append(vis)
    return input_sequences, vis_grids