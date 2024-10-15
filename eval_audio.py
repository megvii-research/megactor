import argparse
import datetime
import inspect
import os
import random
from tqdm import tqdm
from pathlib import Path
import io
import numpy as np
import cv2
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict
import importlib
import pillow_avif

import torch
import torchvision
import torchaudio
import torch.nn.functional as F

from animate.utils.util import save_videos_grid, pad_image, crop_move_face, crop_and_resize_tensor_xpose, crop_and_resize_tensor, wide_crop_face, get_patch_div
from animate.utils.util import crop_and_resize_tensor_face, crop_area_eye_mouth, save_videos_grid_audio, get_patch, crop_and_resize_tensor_small_faces
from accelerate.utils import set_seed
from animate.utils.videoreader import VideoReader
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
import facer
from _preprocess import VideoTransforms
from xpose.inference_on_a_image import detect_one_image
from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor
import traceback


def simulate_head_motion(image, image1, num_frames, eye_patch, video_fps, interval=5, max_offset=8, max_rotation=0, max_scale=0.0):
    """
    Simulate head motion by applying random transformations to an image over a given number of frames.
    
    Parameters:
        image (np.ndarray): The input image.
        num_frames (int): The number of frames to generate.
        interval (int): Number of frames after which to change the transformation direction.
        max_offset (int): Maximum offset in pixels for each interval.
        max_rotation (float): Maximum rotation in degrees for each interval.
        max_scale (float): Maximum scale change for each interval.
    
    Returns:
        frames (list of np.ndarray): List of transformed frames.
    """
    h, w = image.shape[:2]
    frames = []
    frames1 = []
    
    cumulative_dx, cumulative_dy, cumulative_angle, cumulative_scale = 0, 0, 0, 1
    
    dx = random.uniform(-max_offset, max_offset) / interval
    dy = random.uniform(-max_offset, max_offset) / interval
    angle = random.uniform(-max_rotation, max_rotation) / interval
    scale = random.uniform(0, max_scale) / interval
    
    prev = 0
    ratio = [0.3, 0.6, 1, 0.6, 0.3]
    for i in range(num_frames):
        if i  - prev > (1 + 1.5 * random.random()) * video_fps: 
            prev = i + 4
        if i < prev:
            cur_image = np.copy(image)
            # print(i, prev, i - prev - 2)
            for patch in eye_patch[:2]:
                left, right, top, bottom = patch
                cur_image[int(top):int(bottom), int(left):int(right), :] = 0
                patch_area = image[int(top):int(bottom), int(left):int(right), :]
                padding = ratio[abs(i - prev + 2)]
                new_top  = top + ((bottom - top) / 2 * padding)
                new_bottom  = bottom - ((bottom - top) / 2 * padding)
                if int(new_bottom)- int(new_top) > 1:
                    cur_image[int(new_top):int(new_bottom), int(left):int(right), :] = cv2.resize(patch_area, (int(right)- int(left), int(new_bottom)- int(new_top)))
        else:
            cur_image = image
    
        if i % interval == 0 and i != 0:
            # Randomly change the transformation parameters
            dx = 0.25 * dx + 0.75 * random.uniform(-max_offset, max_offset) / interval
            dy = 0.25 * dy + 0.75 * random.uniform(-max_offset, max_offset) / interval
            angle = 0.25 * angle + 0.75 * random.uniform(-max_rotation, max_rotation) / interval
            scale = 0.25 * scale + 0.75 * random.uniform(-max_scale, max_scale) / interval
        
        # Calculate the cumulative transformations
        if cumulative_dx < 100 and cumulative_dx > -100:
            cumulative_dx += dx
        if cumulative_dy < 100 and cumulative_dy > -100:
            cumulative_dy += dy
        if cumulative_angle < 30 and cumulative_angle > -30:
            cumulative_angle += angle
        if cumulative_scale < 1.5 and cumulative_scale >  0.8:
            cumulative_scale += scale
        
        # Calculate transformation matrix for translation
        M_translation = np.float32([[1, 0, cumulative_dx], [0, 1, cumulative_dy]])
        
        # Calculate transformation matrix for rotation and scaling
        center = (w // 2, h // 2)
        M_rotation_scale = cv2.getRotationMatrix2D(center, cumulative_angle, cumulative_scale)
        
        # Apply translation
        translated_image = cv2.warpAffine(cur_image, M_translation, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Apply rotation and scaling
        transformed_image = cv2.warpAffine(translated_image, M_rotation_scale, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        frames.append(torch.tensor(transformed_image))
    

            # Apply translation
        translated_image1 = cv2.warpAffine(image1, M_translation, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Apply rotation and scaling
        transformed_image1 = cv2.warpAffine(translated_image1, M_rotation_scale, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        frames1.append(torch.tensor(transformed_image1))
    return frames, frames1


def eval(source_path, driver_path,
    config=None,
    config_path="",
    output_path="./", 
    random_seed=42,
    guidance_scale=4.5,
    weight_type=torch.float16, 
    num_steps=25,
    device=torch.device("cpu"), 
    model=None,
    clip_image_type="",
    concat_noise_image_type="",
    do_classifier_free_guidance="",
    contour_preserve_generation=False,
    frame_sample_config=[0, -1, 1],
    show_progressbar=True,
    visualization=True,
    no_audio=False,
    no_visual=False,
    second_limit = 2.5,
    fix=False,
    noseless=False,
    simulate=False,
    mouthless=True,
    **kwargs
    ):
    set_seed(random_seed)
    if config is None:
        config = OmegaConf.load(config_path)
    model_type = config.model_type
    MagicAnimate = getattr(importlib.import_module(f'animate.{model_type}.animate'), 'MagicAnimate')
    if model is None:
        pipeline = MagicAnimate(config=config,
                                train_batch_size=1,
                                device=device,
                                unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs))
    else:
        pipeline = model
    pipeline.to(device, dtype=weight_type)
    pipeline.eval()

    size = config.size
    control_data, audio_signal, meta_info = torchvision.io.read_video(driver_path, pts_unit='sec')
    video_fps = meta_info["video_fps"]
    control_data = control_data[:int(video_fps * second_limit)].numpy()

    audio_sampling_rate = meta_info['audio_fps']
    audio_signal = audio_signal[:, :int(audio_sampling_rate * second_limit)]

    print(f'Length of audio is {audio_signal.shape[1]} with the sampling rate of {audio_sampling_rate}.')
    if audio_sampling_rate != 16000:
        audio_signal = torchaudio.functional.resample(
            audio_signal,
            orig_freq=audio_sampling_rate,
            new_freq=16000,
        )
    audio_signal = audio_signal.mean(dim=0)
    audio_vis = audio_signal
    if no_audio:
        cur_seconds = control_data.shape[0] / video_fps
        audio_sampling_rate = 16000

        audio_signal = torch.zeros(int(16000 * min(second_limit, cur_seconds)))
 
    video_length = control_data.shape[0]
    
    if source_path.endswith(".mp4") or source_path.endswith(".mp4"):
        source_image_data = VideoReader(source_path).read()[0]
    else:
        source_image_data = Image.open(source_path)
        if np.array(source_image_data).shape[2] == 4:
            source_image_data = source_image_data.convert("RGB")
    
    source_image_data = np.array(source_image_data)
    source_image = torch.tensor(source_image_data).unsqueeze(0)
    ref_image = rearrange(source_image, "b h w c -> b c h w") # b c h w
    # ref_image = source_image.clone()  # .to(device, dtype=weight_type)
    faces_ref = [detect_one_image(source_image_data)[0]]
    if len(faces_ref[0]) == 0:
        ref_image = crop_and_resize_tensor(ref_image, target_size=size)
    elif contour_preserve_generation:
        _, _, ref_bbox, ref_image  =  crop_and_resize_tensor_xpose(ref_image, faces_ref, target_size=size)
    else:
        _, _, ref_bbox, ref_image  = crop_and_resize_tensor_xpose(ref_image, faces_ref, target_size=size)

    control = torch.tensor(control_data).to(torch.device("cpu"), dtype=weight_type)
    control = rearrange(control, "b h w c -> b c h w") # b c h w

    faces = [detect_one_image(control_data[i])[0] for i in range(control_data.shape[0])]
    _, _, all_face_rects, control_cropped  = crop_and_resize_tensor_xpose(control, faces, target_size=size)
    patch_search = [([22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47],12), ([17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41], 12), ([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67], 12)]
    eye_patch = list()
    if mouthless:
        patch_search = patch_search[:2]
    org_video = rearrange(
        control_cropped, "(b f) c h w -> b c f h w", b=1)
    cur_ref = ref_image.permute(0, 2, 3, 1)[0].cpu().numpy()

    if no_visual:
        patch_search = [([[27, 28, 29, 30, 31, 32, 33, 34, 35]], 8)]
        if noseless:
            patch_search = []
        if simulate:
            patch_search = [([22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47],12), ([17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41], 12), ([27, 28, 29, 30, 31, 32, 33, 34, 35], 8)]

        # cur_control = cur_ref
        cur_control = control_crop = control_cropped.permute(0, 2, 3, 1).cpu().numpy()[0]
        _, ldm = detect_one_image(cur_control)
        org_point = ldm = ldm[0]
        masked_frame = np.zeros_like(cur_control)
        right, bottom = org_point[:, :].max(axis=0)
        left, top = org_point[:, :].min(axis=0)
        src_point = np.array([[left, top], [right, bottom], [left, bottom]]).astype("int32")
        
        for patch_idx, (kp_indices, div_n) in enumerate(patch_search):
            # xs = ldm[kp_indices][..., 0]  # left eyes
            # ys = ldm[kp_indices][..., 1]  # left eyes
            # print(ldm.shape, ret)
            # print(ret)=1.5
            if patch_idx != 2:
                ret = np.mean(ldm[kp_indices], axis=0)
                x_mean, y_mean = ret # left eyes
                left, right, top, bottom = get_patch_div(x_mean, y_mean, size[0], size[1], div_n)
            else:
                xs = ldm[kp_indices][..., 0]  # left eyes
                ys = ldm[kp_indices][..., 1]  # left eyes
                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                left, right, top, bottom = get_patch(x0, y0, x1, y1, size[0], size[1], h_ratio=0.4 if patch_idx == 2 else 0.75, w_ratio=0.6  if patch_idx == 2 else 0.75)
  
            masked_frame[int(top):int(bottom), int(left):int(right), :] = cur_control[int(top):int(bottom), int(left):int(right), :]
            eye_patch.append([left, right, top, bottom])

        masked_frame1 = np.zeros_like(cur_control)
        for patch_idx, (kp_indices, div_n) in enumerate(patch_search[2:]):
            xs = ldm[kp_indices][..., 0]  # left eyes
            ys = ldm[kp_indices][..., 1]  # left eyes
            # ret = np.mean(ldm[kp_indices], axis=0)
            # print(ret)=1.5
            # x_mean, y_mean = ret # left eyes
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            left, right, top, bottom = get_patch(x0, y0, x1, y1, size[0], size[1], h_ratio=0.4, w_ratio=0.6)
            # left, right, top, bottom = get_patch_div(x_mean, y_mean, size[0], size[1], div_n)
            masked_frame1[int(top):int(bottom), int(left):int(right), :] = cur_control[int(top):int(bottom), int(left):int(right), :]
        control_frames = [masked_frame, masked_frame1]  # * video_length

        control_frames = [torch.Tensor(item) for item in control_frames]

        # pixel_values_pose = torch.zeros_like(control_cropped).unsqueeze(0)
        # pixel_values_vis = pixel_values_pose.clone().permute(0, 2, 1, 3, 4)
    else:
        control_crop = control_cropped.permute(0, 2, 3, 1).cpu().numpy()

        dist_box, dist_point = detect_one_image(cur_ref)
        dist_point = dist_point[0].reshape(-1, 2)  # * size[0]
        right, bottom = dist_point[:, :].max(axis=0)
        left, top = dist_point[:, :].min(axis=0)
        dist_point = np.array([[left, top], [right, bottom], [left, bottom]]).astype("int32")
        control_frames = []
        patch_indices = [[1e6, 0, 1e6, 0], ] * 4

        if not fix:
            for frame_index in range(video_length):
                cur_control = control_crop[frame_index]
                _, ldm = detect_one_image(cur_control)
                org_point = ldm = ldm[0]
                cur_control = control_crop[frame_index]
                masked_frame = np.zeros_like(cur_control)
                if frame_index == 0:
                    right, bottom = org_point[:, :].max(axis=0)
                    left, top = org_point[:, :].min(axis=0)
                    src_point = np.array([[left, top], [right, bottom], [left, bottom]]).astype("int32")
                for patch_idx, (kp_indices, div_n) in enumerate(patch_search):
                    xs = ldm[kp_indices][..., 0]  # left eyes
                    ys = ldm[kp_indices][..., 1]  # left eyes
                    # x_mean, y_mean = np.mean(ldm[kp_index_begin: kp_index_end], axis=0) # left eyes
                    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                    left, right, top, bottom = get_patch(x0, y0, x1, y1, size[0], size[1])

                    # left, right, top, bottom = get_patch_div(x_mean, y_mean, size[0], size[1], div_n)
                    masked_frame[int(top):int(bottom), int(left):int(right), :] = cur_control[int(top):int(bottom), int(left):int(right), :]
                control_frames.append(masked_frame)
        else:
            for frame_index in range(video_length):
                cur_control = control_crop[frame_index]
                _, ldm = detect_one_image(cur_control)
                org_point = ldm = ldm[0]
                cur_control = control_crop[frame_index]
                masked_frame = np.zeros_like(cur_control)
                if frame_index == 0:
                    right, bottom = org_point[:, :].max(axis=0)
                    left, top = org_point[:, :].min(axis=0)
                    src_point = np.array([[left, top], [right, bottom], [left, bottom]]).astype("int32")
                for patch_idx, (kp_indices, div_n) in enumerate(patch_search):
 
                    xmin, xmax, ymin, ymax = patch_indices[patch_idx]
                    x_mean, y_mean = np.mean(ldm[kp_indices, :], axis=0) # left eyes
                    left, right, top, bottom = get_patch_div(x_mean, y_mean, size[0], size[1], div_n)
                    xmin = min(xmin, left)
                    xmax = max(xmax, right)
                    ymin = min(ymin, top)
                    ymax = max(ymax, bottom)
                    patch_indices[patch_idx] = [xmin, xmax, ymin, ymax]

            for frame_index in range(video_length):
                cur_control = control_crop[frame_index]
                # masked_frame = cur_control / 2
                masked_frame = np.zeros_like(cur_control)
                for xmin, xmax, ymin, ymax in patch_indices:
                    masked_frame[int(ymin):int(ymax), int(xmin):int(xmax), :] = cur_control[int(ymin):int(ymax), int(xmin):int(xmax), :]
                control_frames.append(masked_frame)

        transform_matrix = cv2.getAffineTransform(np.float32(src_point), np.float32(dist_point))
        control_frames = [torch.Tensor(cv2.warpAffine(item, transform_matrix, size)) for item in control_frames]

    if simulate:
        # cv2.imwrite('x.png', control_frames[0].numpy().astype('uint8'))
        # cv2.imwrite('y.png', control_frames[1].numpy().astype('uint8'))
        # exit(-1)
        # control_frames2 = simulate_head_motion(control_frames[1].numpy(), video_length, video_fps)
        pre_control_frames = control_frames
        control_frames, control_frames1 = simulate_head_motion(pre_control_frames[0].numpy(), pre_control_frames[1].numpy(), video_length, eye_patch, video_fps, )
        pixel_values_pose = torch.stack(control_frames, dim=0).to(device, dtype=weight_type).permute(0, 3, 1, 2).unsqueeze(0)
        pixel_values_pose1 = torch.stack(control_frames1, dim=0).to(device, dtype=weight_type).permute(0, 3, 1, 2).unsqueeze(0)
        # pixel_values_pose2 = torch.stack(control_frames2, dim=0).to(device, dtype=weight_type).permute(0, 3, 1, 2).unsqueeze(0)
        # prev = 0
        # for i in range(video_length):
        #     if i  - prev > (1 + 1.5 * random.random()) * video_fps: 
        #         # blink_time = 3  # random.randint(2, 4)
        #         # pixel_values_pose[:, i: i + blink_time - 1, :, :, :] = pixel_values_pose1[:, i: i + blink_time - 1, :, :, :]
        #         pixel_values_pose[:, i: i, :, :, :] = pixel_values_pose[:, i: i + 1, :, :, :] * 0.5 + pixel_values_pose1[:, i: i + 1, :, :, :] * 0.5
        #         pixel_values_pose[:, i + 1: i + 2, :, :, :] = pixel_values_pose[:, i + 1: i + 2, :, :, :] * 0.3 + pixel_values_pose1[:, i + 1: i + 2, :, :, :] * 0.7
        #         pixel_values_pose[:, i + 2: i + 3, :, :, :] = pixel_values_pose1[:, i + 2: i + 3, :, :, :]
        #         pixel_values_pose[:, i + 3: i + 4, :, :, :] = pixel_values_pose[:, i + 3: i + 4, :, :, :] * 0.3 + pixel_values_pose1[:, i + 3: i + 4, :, :, :] * 0.7
        #         pixel_values_pose[:, i + 4: i + 5, :, :, :] = pixel_values_pose[:, i + 4: i + 5, :, :, :] * 0.5 + pixel_values_pose1[:, i + 4: i + 5, :, :, :] * 0.5
        #         # prev = i + blink_time - 1
        #         prev = i + 4
    else:
        pixel_values_pose = torch.stack(control_frames, dim=0).to(device, dtype=weight_type).permute(0, 3, 1, 2).unsqueeze(0)
    color_BW_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 1, 3, 1, 1).to(device, dtype=weight_type)
    pixel_values_pose = torch.sum(pixel_values_pose * color_BW_weights, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
    pixel_values_pose = pixel_values_pose.clamp(0, 255.)
    pixel_values_vis = pixel_values_pose.clone().permute(0, 2, 1, 3, 4)

    # print(source_image.shape, pixel_values_pose.shape)
    input_dict = pipeline.preprocess_eval(
        None, ref_image, pixel_values_pose, torch.zeros_like(audio_signal) if no_audio else audio_signal,
        guidance_scale=guidance_scale, do_classifier_free_guidance=do_classifier_free_guidance, driver_relax=0.4)
    context=config.context

    with torch.inference_mode():
        samples_per_video = pipeline.infer(
            step=num_steps,
            guidance_scale=guidance_scale,
            random_seed=random_seed,
            context=context,
            size=config.size,
            froce_text_embedding_zero=config.get('froce_text_embedding_zero', False),
            do_classifier_free_guidance=do_classifier_free_guidance,
            add_noise_image_type="",
            show_progressbar=True,
            visualization=visualization,
            fps=video_fps,
            ** input_dict
        )
    if isinstance(samples_per_video, list):
        samples_per_video[1] = pixel_values_vis.to(device=samples_per_video[0].device) / 255
        samples_per_video.insert(1, org_video / 255)
        print([i.shape for i in samples_per_video])
        samples_per_video = torch.cat(samples_per_video)
    video_name = os.path.basename(driver_path)[:-4]
    source_name = os.path.basename(
        source_path).split(".")[0]
    if output_path != '':
        if '.' not in output_path.split('/')[-1]:
            os.makedirs(output_path, exist_ok=True)
            save_videos_grid_audio(
                samples_per_video[:, :, 1:, ...], audio_signal, f"{output_path}/{source_name}_{video_name}.mp4", fps=video_fps)
        else:
            save_videos_grid_audio(
                samples_per_video[:, :, 1:, ...], audio_signal, f"{output_path}", fps=video_fps)
    else:
        save_videos_grid_audio(
            samples_per_video[:, :, 1:, ...], audio_signal, f"./{source_name}_{video_name}.mp4", fps=video_fps)
    return pipeline  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='Specify path of the config yaml for inference.')
    parser.add_argument("--source", type=str, default=None, help='Specify the source path, can be video (will use the first frame) or image.')
    parser.add_argument("--driver", type=str,  default=None, help='Specify the driving video path.')
    parser.add_argument("--output-path", type=str, default='', help='Specify the result video path.')
    parser.add_argument("--seed", type=int, default=42, help='Specify random seed.')
    parser.add_argument("--num-steps", type=int, default=25, help='Specify steps of denoising, more steps take more time to yield better result.')
    parser.add_argument("--guidance-scale", type=float, default=4.5, help='Specify classifier-free guidance scale.')
    parser.add_argument("--split", type=int, default=1, help='Specify classifier-free guidance scale.')
    parser.add_argument("--cur", type=int, default=0, help='Specify classifier-free guidance scale.')
    parser.add_argument("--limit", type=float, default=2.5, help='max seconds to eval.')
    parser.add_argument("--contour-preserve", action='store_true', help='Specify whether to mask the face other  than eyes and mouth to better align face shape.')
    parser.add_argument("--no-audio", action='store_true', help='Specify whether to mask the face other  than eyes and mouth to better align face shape.')
    parser.add_argument("--no-visual", action='store_true', help='Specify whether to mask the face other  than eyes and mouth to better align face shape.')
    parser.add_argument("--visualization", action='store_true', help='Specify whether to mask the face other  than eyes and mouth to better align face shape.')
    parser.add_argument("--fix", action='store_true', help='Specify whether to mask the face other  than eyes and mouth to better align face shape.')
    parser.add_argument("--noseless", action='store_true', help='Specify whether to mask the face other  than eyes and mouth to better align face shape.')
    parser.add_argument("--mouthless", action='store_true', help='Specify whether to mask the face other  than eyes and mouth to better align face shape.')
    parser.add_argument("--simulate", action='store_true', help='Specify whether to mask the face other  than eyes and mouth to better align face shape.')

    args = parser.parse_args()
    if args.source is not None and  args.driver is not None:
        eval(args.source, args.driver, 
            config=None,
            config_path=args.config,
            output_path=args.output_path, 
            random_seed=args.seed,
            guidance_scale=args.guidance_scale,
            weight_type=torch.float16, 
            num_steps=args.num_steps,
            device=torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"), 
            model=None,
            clip_image_type="background",
            concat_noise_image_type="origin",
            do_classifier_free_guidance=True,
            contour_preserve_generation=args.contour_preserve,
            frame_sample_config=[0, -1, 1],
            no_audio=args.no_audio,
            no_visual=args.no_visual,
            second_limit=args.limit,
            visualization=args.visualization,
            fix=args.fix,
            noseless=args.noseless,
            simulate=args.simulate,
            mouthless=args.mouthless
            )
    else:
        model = None
        conf_dict = OmegaConf.load(args.config)
        for s, d in tqdm(list(zip(conf_dict['source_image'], conf_dict['video_path']))[args.cur:: args.split],):
            try:
                model = eval(s, d, 
                    config=None,
                    config_path=args.config,
                    output_path=args.output_path, 
                    random_seed=args.seed,
                    guidance_scale=args.guidance_scale,
                    weight_type=torch.float16, 
                    num_steps=args.num_steps,
                    device=torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"), 
                    model=model,
                    clip_image_type="background",
                    concat_noise_image_type="origin",
                    do_classifier_free_guidance=True,
                    contour_preserve_generation=args.contour_preserve,
                    frame_sample_config=[0, -1, 1],
                    no_audio=args.no_audio,
                    no_visual=args.no_visual,
                    second_limit=args.limit,
                    visualization=args.visualization,
                    fix=args.fix,
                    noseless=args.noseless,
                    simulate=args.simulate,
                    mouthless=args.mouthless
                    )
            except:
                traceback.print_exc()
                continue
