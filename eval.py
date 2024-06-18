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

import torch
import torch.nn.functional as F

from animate.utils.util import save_videos_grid, pad_image, crop_move_face, crop_and_resize_tensor_with_face_rects, crop_and_resize_tensor, wide_crop_face, get_patch_div
from animate.utils.util import crop_and_resize_tensor_face
from accelerate.utils import set_seed
from animate.utils.videoreader import VideoReader
from animate.unet_magic_noiseAttenST_Ada.animate import MagicAnimate    
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
import facer
from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor


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
    # face_detector=None,
    # dwpose_model=None,
    image_processor=None,
    image_encoder=None,
    clip_image_type="",
    concat_noise_image_type="",
    do_classifier_free_guidance="",
    contour_preserve_generation=False,
    frame_sample_config=[0, -1, 1],
    show_progressbar=True,
    visualization=False
    ):
    set_seed(random_seed)
    if config is None:
        config = OmegaConf.load(config_path)
    if model is None:
        pipeline = MagicAnimate(config=config,
                                train_batch_size=1,
                                device=device,
                                unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs))
    else:
        pipeline = model
    pipeline.to(device, dtype=weight_type)
    pipeline.eval()

    face_detector = facer.face_detector('retinaface/mobilenet', device=torch.device("cpu"))
    face_detector.requires_grad_(False)
    dwpose_model = DenseDWposePredictor("cpu", resolution=config.size)

    if image_processor is None:
        image_processor = CLIPImageProcessor.from_pretrained(config.pretrained_model_path, subfolder="feature_extractor")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(config.pretrained_model_path, subfolder="image_encoder")
        image_encoder.to(device, weight_type)
        image_encoder.requires_grad_(False)

    size = config.size

    # Load control and source image
    control_data = VideoReader(driver_path).read()[frame_sample_config[0]:frame_sample_config[1]:frame_sample_config[2]][:50]
    video_length = control_data.shape[0]
    
    if source_path.endswith(".mp4") or source_path.endswith(".gif"):
        source_image_data = VideoReader(source_path).read()[0]
    else:
        source_image_data = Image.open(source_path)
        if np.array(source_image_data).shape[2] == 4:
            source_image_data = source_image_data.convert("RGB")
    
    source_image_data = np.array(source_image_data)
    source_image = torch.tensor(source_image_data).unsqueeze(0)
    source_image = rearrange(source_image, "b h w c -> b c h w") # b c h w
    ref_image = source_image.clone().to(device, dtype=weight_type)

    faces_ref = face_detector(ref_image.cpu())
    if 'image_ids' not in faces_ref.keys() or faces_ref['image_ids'].numel() == 0:
        ref_image = crop_and_resize_tensor(ref_image, target_size=size)
    elif contour_preserve_generation:
        _, _, ref_bbox, ref_image  =  crop_and_resize_tensor_with_face_rects(ref_image, faces_ref, target_size=size)
    else:
        _, _, ref_bbox, ref_image  = crop_and_resize_tensor_with_face_rects(ref_image, faces_ref, target_size=size)

    control = torch.tensor(control_data).to(torch.device("cpu"), dtype=weight_type)
    control = rearrange(control, "b h w c -> b c h w") # b c h w
    faces = face_detector(control)
    _, all_face_rects, control_bbox, control_cropped  = crop_and_resize_tensor_with_face_rects(control, faces, target_size=size)

    cur_ref = ref_image.permute(0, 2, 3, 1)[0].cpu().numpy()
    control_crop = control_cropped.permute(0, 2, 3, 1).cpu().numpy()
    if contour_preserve_generation:
        _, __, dist_point = dwpose_model.dwpose_model(cur_ref, output_type='np', image_resolution=size[0], get_mark=True)
        dist_point = dist_point["faces_all"][0] * size[0]
        right, bottom = dist_point[:, :].max(axis=0)
        left, top = dist_point[:, :].min(axis=0)
        dist_point = np.array([[left, top], [right, bottom], [left, bottom]]).astype("int32")
        control_frames = []
        patch_indices = [[1e6, 0, 1e6, 0], ] * 4

        for frame_index in range(video_length):
            cur_control = control_crop[frame_index]
            _, __, ldm = dwpose_model.dwpose_model(cur_control, output_type='np', image_resolution=size[0], get_mark=True)
            ldm = ldm["faces_all"][0] * size[0]

            for patch_idx, (kp_index_begin, kp_index_end, div_n) in enumerate([(36, 42, 8), (42, 48, 8), (48, 68, 8), (17, 27, 8)]):
                xmin, xmax, ymin, ymax = patch_indices[patch_idx]
                x_mean, y_mean = np.mean(ldm[kp_index_begin: kp_index_end], axis=0) # left eyes
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

            if frame_index == 0:
                right, bottom = ldm[:, :].max(axis=0)
                left, top = ldm[:, :].min(axis=0)
                src_point = np.array([[left, top], [right, bottom], [left, bottom]]).astype("int32")
        transform_matrix = cv2.getAffineTransform(np.float32(src_point), np.float32(dist_point))
        control_frames = [torch.Tensor(cv2.warpAffine(item, transform_matrix, size)) for item in control_frames]
        pixel_values_pose = torch.stack(control_frames, dim=0).to(device, dtype=weight_type).permute(0, 3, 1, 2)
        
    else:        
        cropped_faces = face_detector(control_cropped)
        control_cropped = control_cropped.to(device, dtype=weight_type)
        pixel_values_pose = crop_move_face(control_cropped, cropped_faces, target_size=size)

    cv2.imwrite('test.png', pixel_values_pose[0].permute(1, 2, 0).cpu().numpy().astype('uint8'))
    color_BW_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(device, dtype=weight_type)
    pixel_values_pose = torch.sum(pixel_values_pose * color_BW_weights, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    pixel_values_pose = pixel_values_pose.clamp(0, 255.)

    pixel_values_pose = rearrange(
        pixel_values_pose, "(b f) c h w -> b f c h w", b=1)

    with torch.inference_mode():
        ref_concat_image_noises = []
        ref_img_background_masks = []
        ref_img_clips = []
        image_np = rearrange(ref_image, "b c h w -> b h w c")
        image_np = image_np.cpu().numpy().astype(np.uint8)
        for i, ref_img in enumerate(image_np):
            ref_img = Image.fromarray(ref_img)
            dwpose_model_result_dict = dwpose_model(ref_img)

            ref_concat_image_noise = dwpose_model_result_dict[concat_noise_image_type]
            ref_concat_image_noises.append(torch.tensor(ref_concat_image_noise).permute(2, 0, 1))

            ref_img_background_mask = dwpose_model_result_dict['background_mask']
            ref_img_background_masks.append(torch.tensor(ref_img_background_mask).squeeze())                                

            ref_img_clip = dwpose_model_result_dict[clip_image_type]
            ref_img_clips.append(torch.tensor(ref_img_clip).permute(2, 0, 1))
        concat_poses = torch.stack(ref_concat_image_noises, dim=0)[None, ...]
        concat_background = torch.stack(ref_img_background_masks, dim=0)[None, ...]
        # concat_background = - F.max_pool2d(-concat_background,  7, padding=3)
        clip_conditions = torch.stack(ref_img_clips, dim=0)[None, ...]

    pixel_values_ref_img, image_prompt_embeddings, pixel_values_pose, ref_concat_image_noises_latents, ref_img_condition = pipeline.preprocess_eval(
        ref_image, pixel_values_pose, concat_poses, concat_background, clip_conditions, image_processor, image_encoder,
        guidance_scale=guidance_scale, do_classifier_free_guidance=do_classifier_free_guidance)
    context=config.context

    with torch.inference_mode():
        samples_per_video = pipeline.infer(
            source_image=pixel_values_ref_img,
            image_prompts=image_prompt_embeddings,
            motion_sequence=pixel_values_pose,
            step=num_steps,
            guidance_scale=guidance_scale,
            random_seed=random_seed,
            context=context,
            size=config.size,
            froce_text_embedding_zero=config.get('froce_text_embedding_zero', False),
            ref_concat_image_noises_latents=ref_concat_image_noises_latents,
            do_classifier_free_guidance=do_classifier_free_guidance,
            add_noise_image_type="",
            ref_img_condition=ref_img_condition,
            show_progressbar=True,
            visualization=False
        )
    if isinstance(samples_per_video, list):
        samples_per_video = torch.cat(samples_per_video)
    video_name = os.path.basename(driver_path)[:-4]
    source_name = os.path.basename(
        source_path).split(".")[0]
    if output_path != '':
        if os.path.exists(output_path):
            save_videos_grid(
                samples_per_video[:, :, 1:, ...], f"{output_path}/{source_name}_{video_name}.mp4", save_every_image=False, fps=25)
        else:
            save_videos_grid(
                samples_per_video[:, :, 1:, ...], f"{output_path}", save_every_image=False, fps=25)
    else:
        save_videos_grid(
            samples_per_video[:, :, 1:, ...], f"./{source_name}_{video_name}.mp4", save_every_image=False, fps=25)
    return model, image_processor, image_encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='Specify path of the config yaml for inference.')
    parser.add_argument("--source", type=str, required=True, help='Specify the source path, can be video (will use the first frame) or image.')
    parser.add_argument("--driver", type=str, required=True, help='Specify the driving video path.')
    parser.add_argument("--output-path", type=str, default='', help='Specify the result video path.')
    parser.add_argument("--seed", type=int, default=42, help='Specify random seed.')
    parser.add_argument("--num-steps", type=int, default=25, help='Specify steps of denoising, more steps take more time to yield better result.')
    parser.add_argument("--guidance-scale", type=float, default=4.5, help='Specify classifier-free guidance scale.')
    parser.add_argument("--contour-preserve", action='store_true', help='Specify whether to mask the face other  than eyes and mouth to better align face shape.')

    args = parser.parse_args()
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
        # face_detector=None,
        # dwpose_model=None,
        image_processor=None,
        image_encoder=None,
        clip_image_type="background",
        concat_noise_image_type="origin",
        do_classifier_free_guidance=True,
        contour_preserve_generation=args.contour_preserve,
        frame_sample_config=[0, -1, 1]
        )
