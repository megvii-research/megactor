import os
import imageio
import numpy as np
from typing import Union

import torch
import torchvision
import torch.distributed as dist

from safetensors import safe_open
from tqdm import tqdm
from einops import rearrange
from animate.utils.convert_lora_safetensor_to_diffusers import convert_lora, convert_motion_lora_ckpt_to_diffusers
from PIL import Image, ImageOps


def pad_image(image):
    # Get the dimensions of the image
    width, height = image.size

    # Calculate the padding needed
    if width < height:
        diff = height - width
        padding = (diff // 2, 0, diff - (diff // 2), 0)  # left, top, right, bottom
    else:
        diff = width - height
        padding = (0, diff // 2, 0, diff - (diff // 2))  # left, top, right, bottom

    # Pad the image and return
    return ImageOps.expand(image, padding)

tensor_interpolation = None

def get_tensor_interpolation_method():
    return tensor_interpolation

def set_tensor_interpolation_method(is_slerp):
    global tensor_interpolation
    tensor_interpolation = slerp if is_slerp else linear

def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2

def slerp(
    v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
) -> torch.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        #logger.info(f'warning: v0 and v1 close to parallel, using linear interpolation instead.')
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=None, save_every_image=False, dir_path=None):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    video_length = videos.shape[0]
    outputs = []
    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        if x.max() <= 1.0:
            x = (x * 255).numpy().astype(np.uint8)
        else:
            x = x.numpy().astype(np.uint8)
        
        outputs.append(x)

    # os.makedirs(os.path.dirname(path), exist_ok=True)
    if fps is None:
        fps = (video_length // 2) if video_length > 1 else 1
    
    if path.endswith('.gif'):
        imageio.mimsave(path, outputs, fps=fps, loop=0)
    else:
        imageio.mimsave(path, outputs, fps=fps, codec='libx264')
    
    if save_every_image:
        dir_base_path = path[:-4]
        os.makedirs(dir_base_path, exist_ok=True)
        for i, x in enumerate(videos):
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            if x.max() <= 1.0:
                x = (x * 255).numpy().astype(np.uint8)
            else:
                x = x.numpy().astype(np.uint8)
            
            Image.fromarray(x).save(f"{dir_base_path}/_{i}.png")

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


def load_weights(
        animation_pipeline,
        # motion module
        motion_module_path="",
        motion_module_lora_configs=[],
        # image layers
        dreambooth_model_path="",
        lora_model_path="",
        lora_alpha=0.8,
):
    # 1.1 motion module
    unet_state_dict = {}
    if motion_module_path != "":
        print(f"load motion module from {motion_module_path}")
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        motion_module_state_dict = motion_module_state_dict[
            "state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
        unet_state_dict.update(
            {name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})

    missing, unexpected = animation_pipeline.unet.load_state_dict(unet_state_dict, strict=False)
    assert len(unexpected) == 0
    del unet_state_dict

    if dreambooth_model_path != "":
        print(f"load dreambooth model from {dreambooth_model_path}")
        if dreambooth_model_path.endswith(".safetensors"):
            dreambooth_state_dict = {}
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
        elif dreambooth_model_path.endswith(".ckpt"):
            dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")

        # 1. vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, animation_pipeline.vae.config)
        animation_pipeline.vae.load_state_dict(converted_vae_checkpoint)
        # 2. unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, animation_pipeline.unet.config)
        animation_pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
        # 3. text_model
        animation_pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
        del dreambooth_state_dict

    if lora_model_path != "":
        print(f"load lora model from {lora_model_path}")
        assert lora_model_path.endswith(".safetensors")
        lora_state_dict = {}
        with safe_open(lora_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                lora_state_dict[key] = f.get_tensor(key)

        animation_pipeline = convert_lora(animation_pipeline, lora_state_dict, alpha=lora_alpha)
        del lora_state_dict

    for motion_module_lora_config in motion_module_lora_configs:
        path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]
        print(f"load motion LoRA from {path}")

        motion_lora_state_dict = torch.load(path, map_location="cpu")
        motion_lora_state_dict = motion_lora_state_dict[
            "state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict

        animation_pipeline = convert_motion_lora_ckpt_to_diffusers(animation_pipeline, motion_lora_state_dict, alpha)

    return animation_pipeline

from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import cv2
def generate_random_params(image_width, image_height):
    """生成包含随机参数的字典"""
    # 生成起始点（图像的四个角）
    startpoints = [(0, 0), (image_width, 0), (image_width, image_height), (0, image_height)]
    max_offset = int(0.2 * image_width)
    # 生成结束点，每个点在原位置基础上加上一个随机偏移
    endpoints = [
        (random.randint(-max_offset, max_offset), random.randint(-max_offset, max_offset)),
        (image_width + random.randint(-max_offset, max_offset), random.randint(-max_offset, max_offset)),
        (image_width + random.randint(-max_offset, max_offset), image_height + random.randint(-max_offset, max_offset)),
        (random.randint(-max_offset, max_offset), image_height + random.randint(-max_offset, max_offset))
    ]
    params = {
        'rotate': random.uniform(-5, 5),  # 在-30到30度之间随机选择一个角度
        'affine': {
            'degrees': random.uniform(0, 0),  # 仿射变换的角度，这里设定为-15到15度之间
            'translate': (random.uniform(-0.0, 0.0), random.uniform(-0.0, 0.0)),  # 平移比例
            'scale': random.uniform(0.8, 1.2),  # 缩放比例
            'shear': random.uniform(0, 0),  # 剪切强度
        },
        'perspective': {'distortion_scale': random.uniform(0.1, 0.5), "startpoints": startpoints, "endpoints": endpoints},  # 透视变换强度
        'flip': {'horizontal': random.random() < 0.5, 'vertical': random.random() < 0.5},  # 翻转概率
        'aspect_ratio': random.uniform(0.8, 1.2),  # 宽高比调整
    }
    return params

def apply_transforms(image, params=None):
    if not isinstance(image, torch.Tensor):
        H, W = image.size
    else:
        assert len(image.shape) == 4
        H, W = image.shape[2:]

    # 调整宽高比后，还原到原始大小
    if 'aspect_ratio' in params and params['aspect_ratio'] != 1.:
        # Current shape of the image tensor

        # Resize the image while preserving aspect ratio
        target_size = (H, int(params['aspect_ratio']*W)) if params['aspect_ratio'] <= 1. else (int(H/params['aspect_ratio']), W)
        resized_image = TF.resize(image, target_size, antialias=True)

        # Get the size of the resized image
        resized_height, resized_width = target_size

        # Calculate the padding needed to make the resized image centered in the original size
        left_pad = (W - resized_width) // 2
        top_pad = (H - resized_height) // 2
        right_pad = W - resized_width - left_pad
        bottom_pad = H - resized_height - top_pad

        # Apply padding to the resized image
        image = TF.pad(resized_image, (int(left_pad), int(top_pad), int(right_pad), int(bottom_pad)))

    # 应用仿射变换
    if 'affine' in params:
        angle = params['affine']['degrees']
        translate = params['affine']['translate']
        scale = params['affine']['scale']
        shear = params['affine']['shear']
        image = TF.affine(image, angle=angle, 
                    translate=(translate[0] * H, translate[1] * W), 
                    scale=scale, 
                    shear=shear)

    return image

def crop_and_resize_tensor(frame, target_size = (512, 512), crop_rect = None, center = None):
    # 假设 frame 是 (B, C, H, W) 的格式
    b, _, height, width = frame.shape

    if crop_rect is not None:
        left, top, right, bottom = crop_rect
        face_w = right - left
        face_h = bottom - top
        # padding = max(face_w, face_h) // 2
        
        if face_w < face_h:
            left = left - (face_h - face_w) // 2
            right = right + (face_h - face_w) // 2
        else:
            top = top - (face_w - face_h) // 2
            bottom = bottom + (face_w - face_h) // 2
        # left, top, right, bottom = left - padding, top - padding, right + padding, bottom + padding
        left, top, right, bottom = max(left, 0), max(top, 0), min(right, width), min(bottom, height)

    elif center is not None:
        # 假设已经给定了 center_x, center_y 以及原始图像的 width 和 height
        center_x, center_y = center
        # 计算从中心点到图像边界的最小距离
        distance_to_edge = min(center_x, center_y, width - center_x, height - center_y)
        # 使用这个距离来确定裁剪的正方形边界
        left = center_x - distance_to_edge
        right = center_x + distance_to_edge
        top = center_y - distance_to_edge
        bottom = center_y + distance_to_edge
        # 确保裁剪的坐标不会超出图像的原始尺寸
        left = max(left, 0)
        right = min(right, width)
        top = max(top, 0)
        bottom = min(bottom, height)

    else:
        short_edge = min(height, width)
        left = (width - short_edge) // 2
        top = (height - short_edge) // 2
        right = left + short_edge
        bottom = top + short_edge
    
    frame_cropped = frame[:, :, int(top):int(bottom), int(left):int(right)].float()
    target_height, target_width = target_size
    frame_resized = torch.nn.functional.interpolate(frame_cropped, size=(target_height, target_width), mode='bilinear', align_corners=False)
    return frame_resized, (int(top), int(bottom), int(left), int(right))

def crop_and_resize_tensor_with_face_rects(pixel_values, faces, target_size = (512, 512)) -> torch.Tensor:
    L, __, H, W = pixel_values.shape
    l, t, r, b = W, H, 0, 0
    min_face_size = 1
    all_face_rects = list()
    for i in range(L):
        face_rects = list()
        for j, ids in enumerate(faces['image_ids']):
            if i == ids:
                face_rects.append(faces['rects'][j])
        if len(face_rects) == 0:
            if len(all_face_rects) > 0:
                face_rects = [all_face_rects[-1]]
            else:
                return None, None, None, None
        face_rect = face_rects[0]
        left, top, right, bottom = face_rect
        min_face_size = min(min_face_size, (right - left) * (bottom - top) / (H * W))
        l = min(left, l)
        t = min(top, t)
        r = max(r, right)
        b = max(b, bottom)
        all_face_rects.extend(face_rects[:1])
    face_center = ((l + r) // 2, (t + b) // 2)
    output_values, bbox = crop_and_resize_tensor(pixel_values, target_size=target_size, center=face_center)

    return min_face_size, all_face_rects, bbox, output_values
                

def crop_and_resize_tensor_face(pixel_values : torch.Tensor,
                        target_size = (512, 512),
                        crop_face_center = True, face_detector = None) -> torch.Tensor:
    # 和crop_and_resize_tensor_face 一样，但是把人脸放到中间，并且裁剪人脸大小占整个图像的0.25
    pixel_values = pixel_values.to("cuda")
    pixel_values_det = pixel_values.clone()
    assert face_detector is not None
    face_stride = len(pixel_values_det) // 32
    if face_stride == 0:
        face_stride += 1
    faces = face_detector(pixel_values_det[::face_stride])
    if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0 or not crop_face_center:
        pixel_values = crop_and_resize_tensor(pixel_values_det, target_size=target_size)
        # print("no face find in first frame")
    else:
        L, __, H, W = pixel_values.shape
        l, t, r, b = W, H, 0, 0
        for i in range(L):
            face_rects = []
            for j, ids in enumerate(faces['image_ids']):
                if i == ids:
                    face_rects.append(faces['rects'][j])
            if len(face_rects) == 0:
                continue
            face_rect = face_rects[0]
            left, top, right, bottom = face_rect
            l = min(left, l)
            t = min(top, t)
            r = max(r, right)
            b = max(b, bottom)
            center_x, center_y = (l + r) // 2, (t + b) // 2

        pixel_values, bbox = crop_and_resize_tensor(pixel_values, target_size=target_size, center=(center_x, center_y))

    return pixel_values.cpu()

def crop_move_face(frames, faces, target_size = (512, 512), top_margin=0.4, bottom_margin=0.1):
    # 将除要裁剪的人脸以外的其他区域全部保留下来，但是涂成黑色，只有人脸区域保留
    # is_get_head: 是否获取包含更多头部（发际线以上的部位作为Condition）
    # 假设 frame 是 (B, C, H, W) 的格式
    L = frames.shape[0]
    output_sequence = list()
    b, channels, height, width = frames.shape
    all_face_rects = []
    for i in range(L):
        frame = frames[i: i + 1]
        face_rects = []
        for j, ids in enumerate(faces['image_ids']):
            if i == ids:
                face_rects.append(faces['rects'][j])
        if len(face_rects) == 0:
            if len(all_face_rects) > 0:
                face_rects = [all_face_rects[-1]]
            else:
                return None, None, None, None
        face_rect = face_rects[0]
        all_face_rects.append(face_rect)
        left, top, right, bottom = face_rect
        face_w = right - left
        face_h = bottom - top
        # padding = max(face_w, face_h) // 2
        
        if face_w < face_h:
            left = left - (face_h - face_w) // 2
            right = right + (face_h - face_w) // 2
        else:
            top = top - (face_w - face_h) // 2
            bottom = bottom + (face_w - face_h) // 2

        delta_hight = (bottom - top)
        top -= top_margin * delta_hight
        bottom += bottom_margin * delta_hight
        left, top, right, bottom = max(left, 0), max(top, 0), min(right, width), min(bottom, height)
        
        frame_cropped = torch.zeros_like(frame).float()
        move_face = frame[:, :, int(top):int(bottom), int(left):int(right)].float()
        frame_cropped[:, :, int(top):int(bottom), int(left):int(right)] = move_face
        # frame_cropped = frame_cropped[:, :, ptop: pbottom, pleft: pright]

        target_height, target_width = target_size
        frame_resized = torch.nn.functional.interpolate(frame_cropped, size=(target_height, target_width), mode='bilinear', align_corners=False)
        output_sequence.append(frame_resized)
    output_frames = torch.cat(output_sequence, dim=0)
    return output_frames


# def crop_move_face(frames, faces, p_bbox, target_size = (512, 512), top_margin=0.4, bottom_margin=0.1):
#     # 将除要裁剪的人脸以外的其他区域全部保留下来，但是涂成黑色，只有人脸区域保留
#     # is_get_head: 是否获取包含更多头部（发际线以上的部位作为Condition）
#     # 假设 frame 是 (B, C, H, W) 的格式
#     L = frames.shape[0]
#     output_sequence = list()
#     b, channels, height, width = frames.shape
#     ptop, pbottom, pleft, pright = p_bbox
#     for i in range(L):
#         frame = frames[i: i + 1]

#         left, top, right, bottom = faces[i]
#         face_w = right - left
#         face_h = bottom - top
        
#         if face_w < face_h:
#             left = left - (face_h - face_w) // 2
#             right = right + (face_h - face_w) // 2
#         else:
#             top = top - (face_w - face_h) // 2
#             bottom = bottom + (face_w - face_h) // 2

#         delta_hight = (bottom - top)
#         top -= top_margin * delta_hight
#         bottom += bottom_margin * delta_hight
#         left, top, right, bottom = max(left, 0), max(top, 0), min(right, width), min(bottom, height)

#         frame_cropped = torch.zeros_like(frame).float()
#         move_face = frame[:, :, int(top):int(bottom), int(left):int(right)].float()
#         frame_cropped[:, :, int(top):int(bottom), int(left):int(right)] = move_face
#         frame_cropped = frame_cropped[:, :, ptop: pbottom, pleft: pright]

#         target_height, target_width = target_size
#         frame_resized = torch.nn.functional.interpolate(frame_cropped, size=(target_height, target_width), mode='bilinear', align_corners=False)
#         output_sequence.append(frame_resized)
#     output_frames = torch.cat(output_sequence, dim=0)
#     return output_frames


def get_rect_length(left, top, right, bottom, width, height):
    # 获取一个，以给定矩形中心为中心的最大外部正方形
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    distance_to_edge = min(center_x, center_y, width - center_x, height - center_y)
    return 2 * distance_to_edge, center_x, center_y

def wide_crop_face(pixel_values, faces, target_size = (512, 512)) -> torch.Tensor:
    L, __, H, W = pixel_values.shape
    l, t, r, b = W, H, 0, 0
    min_face_size = 1
    all_face_rects = list()
    for i in range(L):
        face_rects = list()
        for j, ids in enumerate(faces['image_ids']):
            if i == ids:
                face_rects.append(faces['rects'][j])
        if len(face_rects) == 0:
            if len(all_face_rects) > 0:
                face_rects = [all_face_rects[-1]]
            else:
                return None, None, None, None
        face_rect = face_rects[0]
        left, top, right, bottom = face_rect
        min_face_size = min(min_face_size, (right - left) * (bottom - top) / (H * W))
        l = min(left, l)
        t = min(top, t)
        r = max(r, right)
        b = max(b, bottom)
        all_face_rects.extend(face_rects)
    w, h = (r - l), (b - t)
    x_c, y_c = (l + r) / 2, (t + b) / 2
    expand_dis = max(w, h)
    left, right = max(x_c - expand_dis * 0.9, 0), min(x_c + expand_dis * 0.9, W)
    top, bottom = max(y_c - expand_dis, 0), min(y_c + expand_dis * 0.8, H)
    # Get new center and new rect
    x_c, y_c = (left + right) / 2, (bottom + top) / 2
    distance_to_edge = min(x_c - left, right - x_c, y_c - top, bottom - y_c)
    left = x_c - distance_to_edge
    right = x_c + distance_to_edge
    top = y_c - distance_to_edge
    bottom = y_c + distance_to_edge
    face_center = ((l + r) // 2, (t + b) // 2)
    bbox = [int(top), int(bottom), int(left), int(right)]
    pixel_values = pixel_values[:, :, int(top):int(bottom), int(left):int(right)].float()
    target_height, target_width = target_size
    output_values = torch.nn.functional.interpolate(pixel_values, size=(target_height, target_width), mode='bilinear', align_corners=False)
    return min_face_size, all_face_rects, bbox, output_values


def get_patch_div(x_mean, y_mean, H, W, n):
    # 以x_mean, y_mean为中心，生成一个大小为H/4,W/4的矩形
    # 计算矩形半高和半宽
    half_height = H / n
    half_width = W / n

    # 计算矩形的边界
    xmin = max(0, x_mean - half_width)
    xmax = min(W, x_mean + half_width)
    ymin = max(0, y_mean - half_height)
    ymax = min(H, y_mean + half_height)

    return int(xmin), int(xmax), int(ymin), int(ymax)

def get_025_gaze_mouth(control:torch.Tensor, 
                    origin_video:torch.Tensor, 
                    dwpose_model, 
                    faces,
                    target_size=(512, 512),
                    move_face=True, is_get_head=True, is_get_gaze=True):
    # 推理时使用。获取以眼睛和嘴巴为中心的，大小为H/4,W/4 patch的图像
    # control: b, c, h, w
    # origin_video: b c h w
    # return control_condition (fix with ref-image), origin control video after crop
    # is_get_head 是否多裁一些人脸，让头部也保留，只在move_face为True的时候有用
    H, W = control.shape[2:]
    control_crop = control.clone()
    control = rearrange(control, "b c h w -> b h w c")
    control = control.numpy() # b h w c, numpy
    if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
        face_rect = [None] * len(control_crop)
    else:
        face_rect = []
        for i in range(len(control_crop)):
            for j, ids in enumerate(faces['image_ids']):
                if i == ids:
                    face_rect.append(faces['rects'][j])
                    break
            if len(face_rect) == i:
                face_rect.append(None)                                                
    face_image_list = []
    for i, face_rect_item in enumerate(face_rect):
        face_image = crop_move_face(control_crop[i].unsqueeze(0), 
                                    target_size=target_size, 
                                    crop_rect=face_rect_item, 
                                    is_get_head=is_get_head)            
        face_image_list.append(face_image)
    
    # 获取source image和control condition的landmark，并进行基于眼睛和嘴巴中心的对齐
    control_crop = rearrange(control_crop, "b c h w -> b h w c").numpy()
    control_crop = control_crop.astype("uint8")
    origin_video = rearrange(origin_video, "b c h w -> b h w c")
    origin_video = origin_video[0].numpy() # h w c, numpy
    _, __, source_landmark = dwpose_model.dwpose_model(origin_video, output_type='np', image_resolution=H, get_mark=True)
    source_landmark = source_landmark["faces_all"][0] * target_size[0]
    _, __, control_landmark = dwpose_model.dwpose_model(control_crop[0], output_type='np', image_resolution=H, get_mark=True)
    control_landmark = control_landmark["faces_all"][0] * target_size[0]
    src_point = control_landmark
    right, bottom = src_point[:, :].max(axis=0)
    left, top = src_point[:, :].min(axis=0)
    src_point = np.array([[left, top], [right, bottom], [left, bottom]]).astype("int32")
    dist_point = source_landmark
    right, bottom = dist_point[:, :].max(axis=0)
    left, top = dist_point[:, :].min(axis=0)
    dist_point = np.array([[left, top], [right, bottom], [left, bottom]]).astype("int32")
    
    transform_matrix = cv2.getAffineTransform(np.float32(src_point), np.float32(dist_point))
    # 获取对齐矩阵后，再将moveFace结果进行转移
    control_crop = torch.cat(face_image_list).cpu()
    control_crop = rearrange(control_crop, "b c h w -> b h w c").numpy().astype('uint8')
    # 转移完moveFace的结果后，再获取025的gaze_mouth
    frame_gaze_list = []
    for control_item in control_crop:
        _, __, ldm = dwpose_model.dwpose_model(control_item, output_type='np', image_resolution=target_size[0], get_mark=True)
        ldm = ldm["faces_all"][0] * target_size[0]
        frame_gaze = np.zeros_like(control_item)
        x_mean, y_mean = np.mean(ldm[60 - 24: 66 - 24], axis=0) # left eyes
        xmin, xmax, ymin, ymax = get_patch_025(x_mean, y_mean, target_size[0], target_size[1])
        frame_gaze[int(ymin):int(ymax), int(xmin):int(xmax), :] = control_item[int(ymin):int(ymax), int(xmin):int(xmax), :]
        
        x_mean, y_mean = np.mean(ldm[66 - 24: 72 - 24], axis=0) # right eyes
        xmin, xmax, ymin, ymax = get_patch_025(x_mean, y_mean, target_size[0], target_size[1])
        frame_gaze[int(ymin):int(ymax), int(xmin):int(xmax), :] = control_item[int(ymin):int(ymax), int(xmin):int(xmax), :]
        x_mean, y_mean = np.mean(ldm[72 - 24: 92 - 24], axis=0) # mouth
        xmin, xmax, ymin, ymax = get_patch_025(x_mean, y_mean, target_size[0], target_size[1])
        frame_gaze[int(ymin):int(ymax), int(xmin):int(xmax), :] = control_item[int(ymin):int(ymax), int(xmin):int(xmax), :]
        frame_gaze_list.append(frame_gaze)
    frame_gaze_list = np.array(frame_gaze_list).astype('uint8')    
    # 转移完后再使用对齐矩阵进行对齐操作
    control_aligns = []
    frame_gaze_aligns = []
    for item, item_gaze_mouth in zip(control_crop, frame_gaze_list):
        aligned_img = cv2.warpAffine(item, transform_matrix, (H, W))
        aligned_item_gaze_mouth = cv2.warpAffine(item_gaze_mouth, transform_matrix, (H, W)) 
        control_aligns.append(aligned_img)
        frame_gaze_aligns.append(aligned_item_gaze_mouth)
    control_crop = np.array(control_aligns)
    frame_gaze_list = np.array(frame_gaze_aligns)
    return control_crop, control, frame_gaze_list


def crop_move_face_org(
    frame : torch.Tensor,
    target_size = (512, 512), 
    crop_rect = None, 
    use_mask_rate=0., mask_rate=0., color_jit_rate=0., 
    is_get_head=False,) -> torch.Tensor:
    # 假设 frame 是 (B, C, H, W) 的格式
    b, channels, height, width = frame.shape
    
    if crop_rect is not None:
        left, top, right, bottom = crop_rect
        face_w = right - left
        face_h = bottom - top
        # padding = max(face_w, face_h) // 2
        
        if face_w < face_h:
            left = left - (face_h - face_w) // 2
            right = right + (face_h - face_w) // 2
        else:
            top = top - (face_w - face_h) // 2
            bottom = bottom + (face_w - face_h) // 2

        if is_get_head:
            delta_hight = (bottom - top)
            top -= 0.4 * delta_hight
            bottom += 0.1 * delta_hight

        left, top, right, bottom = max(left, 0), max(top, 0), min(right, width), min(bottom, height)

    else:
        # 相当于整张图片都被设置成1了
        left = 0
        top = 0
        right = 2
        bottom = 2

    frame_cropped = torch.zeros_like(frame).float()
    move_face = frame[:, :, int(top):int(bottom), int(left):int(right)].float()
    frame_cropped[:, :, int(top):int(bottom), int(left):int(right)] = move_face

    target_height, target_width = target_size
    frame_resized = torch.nn.functional.interpolate(frame_cropped, size=(target_height, target_width), mode='bilinear', align_corners=False)
    
    cur_mask_rate = random.uniform(0, 1)
    if mask_rate != 0. and cur_mask_rate < use_mask_rate:
        num_patches_per_dim = 16
        patch_size = target_height // num_patches_per_dim
        mask = torch.zeros(target_height, target_width)
        mask_rect_rate = torch.rand(num_patches_per_dim, num_patches_per_dim)
        for i in range(num_patches_per_dim):
            for j in range(num_patches_per_dim):
                if mask_rect_rate[i, j] < mask_rate:
                    mask[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 1.
        mask = mask.to(dtype=torch.bool)
        frame_resized[:, :, mask] = 0.

    return frame_resized

def get_condition_face(control:torch.Tensor, 
                    origin_video:torch.Tensor, 
                    dwpose_model, 
                    face_detector, local_rank, weight_type, 
                    switch_control_to_source = False,
                    target_size=(512, 512),
                    move_face=False, is_get_head=False, is_get_gaze=False):
    # control: b, c, h, w
    # origin_video: b c h w
    # return control_condition (fix with ref-image), origin control video after crop
    # is_get_head 是否多裁一些人脸，让头部也保留，只在move_face为True的时候有用
    H, W = control.shape[2:]
    control_crop = control.clone()
    control = rearrange(control, "b c h w -> b h w c")
    control = control.cpu().numpy() # b h w c, numpy
    faces = face_detector(control_crop.to(device=local_rank, dtype=weight_type))
    if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
        face_rect = [None] * len(control_crop)
    else:
        face_rect = []
        for i in range(len(control_crop)):
            for j, ids in enumerate(faces['image_ids']):
                if i == ids:
                    face_rect.append(faces['rects'][j])
                    break
            if len(face_rect) == i:
                face_rect.append(None)                                                
    face_image_list = []
    for i, face_rect_item in enumerate(face_rect):
        face_image = crop_move_face_org(control_crop[i].unsqueeze(0), 
                                    target_size=target_size, 
                                    crop_rect=face_rect_item, 
                                    is_get_head=is_get_head)            
        face_image_list.append(face_image)
    
    control_crop = torch.cat(face_image_list).cpu()
    control_crop = rearrange(control_crop, "b c h w -> b h w c").numpy()

    if is_get_gaze:
        frame_gaze_list = []
        for control_item in control_crop:
            _, __, ldm = dwpose_model.dwpose_model(control_item, output_type='np', image_resolution=target_size[0], get_mark=True)
            ldm = ldm["faces_all"][0] * target_size[0]
            frame_gaze = np.zeros_like(control_item)
            xmin = np.min(ldm[60 - 24: 72 - 24, 0])
            xmax = np.max(ldm[60 - 24: 72 - 24, 0])
            ymin = np.min(ldm[60 - 24: 72 - 24, 1])
            ymax = np.max(ldm[60 - 24: 72 - 24, 1])
            frame_gaze[int(ymin):int(ymax), int(xmin):int(xmax), :] = control_item[int(ymin):int(ymax), int(xmin):int(xmax), :]
            # Image.fromarray(frame_gaze.astype('uint8')).save("infer_gaze.png")
            # Image.fromarray(control_item.astype('uint8')).save("infer_origin.png")
            frame_gaze_list.append(frame_gaze)
        return control_crop, control, np.array(frame_gaze_list)

    return control_crop, control                

import subprocess
import threading
def get_checkpoint(path):
    base_path = path.split("/")[-1]
    if path.find("s3://") == -1:
        return torch.load(path, map_location="cpu")
    else:
        subprocess.run(['aws',
                        f'--endpoint-url=http://oss.hh-b.brainpp.cn',
                        's3',
                        'cp',
                        path,
                        f"./"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        ans = torch.load(f"./{base_path}", map_location="cpu")
        def remove():
            subprocess.run(['rm', '-rf', f"./{base_path}"], check=True)
        thread = threading.Thread(target=remove)
        thread.start()
        return ans
