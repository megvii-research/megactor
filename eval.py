import argparse
import datetime
import inspect
import os
import random
from tqdm import tqdm
from pathlib import Path
import io
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import torch
import torch.nn.functional as F

from animate.utils.util import save_videos_grid, get_condition_face, pad_image
from animate.utils.util import crop_and_resize_tensor_face
from accelerate.utils import set_seed
from animate.utils.videoreader import VideoReader
from animate.unet_magic_noiseAttenST_Ada.animate import MagicAnimate    
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
import facer
from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor


def main(args):

    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    config = OmegaConf.load(args.config)
    device = torch.device(f"cuda:0")
    weight_type = torch.float16

    test_video_path = args.driver
    source_image_path = args.source
    output_path = args.output_path
    seed = args.seed
    guidance_scale = args.guidance_scale
    num_steps = args.num_steps


    do_classifier_free_guidance = config.get(
        "do_classifier_free_guidance", True
    )
    clip_image_type = config.get(
        "clip_image_type", "foreground"
    )
    concat_noise_image_type = config.get(
        "concat_noise_image_type", ""
    )
    ref_image_type = config.get(
        "ref_image_type", "origin"
    )
    add_noise_image_type = config.get(
        "add_noise_image_type", ""
    )
    save_every_image = config.get(
        "save_every_image", False
    )
    model_type = config.get(
        "model_type", "unet"
    )
    switch_control_to_source = config.get(
        "switch_control_to_source", True
    )
    crop_face_center = config.get(
        "crop_face_center", True
    )
    control_aux_type = config.control_aux_type
    guidance_scale = config.guidance_scale

    pipeline = MagicAnimate(config=config,
                            train_batch_size=1,
                            device=device,
                            unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs))
    
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    face_detector.requires_grad_(False)
    pipeline.to(device, dtype=weight_type)
    pipeline.eval()

    dwpose_model = DenseDWposePredictor(device, resolution=config.size)

    # -------- IP adapter encoder--------#
    if clip_image_type != "":

        image_processor = CLIPImageProcessor.from_pretrained(config.pretrained_model_path, subfolder="feature_extractor")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(config.pretrained_model_path, subfolder="image_encoder")
        
        image_encoder.to(device, weight_type)
        image_encoder.requires_grad_(False)
        # face_detector = facer.face_detector('retinaface/mobilenet', device=device)

    random_seed = seed
    size = config.size
    steps = config.S

    config.random_seed = []
    samples_per_video = []
    samples_per_clip = []
    # manually set random seed for reproduction
    if random_seed != -1:
        print(f'manual random seed is {random_seed}')
        torch.manual_seed(random_seed)
        set_seed(random_seed)
    else:
        torch.seed()
    config.random_seed.append(torch.initial_seed())

    # Load control and source image
    if test_video_path.endswith('.mp4') or test_video_path.endswith('.gif'):
        print('Control Condition', test_video_path)
        control = VideoReader(test_video_path).read()[::10]
        video_length = control.shape[0]
        print('control', control.shape)
    else:
        print("!!!WARNING: SKIP this case since it is not a video")
    
    print('Reference Image', source_image_path)
    if source_image_path.endswith(".mp4") or source_image_path.endswith(".gif"):
        source_image = VideoReader(source_image_path).read()[0]
    else:
        source_image = Image.open(source_image_path)
        if np.array(source_image).shape[2] == 4:
            source_image = source_image.convert("RGB")

    source_image = torch.tensor(np.array(source_image)).unsqueeze(0)
    source_image = rearrange(source_image, "b h w c -> b c h w") # b c h w
    control = torch.tensor(control)
    control = rearrange(control, "b h w c -> b c h w") # b c h w

    control = crop_and_resize_tensor_face(control, size, crop_face_center=crop_face_center, face_detector=face_detector)
    source_image = crop_and_resize_tensor_face(source_image, size, crop_face_center=crop_face_center, face_detector=face_detector)

    ref_img_condition = source_image.clone() / 255.
    ref_img_condition = ref_img_condition.to(device, dtype=weight_type)

    control_condition, control = get_condition_face(control, source_image, dwpose_model, 
                                                    face_detector, device, weight_type, 
                                                    switch_control_to_source = True, 
                                                    target_size=size, move_face=True, 
                                                    is_get_head=True)

    pixel_values_pose = torch.Tensor(np.array(control_condition))

    color_BW_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 1, 1, 3)
    pixel_values_pose = torch.sum(pixel_values_pose * color_BW_weights, dim=3, keepdim=True).repeat(1, 1, 1, 3)
    pixel_values_pose = pixel_values_pose.clamp(0, 255.)

    pixel_values_pose = rearrange(
        pixel_values_pose, "(b f) h w c -> b f h w c", b=1)
    pixel_values_pose = pixel_values_pose.to(device, dtype=weight_type)
    pixel_values_pose = pixel_values_pose / 255.

    
    with torch.inference_mode():
        source_image_pil = Image.fromarray(source_image[0].permute(1, 2, 0).numpy().astype("uint8"))
        dwpose_model_result_dict = dwpose_model(source_image_pil)
        # Image.fromarray(ref_image_control).save('ref_image_control.png')
        ref_img_foreground = dwpose_model_result_dict['foreground']
        ref_img_convert = dwpose_model_result_dict[ref_image_type]
        if concat_noise_image_type != "":
            ref_concat_image_noise = dwpose_model_result_dict[concat_noise_image_type]
            ref_img_background_mask = dwpose_model_result_dict['background_mask']
        if add_noise_image_type != "":
            ref_add_image_noise = dwpose_model_result_dict[add_noise_image_type]
        if clip_image_type != "":
            ref_img_clip = dwpose_model_result_dict[clip_image_type]  
            ref_img_clip = Image.fromarray(ref_img_clip)

    source_image = np.array(source_image_pil)
    if ref_image_type != "origin":
        source_image = ref_img_convert
    source_image = ((torch.Tensor(source_image).unsqueeze(
        0).to(device, dtype=weight_type) / 255.0) - 0.5) * 2

    B, H, W, C = source_image.shape

    # concat noise with background latents
    ref_concat_image_noises_latents = None
    if concat_noise_image_type != "":
        ref_concat_image_noises = torch.Tensor(np.array(ref_concat_image_noise)).unsqueeze(0).to(device, dtype=weight_type)
        one_img_have_more = False
        if len(ref_concat_image_noises.shape) == 5:
            ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b f h w c -> (b f) h w c')
            one_img_have_more = True
        ref_concat_image_noises = rearrange(ref_concat_image_noises, 'b h w c -> b c h w')
        ref_concat_image_noises = ref_concat_image_noises / 127.5 - 1
        ref_concat_image_noises_latents = pipeline.vae.encode(ref_concat_image_noises).latent_dist
        ref_concat_image_noises_latents = ref_concat_image_noises_latents.sample().unsqueeze(2)
        ref_concat_image_noises_latents = ref_concat_image_noises_latents * 0.18215
        
        if one_img_have_more == True:
            B, C, _, H, W = ref_concat_image_noises_latents.shape
            ref_concat_image_noises_latents = ref_concat_image_noises_latents.reshape(B//2, C*2, _, H, W)

        ref_img_back_mask_latents = torch.tensor(np.array(ref_img_background_mask)[None, ...].transpose(0, 3, 1, 2)).to(device, dtype=weight_type)
        H, W = ref_concat_image_noises_latents.shape[3:]
        ref_img_back_mask_latents = F.interpolate(ref_img_back_mask_latents, size=(H, W), mode='nearest').unsqueeze(2)
        ref_concat_image_noises_latents = torch.cat([
            ref_concat_image_noises_latents, ref_img_back_mask_latents
        ], dim=1).repeat(1, 1, video_length, 1, 1)

        if guidance_scale > 1.0 and do_classifier_free_guidance:
            ref_concat_image_noises_latents = torch.cat([ref_concat_image_noises_latents,
                ref_concat_image_noises_latents])

    ######################### image encoder#########################
    image_prompt_embeddings = None
    if clip_image_type != "":
        with torch.inference_mode():
            clip_image = image_processor(
                images=ref_img_clip, return_tensors="pt").pixel_values
            image_emb = image_encoder(clip_image.to(
                device, dtype=weight_type), output_hidden_states=True).last_hidden_state
            image_emb = image_encoder.vision_model.post_layernorm(image_emb)
            image_emb = image_encoder.visual_projection(image_emb)# image_emb = image_encoder.vision_model.post_layernorm(image_emb)

            image_prompt_embeddings = image_emb
            if guidance_scale > 1.0 and do_classifier_free_guidance:
                image_prompt_embeddings = torch.cat([image_emb, image_emb])

    context=config.context
    with torch.inference_mode():
        source_image = rearrange(source_image, 'b h w c -> b c h w')
        samples_per_video = pipeline.infer(
            source_image=source_image,
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
            add_noise_image_type=add_noise_image_type,
            ref_img_condition=ref_img_condition,
            visualization=False
        )

    if output_path != '':
        save_videos_grid(
            samples_per_video[:, :, 1:, ...], output_path, save_every_image=False, fps=25)
    else:
        video_name = os.path.basename(test_video_path)[:-4]
        source_name = os.path.basename(
            source_image_path).split(".")[0]
        save_videos_grid(
            samples_per_video[:, :, 1:, ...], f"./{source_name}_{video_name}.mp4", save_every_image=False, fps=25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='Specify path of the config yaml for inference.')
    parser.add_argument("--source", type=str, required=True, help='Specify the source path, can be video (will use the first frame) or image.')
    parser.add_argument("--driver", type=str, required=True, help='Specify the driving video path.')
    parser.add_argument("--output-path", type=str, default='', help='Specify the result video path.')
    parser.add_argument("--seed", type=int, default=42, help='Specify random seed.')
    parser.add_argument("--guidance-scale", type=float, default=4.5, help='Specify classifier-free guidance scale.')
    parser.add_argument("--num-steps", type=int, default=25, help='Specify steps of denoising, more steps take more time to yield better result.')

    args = parser.parse_args()
    main(args)
