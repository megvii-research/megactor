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
import facer
from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor


def main(args):

    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    config = OmegaConf.load(args.config)
    # Initialize distributed training
    device = torch.device(f"cuda:{args.rank}")
    weight_type = torch.float16
    dist_kwargs = {"rank": args.rank,
                   "world_size": args.world_size, "dist": args.dist}

    if config.savename is None:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = f"samples/{Path(args.config).stem}-{time_str}"
    else:
        savedir = f"samples/{config.savename}"
    
    if args.dist:
        dist.broadcast_object_list([savedir], 0)
        dist.barrier()

    if args.rank == 0:
        os.makedirs(savedir, exist_ok=True)
        OmegaConf.save(config, f"{savedir}/config.yaml")

    # inference_config = OmegaConf.load(config.inference_config)

    test_videos = config.video_path
    source_images = config.source_image
    animate_images = config.animate_image
    num_actual_inference_steps = config.get(
        "num_actual_inference_steps", config.steps)

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
    if model_type in ["unet"]:
        from animatediff.magic_animate.unet_model.animate import MagicAnimate
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
    if model_type in ["unet_condition_refer"]:
        from animatediff.magic_animate.unet_condition_refer.animate import MagicAnimate
    if model_type in ["unet_condition"]:
        from animatediff.magic_animate.unet_condition.animate import MagicAnimate
    if model_type in ["unet_magic_noiseAttenST"]:
            from animatediff.magic_animate.unet_magic_noiseAttenST.animate import MagicAnimate

    pipeline = MagicAnimate(config=config,
                            train_batch_size=1,
                            device=device,
                            unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs))
    
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    face_detector.requires_grad_(False)
    pipeline.to(device, dtype=weight_type)
    pipeline.eval()

    dwpose_model = DenseDWposePredictor(device)

    # -------- IP adapter encoder--------#
    if clip_image_type != "":
        # image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        #     config.image_encoder_path).to(device)
        # image_encoder.requires_grad_(False)
        # image_processor = CLIPImageProcessor()

        image_processor = CLIPImageProcessor.from_pretrained(config.pretrained_model_path, subfolder="feature_extractor")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(config.pretrained_model_path, subfolder="image_encoder")
        
        # from controlnet_resource.arcface_backbones import get_model
        # arcface_encoder = get_model('r100', fp16=False)
        # arcface_weight_path = '/root/.cache/yangshurong/magic_pretrain/arcface_backbone.pth'
        # arcface_encoder.load_state_dict(torch.load(arcface_weight_path))
        # arcface_encoder.to(device, weight_type)
        # arcface_encoder.requires_grad_(False)
        # arcface_encoder.eval()

        image_encoder.to(device, weight_type)
        image_encoder.requires_grad_(False)
        # face_detector = facer.face_detector('retinaface/mobilenet', device=device)

    ### <<< create validation pipeline <<< ###

    random_seeds = config.get("valid_seed", [-1])
    random_seeds = [random_seeds] if isinstance(
        random_seeds, int) else list(random_seeds)
    random_seeds = random_seeds * \
        len(config.source_image) if len(random_seeds) == 1 else random_seeds

    # input test videos (either source video/ conditions)

    # read size, step from yaml file
    sizes = [config.size] * len(test_videos)
    steps = [config.S] * len(test_videos)

    config.random_seed = []
    prompt = n_prompt = ""

    for idx, (source_image, test_video, animate_image, random_seed, size, step) in tqdm(
        enumerate(zip(source_images, test_videos, animate_images, random_seeds, sizes, steps)),
        total=len(test_videos),
        disable=(args.rank != 0)
    ):
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

        print('Animate Image', animate_image)
        if animate_image.endswith(".mp4") or animate_image.endswith(".gif"):
            animate_image = VideoReader(animate_image).read()[0]
        else:
            animate_image = Image.open(animate_image)
            if np.array(animate_image).shape[2] == 4:
                animate_image = animate_image.convert("RGB")

        animate_image = np.array(animate_image)

        source_image = torch.tensor(np.array(source_image)).unsqueeze(0)
        source_image = rearrange(source_image, "b h w c -> b c h w") # b c h w
        control = torch.tensor(control)
        control = rearrange(control, "b h w c -> b c h w") # b c h w

        # print("crop control")
        control = crop_and_resize_tensor_face(control, size, crop_face_center=crop_face_center, face_detector=face_detector)
        # print("crop source_image")
        source_image = crop_and_resize_tensor_face(source_image, size, crop_face_center=crop_face_center, face_detector=face_detector)
            # print("source image shape is", np.array(source_image).shape, np.unique(np.array(source_image)))

        control_condition, control = get_condition(control, source_image, dwpose_model, control_aux_type, switch_control_to_source = switch_control_to_source)

        pixel_values_pose = torch.Tensor(np.array(control_condition))
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
            ref_img_condition = dwpose_model_result_dict[control_aux_type]
            if concat_noise_image_type != "":
                ref_concat_image_noise = dwpose_model_result_dict[concat_noise_image_type]
                ref_img_background_mask = dwpose_model_result_dict['background_mask']
            if add_noise_image_type != "":
                ref_add_image_noise = dwpose_model_result_dict[add_noise_image_type]
            if clip_image_type != "":
                # ANIMATE CHANGE:1
                # using background mask from source, to get background from animate image
                ref_img_background_mask = dwpose_model_result_dict['background_mask']
                ref_img_clip = ref_img_background_mask * animate_image
                ref_img_clip = Image.fromarray(ref_img_clip.astype("uint8"))
                # ref_img_clip.save("check_Ani_C1.png")

        ref_img_condition = torch.Tensor(np.array(ref_img_condition)).unsqueeze(0).to(device, dtype=weight_type)
        ref_img_condition = rearrange(ref_img_condition, 'b h w c -> b c h w')
        ref_img_condition = ref_img_condition / 255.
        source_image = np.array(source_image_pil)
        if ref_image_type != "origin":
            source_image = ref_img_convert
        # ANIMATE CHANGE:1
        # source image is animate image
        source_image = ((torch.Tensor(source_image).unsqueeze(
            0).to(device, dtype=weight_type) / 255.0) - 0.5) * 2
        # source_image = ((torch.Tensor(animate_image).unsqueeze(
        #     0).to(device, dtype=weight_type) / 255.0) - 0.5) * 2

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
            # print('ref_img_backgrounds unique is', ref_img_backgrounds.unique())
            ref_concat_image_noises_latents = pipeline.vae.encode(ref_concat_image_noises).latent_dist
            ref_concat_image_noises_latents = ref_concat_image_noises_latents.sample().unsqueeze(2)
            ref_concat_image_noises_latents = ref_concat_image_noises_latents * 0.18215
            
            if one_img_have_more == True:
                B, C, _, H, W = ref_concat_image_noises_latents.shape
                ref_concat_image_noises_latents = ref_concat_image_noises_latents.reshape(B//2, C*2, _, H, W)

            ref_img_back_mask_latents = torch.tensor(np.array(ref_img_background_mask)[None, ...].transpose(0, 3, 1, 2)).to(device, dtype=weight_type)
            H, W = ref_concat_image_noises_latents.shape[3:]
            ref_img_back_mask_latents = F.interpolate(ref_img_back_mask_latents, size=(H, W), mode='nearest').unsqueeze(2)
            # print('infer ref_img_back_mask_latents unique is', ref_image_back_mask_latents.unique())
            ref_concat_image_noises_latents = torch.cat([
                ref_concat_image_noises_latents, ref_img_back_mask_latents
            ], dim=1).repeat(1, 1, video_length, 1, 1)

            if guidance_scale > 1.0 and do_classifier_free_guidance:
                # ref_concat_image_noises_latents = torch.cat([torch.zeros_like(ref_concat_image_noises_latents),
                #     ref_concat_image_noises_latents])
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
                    # guidance free
                    image_prompt_embeddings = torch.cat([torch.zeros_like(image_emb), image_emb])
                    # image_prompt_embeddings = torch.cat([image_emb, image_emb])
                    print("Arrive Clip")


        context=config.context
        # context['context_frames'] = 1
        with torch.inference_mode():
            source_image = rearrange(source_image, 'b h w c -> b c h w')
            samples_per_video = pipeline.infer(
                source_image=source_image,
                image_prompts=image_prompt_embeddings,
                motion_sequence=pixel_values_pose,
                step=config.steps,
                guidance_scale=config.guidance_scale,
                random_seed=random_seed,
                context=context,
                size=config.size,
                froce_text_embedding_zero=config.get('froce_text_embedding_zero', False),
                ref_concat_image_noises_latents=ref_concat_image_noises_latents,
                do_classifier_free_guidance=do_classifier_free_guidance,
                add_noise_image_type=add_noise_image_type,
                ref_img_condition=ref_img_condition,
            )

            if control_aux_type == "densepose_dwpose_concat":
                control_condition = torch.tensor(control_condition).unsqueeze(0)
                control_condition = rearrange(control_condition, 'b t h w c -> b c t h w') / 255.
                samples_per_video[1] = control_condition
            

            # shape need to be 1 c t h w
            source_image = np.array(source_image_pil) # h w c
            source_image = torch.Tensor(source_image).unsqueeze(
                        0) / 255.
            source_image = source_image.repeat(video_length, 1, 1, 1)
            samples_per_video[0] = rearrange(source_image, "t h w c -> 1 c t h w") 
            
            control = torch.tensor(control).unsqueeze(0)
            control = rearrange(control, 'b t h w c -> b c t h w') / 255.
            samples_per_video.insert(0, control)
            samples_per_video = torch.cat(samples_per_video)

        

        if args.rank == 0:
            video_name = os.path.basename(test_video)[:-4]
            source_name = os.path.basename(
                config.source_image[idx]).split(".")[0]
            save_videos_grid(
                samples_per_video[:, :, 1:, ...], f"{savedir}/videos/{source_name}_{video_name}.gif", save_every_image=False)
            # save_videos_grid(
            #     samples_per_video[-1:], f"{savedir}/videos/{source_name}_{video_name}.gif")
            # save_videos_grid(
            #     samples_per_video, f"{savedir}/videos/{source_name}_{video_name}/grid.gif")

            # if config.save_individual_videos:
            #     save_videos_grid(
            #         samples_per_video[1:2], f"{savedir}/videos/{source_name}_{video_name}/ctrl.gif")
            #     save_videos_grid(
            #         samples_per_video[0:1], f"{savedir}/videos/{source_name}_{video_name}/orig.gif")
                

        if args.dist:
            dist.barrier()

    


def distributed_main(device_id, args):
    args.rank = device_id
    args.device_id = device_id
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        torch.cuda.init()
    distributed_init(args)
    main(args)


def run(args):

    if args.dist:
        args.world_size = max(1, torch.cuda.device_count())
        assert args.world_size <= torch.cuda.device_count()

        if args.world_size > 0 and torch.cuda.device_count() > 1:
            port = random.randint(10000, 20000)
            args.init_method = f"tcp://localhost:{port}"
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args,),
                nprocs=args.world_size,
            )
    else:
        main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dist", action="store_true", required=False)
    parser.add_argument("--rank", type=int, default=0, required=False)
    parser.add_argument("--world_size", type=int, default=1, required=False)

    args = parser.parse_args()
    run(args)
