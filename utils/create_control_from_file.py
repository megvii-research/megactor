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

from tqdm import tqdm

from animatediff.utils.util import save_videos_grid
from animatediff.utils.dist_tools import distributed_init
from accelerate.utils import set_seed

from animatediff.utils.videoreader import VideoReader

from einops import rearrange

from pathlib import Path
from megfile import smart_open
import io
from animatediff.utils.util import save_videos_grid, pad_image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor


def crop_and_resize(frame, target_size, crop_rect=None, is_arcface=False):
    height, width = frame.size
    if is_arcface:
        target_size = (112, 112)

    if crop_rect is not None:
        left, top, right, bottom = crop_rect
        face_w = right-left
        face_h = bottom-top
        padding = max(face_w, face_h) // 2
        if face_w < face_h:
            left = left - (face_h-face_w)//2
            right = right + (face_h-face_w)//2
        else:
            top = top - (face_h-face_w)//2
            bottom = bottom + (face_h-face_w)//2
        left, top, right, bottom = left-padding, top-padding, right+padding, bottom+padding 
    else:
        short_edge = min(height, width)
        width, height = frame.size
        top = (height - short_edge) // 2
        left = (width - short_edge) // 2
        right = (width + short_edge) // 2
        bottom = (height + short_edge) // 2
    frame_cropped = frame.crop((left, top, right, bottom))
    frame_resized = frame_cropped.resize(target_size, Image.ANTIALIAS)
    return frame_resized

def save_one_control(images, savedir, source_name):
    pixel_values_pose = torch.Tensor(np.array(images))
    pixel_values_pose = rearrange(
        pixel_values_pose, "(b f) h w c -> b f h w c", b=1)
    pixel_values_pose = pixel_values_pose / 255.0
    pixel_values_pose = rearrange(pixel_values_pose, "b f h w c -> b c f h w")
    pixel_values_pose = pixel_values_pose.cpu()
    save_videos_grid(
            pixel_values_pose, f"{savedir}/{source_name}.gif")

def main(args):

    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    config = OmegaConf.load(args.config)

    # Initialize distributed training
    device = torch.device(f"cuda:{args.rank}")
    weight_type = torch.float16
    dist_kwargs = {"rank": args.rank,
                   "world_size": args.world_size, "dist": args.dist}

    source_images = config.source_image

    dwpose_model = DenseDWposePredictor(device)

    for idx, source_image in tqdm(
        enumerate(source_images),
        total=len(source_images),
        disable=(args.rank != 0)
    ):

        if source_image.endswith(".mp4") or source_image.endswith(".gif"):
            origin_video = VideoReader(source_image).read()
            item_list = []
            for item in origin_video:
                item = Image.fromarray(item)
                item = crop_and_resize(item, (512, 512), crop_rect=None)
                item_list.append(item)
            origin_video = np.array(item_list)
        else:
            source_image = Image.open(source_image)
            source_image = crop_and_resize(source_image, (512, 512), crop_rect=None)
            origin_video = np.array(source_image)[None, ...]

        with torch.inference_mode():
            densepose_list = []
            dwpose_list = []
            densepose_dwpose_list = []
            foreground_list = []
            for pil_image in origin_video:
                control_res = dwpose_model(pil_image)
                densepose_list.append(control_res['densepose'])
                dwpose_list.append(control_res['dwpose_all'])
                densepose_dwpose_list.append(control_res['densepose_dwpose'])
                foreground_list.append(control_res['foreground'])

                # control_res = dwpose_model(pil_image, control_aux_type="densepose_dwpose_concat")
                # densepose_dwpose_list.append(control_res)
               

        densepose_list = np.array(densepose_list)
        dwpose_list = np.array(dwpose_list)
        densepose_dwpose_list = np.array(densepose_dwpose_list)
        foreground_list = np.array(foreground_list)

        
        if args.rank == 0:
            savedir = 'produce_control'
            os.makedirs(savedir, exist_ok=True)
            source_name = os.path.basename(
                config.source_image[idx]).split(".")[0]
            # np.save(f'{savedir}/{source_name}_dense_dw_concat.npy', densepose_dwpose_list)
            # save_one_control(densepose_list, savedir, source_name+'_dense')
            save_one_control(dwpose_list, savedir, source_name+'_dwposeall')
            # save_one_control(densepose_dwpose_list, savedir, source_name+'_dense_dw')
            # save_one_control(foreground_list, savedir, source_name+'_fore')
            

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
