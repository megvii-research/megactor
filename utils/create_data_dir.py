# import os
# import math
# import wandb
# import random
# import logging
# import inspect
# import argparse
# import datetime
# import subprocess
# import torchvision
# import torch.nn.functional as F
# import torch.distributed as dist
# from torch.optim.swa_utils import AveragedModel
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from pathlib import Path
# from omegaconf import OmegaConf
# from safetensors import safe_open
# from typing import Dict, Optional, Tuple

# import diffusers
# from diffusers import AutoencoderKL, DDIMScheduler
# from diffusers.models import UNet2DConditionModel
# from diffusers.pipelines import StableDiffusionPipeline
# from diffusers.optimization import get_scheduler
# from diffusers.utils import check_min_version
# from diffusers.utils.import_utils import is_xformers_available

# import transformers
# from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
# # from ip_adapter import IPAdapterFull
# from accelerate import Accelerator
# from einops import repeat
# from animate import MagicAnimate
# from animatediff.magic_animate.controlnet import ControlNetModel
# import importlib
# from animatediff.data.dataset import WebVid10M, PexelsDataset

# from animatediff.data.dataset import WebVid10M, PexelsDataset
from animatediff.utils.util import save_videos_grid, pad_image, crop_and_resize_tensor
from PIL import Image
from tqdm.auto import tqdm
from einops import rearrange
import torch
import numpy as np

import webdataset as wds
from face_dataset import S3VideosIterableDataset

local_rank = 0
weight_type = 'float16'
det_config = '/root/code/yangshurong/VividVideoGeneration/controlnet_resource/controlnet_aux/src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
det_ckpt = '/root/.cache/yangshurong/magic_pretrain/control_aux/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
pose_config = '/root/code/yangshurong/VividVideoGeneration/controlnet_resource/controlnet_aux/src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py'
pose_ckpt = '/root/.cache/yangshurong/magic_pretrain/control_aux/dw-ll_ucoco_384.pth'

# dwpose_model = DWposeDetector(
#             det_config=det_config,
#             det_ckpt=det_ckpt,
#             pose_config=pose_config,
#             pose_ckpt=pose_ckpt,
#             device=local_rank)

# samdetect = SamDetector.from_pretrained("/root/.cache/yangshurong/magic_pretrain/models--ybelkada--segment-anything/checkpoints", filename="sam_vit_b_01ec64.pth", model_type='vit_b')
# densepredict = DensePosePredictor(local_rank)
# dwpose_model = DenseDWposePredictor(local_rank)

dataset = S3VideosIterableDataset(
    [
    # 's3://public-datasets/Datasets/Videos/processed/CelebV_webdataset_20231211_videoblip',
    # 's3://public-datasets/Datasets/Videos/processed/pexels_20231217',
    # 's3://public-datasets/Datasets/Videos/processed/VFHQ_webdataset_20240404/'
    's3://public-datasets/Datasets/Videos/processed/VFHQ_webdataset_20240410_dwpose_facebbox',
    's3://public-datasets/Datasets/Videos/processed/CelebV_webdataset_2040410_dwpose_facebbox/',
    's3://public-datasets/Datasets/Videos/processed/pexels_20240410_dwpose_facebbox/'

    # 's3://ljj/Datasets/Videos/processed/hdvila100m_20231216',
    # 's3://ljj/Datasets/Videos/processed/xiaohongshu_webdataset_20231212',
    ],
    video_length   = 16,
    resolution     = [512, 512],
    frame_stride   = 2,
    dataset_length = 100000,
    shuffle        = True,
    resampled      = True,
    
)
# import pdb; pdb.set_trace()

dataloader = wds.WebLoader(
    dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=1,
    collate_fn = None,
).with_length(len(dataset))
# pbar = tqdm()
import cv2
save_num = 1000
for idx, batch in tqdm(enumerate(dataloader)):
    pixel_values = batch["pixel_values"].to(local_rank)
    pixel_values_pose = batch["dwpose_all"].to(local_rank)
    # pixel_values_ref = batch["pixel_values_ref"].to(local_rank)
    pixel_values = rearrange(pixel_values, "b t c h w -> b c t h w")
    pixel_values_pose = rearrange(pixel_values_pose, "b t c h w -> b c t h w")
    # pixel_values_ref = rearrange(pixel_values_ref.repeat(pixel_values_pose.shape[2], 1, 1, 1), "(b t) c h w -> b c t h w", b=1)
    # print("pixel_values_pose is", pixel_values_pose.shape, pixel_values_pose.unique())
    # print("pixel_values is", pixel_values.shape, pixel_values.unique())

    save_path = f"debug/my_data_{idx}.gif"
    # save_videos_grid(torch.cat([pixel_values_ref.cpu(), pixel_values.cpu(), pixel_values_pose.cpu()]) / 255., save_path)
    # save_videos_grid(torch.cat([pixel_values.cpu(), pixel_values_pose.cpu()]) / 255., save_path)
    save_videos_grid(torch.cat([pixel_values_pose.cpu()]) / 255., save_path)


    # break