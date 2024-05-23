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
import importlib

from pathlib import Path
from megfile import smart_open
import io
from animatediff.utils.util import save_videos_grid, pad_image
from animatediff import preprocess
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
import facer
from libs.controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor
from train import eval_model
from accelerate import Accelerator

def main(args):

    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    accelerator = Accelerator()
    config = OmegaConf.load(args.config)
    # Initialize distributed training
    device = torch.device(f"cuda:{args.rank}")
    weight_type = torch.float16

    if config.savename is None:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = f"samples/{Path(args.config).stem}-{time_str}"
    else:
        savedir = f"samples/{config.savename}"

    os.makedirs(savedir, exist_ok=True)
    OmegaConf.save(config, f"{savedir}/config.yaml")

    # num_actual_inference_steps = config.get(
    #     "num_actual_inference_steps", config.steps)

    do_classifier_free_guidance = config.get(
        "do_classifier_free_guidance", True
    )
    condition_config = config.get(
        "condition_config", {}
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
    MagicAnimate = getattr(importlib.import_module(f'animatediff.magic_animate.{model_type}.animate'), 'MagicAnimate')
    pipeline = MagicAnimate(config=config,
                            train_batch_size=1,
                            device=device,
                            unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs))
    pipeline.to(device, dtype=weight_type)
    pipeline.eval()

    validation_data = config['validation_data']
    dwpose_model = DenseDWposePredictor(device, resolution=validation_data['sample_size'])
    random_seeds = config.get("valid_seed", [-1])

    eval_model(validation_data, 
                pipeline, 
                0, # local_rank
                weight_type, 
                config['context'], 
                savedir, 
                0,  # global_step, used to mark ckpt steps
                accelerator,
                random_seeds,
                dwpose_model,
                condition_config,
                config['pretrained_model_path'],
                do_classifier_free_guidance=do_classifier_free_guidance,
                crop_face_center=crop_face_center, 
                switch_control_to_source=switch_control_to_source,
    )


def run(args):
    main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--rank", type=int, default=0, required=False)

    args = parser.parse_args()
    run(args)
