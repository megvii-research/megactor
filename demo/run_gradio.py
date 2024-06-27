# Copyright 2024 Megvii inc.
#
# Copyright (2024) MegActor Authors.
#
# Megvii Inc. retain all intellectual property and proprietary rights in 
# and to this material, related documentation and any modifications thereto. 
# Any use, reproduction, disclosure or distribution of this material and related 
# documentation without an express license agreement from Megvii Inc. is strictly prohibited.
import argparse
import imageio
import os, datetime
import numpy as np
import gradio as gr
from PIL import Image
from subprocess import PIPE, run, Popen
import subprocess
import sys
import torch
sys.path.append('./')
from eval import eval

n_gpus = [False, ] * torch.cuda.device_count() 

os.makedirs("./demo/tmp", exist_ok=True)
savedir = f"demo/outputs"
os.makedirs(savedir, exist_ok=True)

model, image_processor, image_encoder = None, None, None

def animate(reference_image, motion_sequence, steps, guidance_scale, sample_l, sample_r, sample_s):
    global model, image_processor, image_encoder
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    animation_path = f"{savedir}/{time_str}.mp4"
    save_path = f"./demo/tmp/{time_str}.png"
    Image.fromarray(reference_image).save(save_path)
    for i in range(len(n_gpus)):
        if not n_gpus[i]:
            n_gpus[i] = True
            model, image_processor, image_encoder = eval(save_path, motion_sequence, 
                config=None,
                config_path="configs/inference/inference.yaml ",
                output_path=animation_path, 
                random_seed=42,
                guidance_scale=guidance_scale,
                weight_type=torch.float16, 
                num_steps=steps,
                device=torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"), 
                model=model,
                image_processor=image_processor,
                image_encoder=image_encoder,
                clip_image_type="background",
                concat_noise_image_type="origin",
                do_classifier_free_guidance=True,
                contour_preserve_generation=True,
                frame_sample_config=[sample_l, sample_r, sample_s]
                )
            n_gpus[i] = False
            break
    return animation_path

with gr.Blocks() as demo:

    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <a href="https://github.com/megvii-research/MegFaceAnimate" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
        </a>
        <div>
            <h1 >MegActor: Harness the Power of Raw Video for Vivid Portrait Animation</h1>
            <h5 style="margin: 0;">If you like our project, please give us a star ✨ on Github for the latest update.
                <a href='https://github.com/megvii-research/MegFaceAnimate'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
                <a href='https://arxiv.org/abs/2405.20851'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
                <a href='https://megactor.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
            </h5>
        </div>
        </div>
        <h4>Due to lackage of GPUs, driver video longer than 50 frames will be rounded. You may config sampling settings to your liking, we apologize for the inconvenience.</h4>
        """)
    animation = gr.Video(format="mp4", label="Animation Results", autoplay=True, height=512)
    
    with gr.Row():
        reference_image  = gr.Image(label="Source Image")
        motion_sequence  = gr.Video(format="mp4", label="Driver video")
        
        with gr.Column():
            sampling_steps      = gr.Number(label="Sampling steps", value=25, info="default: 25",  precision=0)
            guidance_scale      = gr.Number(label="Guidance scale", value=4.5, info="default: 4.5")
            sample_l      = gr.Number(label="Sample from n-th frame of the video", value=0,  precision=0)
            sample_r     = gr.Number(label="Sample to n-th frame of the video", value=-1, precision=0)
            sample_s     = gr.Number(label="Frame sampling step", value=1, precision=0)
            submit              = gr.Button("Animate")

    def read_video(video, size=512):
        return video
    
    def read_image(image, size=512):
        img = np.array(Image.fromarray(image).resize((size, size)))
        return img
        
    # when user uploads a new video
    motion_sequence.upload(
        read_video,
        motion_sequence,
        motion_sequence
    )
    # when `first_frame` is updated
    reference_image.upload(
        read_image,
        reference_image,
        reference_image
    )
    # when the `submit` button is clicked
    submit.click(
        animate,
        [reference_image, motion_sequence, sampling_steps, guidance_scale, sample_l, sample_r, sample_s], 
        animation
    )

    # Examples
    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            ["test_data/source/1.png", "test_data/driver/1.mp4",], 
            ["test_data/source/2.png", "test_data/driver/1.mp4",], 
            ["test_data/source/3.png", "test_data/driver/1.mp4",], 
        ],
        inputs=[reference_image, motion_sequence],
        outputs=animation,
    )

demo.queue(max_size=10)
demo.launch(share=True)
