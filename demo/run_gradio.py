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
import torch

n_gpus = [False, ] * torch.cuda.device_count() 

os.makedirs("./demo/tmp", exist_ok=True)
savedir = f"demo/outputs"
os.makedirs(savedir, exist_ok=True)

def animate(reference_image, motion_sequence, seed, steps, guidance_scale):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    animation_path = f"{savedir}/{time_str}.mp4"
    save_path = f"./demo/tmp/{time_str}.png"
    Image.fromarray(reference_image).save(save_path)
    for i in range(len(n_gpus)):
        if not n_gpus[i]:
            n_gpus[i] = True
            command = f"CUDA_VISIBLE_DEVICES={i} python3 eval.py --config configs/infer12_catnoise_warp08_power_vasa.yaml  --source {save_path} --driver {motion_sequence} --seed {seed} --num-step {steps} --guidance-scale {guidance_scale} --output-path {animation_path}"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            process.wait()
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
            <h5 style="margin: 0;">If you like our project, please give us a star ✨ on Github for the latest update.</h5>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;>
                <a href='https://github.com/megvii-research/MegFaceAnimate'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
            </div>
        </div>
        </div>
        """)
    animation = gr.Video(format="mp4", label="Animation Results", autoplay=True, height=512)
    
    with gr.Row():
        reference_image  = gr.Image(label="Source Image")
        motion_sequence  = gr.Video(format="mp4", label="Driver video")
        
        with gr.Column():
            random_seed         = gr.Textbox(label="Random seed", value=42, info="default: 42")
            sampling_steps      = gr.Textbox(label="Sampling steps", value=25, info="default: 25")
            guidance_scale      = gr.Textbox(label="Guidance scale", value=4.5, info="default: 4.5")
            submit              = gr.Button("Animate")

    def read_video(video, size=512):
        size = int(size)
        reader = imageio.get_reader(video)
        # fps = reader.get_meta_data()['fps']
        frames = []
        for img in reader:
            frames.append(np.array(Image.fromarray(img).resize((size, size))))
        return frames
    
    def read_image(image, size=512):
        img = np.Array(Image.fromarray(image).resize((size, size)))
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
        [reference_image, motion_sequence, random_seed, sampling_steps, guidance_scale], 
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