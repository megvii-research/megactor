import webdataset as wds
from webdataset import gopen, gopen_schemes
from tqdm import tqdm 
import imageio.v3 as iio
import imageio
import os
import cv2
import math
import numpy as np
import traceback
from PIL import Image

from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image
    )
import torch


def get_clip_frames(video_bytes:bytes) -> np.ndarray:
    frames = []
    fps = iio.immeta(video_bytes, plugin="pyav")["fps"] #有时候会报错拿不到这个属性
    fps = int(math.floor(fps+0.5))
    with iio.imopen(video_bytes, "r", plugin="pyav") as file:
        
        frames = file.read(index=...)  # np.array, [n_frames,h,w,c],  rgb, uint8
        frames = np.array(frames, dtype=frames.dtype) 
            
    return frames, fps


def get_diffusion_model(model_name:str):
    pipe = AutoPipelineForImage2Image.from_pretrained(model_name, variant="fp16", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe


def get_styled_image(pipe, image):
    if type(image) == np.ndarray:
        image = Image.fromarray(image)
    H, W = image.size
    image = image.resize((512, 512))
    generator = torch.Generator(device="cuda").manual_seed(0)
    # prompt = "Keep the opening of mouth consistent with the input image. A high-resolution anime-style portrait of a person with the same facial expression and pose as the original photo, emphasizing fine details and smooth lines, with vibrant colors and a clean, digital art aesthetic, inspired by popular manga and anime illustrations"
    # prompt = "A high-resolution anime-style portrait of a person with the same facial expression and pose as the original photo, emphasizing fine details and smooth lines, with vibrant colors"
    prompt = "A high-resolution anime-style portrait of a person with the same facial expression, same mouth movements and pose as the original photo, emphasizing fine details and smooth lines, with vibrant colors"
    image = pipe(prompt, "", image, num_inference_steps=30, generator=generator, strength=0.5).images[0]
    image = image.resize((H, W))
    image = np.array(image)
    return image


def worker(job):
    src_tarfilepath, dst_tarfilepath = job
    print(f"Processing {src_tarfilepath} -> {dst_tarfilepath}")
    err_msg = None
    model = get_diffusion_model("stabilityai/sdxl-turbo")
    try:
        # 如果字符串中有()的话，必须添加转义符号\
        src_tarfilepath = src_tarfilepath.replace("(","\(")
        src_tarfilepath = src_tarfilepath.replace(")","\)")

        dst_tarfilepath = dst_tarfilepath.replace("(","\(")
        dst_tarfilepath = dst_tarfilepath.replace(")","\)")
        dataset = wds.WebDataset(src_tarfilepath)
        dst_tarfilepath_name = dst_tarfilepath.split('/')
        with open(dst_tarfilepath, "wb") as wf:
            sink = wds.TarWriter(fileobj=wf,)
            for data in tqdm(dataset):
                key          = data["__key__"]
                video_bytes  = data["mp4"]
                try:
                    video_frames, video_fps = get_clip_frames(video_bytes)
                    styled_images = []
                    for idx, image in tqdm(enumerate(video_frames)):
                        styled_image = get_styled_image(model, image)
                        styled_images.append(styled_image)
                        
                    styled_images = np.stack(styled_images, axis=0)
                    bytes_styled_images = iio.imwrite("<bytes>", styled_images, extension='.mp4', plugin="pyav", codec="h264", fps=video_fps)
                    data.update({
                        "mp4_styled"              : bytes_styled_images, 
                    })
                    sink.write(data)
                except Exception as e:
                    traceback.print_exc()
            sink.close()
        dataset.close()
    except Exception as e:
        print(e)
        err_msg = e
    return src_tarfilepath, err_msg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tarfile', type=str, default="")
    parser.add_argument('--dstfile', type=str, default="")
    args = parser.parse_args()    
    
    jobs = [(args.tarfile, args.dstfile)]
    for job in tqdm(jobs):
        src_tarfilepath, err_msg = worker(job)
        print(src_tarfilepath, err_msg)
        