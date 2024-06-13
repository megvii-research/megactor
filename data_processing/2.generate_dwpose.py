import webdataset as wds
from tqdm import tqdm 
import json
import torch
import imageio.v3 as iio
import numpy as np
from PIL import Image
import cv2
import traceback
import math

import sys
sys.path.append('../')

from controlnet_resource.dwpose import SimpleDWposeDetector

def get_clip_frames(video_bytes:bytes) -> torch.Tensor:
    frames = []
    fps = iio.immeta(video_bytes, plugin="pyav")["fps"]
    fps = int(math.floor(fps+0.5))
    with iio.imopen(video_bytes, "r", plugin="pyav") as file:
        
        frames = file.read(index=...)  # np.array, [n_frames,h,w,c],  rgb, uint8
        frames = np.array(frames, dtype=frames.dtype) 
            
    return frames, fps


def get_dwpose_output(dwpose_model, frames_np):
    num_frames, *_ = frames_np.shape

    frames_dwpose_result = []
    frames_dwpose_score = []
    frames_face_bbox = []
    frames_exist_face_index = []
    
    for idx, frame_id in tqdm(enumerate(range(num_frames))):
        frame_np = frames_np[frame_id]
        frame_dwpose_result, frame_dwpose_score, face_bbox = dwpose_model(frame_np) # 输入RGB而不是BGR
        
        frames_dwpose_result.append(frame_dwpose_result.astype(np.float32))
        frames_dwpose_score.append(frame_dwpose_score.astype(np.float32))
        frames_face_bbox.append(face_bbox.astype(np.float32))
        frames_exist_face_index.append(idx)

    frames_dwpose_result   = np.array(frames_dwpose_result)
    frames_dwpose_score = np.array(frames_dwpose_score)
    frames_face_bbox    = np.array(frames_face_bbox)
    frames_exist_face_index    = np.array(frames_exist_face_index)

    return {
        "dwpose_result"          : frames_dwpose_result,
        "dwpose_score"           : frames_dwpose_score,
        "faces_bbox"             : frames_face_bbox,
        "exist_face_frame_index" : frames_exist_face_index,
    }

def worker(job):
    src_tarfilepath, dst_tarfilepath = job
    dwpose_model = SimpleDWposeDetector()
    
    err_msg = None
    try:
        # 如果字符串中有()的话，必须添加转义符号\
        src_tarfilepath = src_tarfilepath.replace("(","\(")
        src_tarfilepath = src_tarfilepath.replace(")","\)")

        dst_tarfilepath = dst_tarfilepath.replace("(","\(")
        dst_tarfilepath = dst_tarfilepath.replace(")","\)")
        print(f"Processing {src_tarfilepath} -> {dst_tarfilepath}")
        
        dataset = wds.WebDataset(src_tarfilepath)
        with open(dst_tarfilepath, "wb") as wf:
            sink = wds.TarWriter(fileobj=wf,)
            for data in tqdm(dataset):
                key          = data["__key__"]
                video_bytes  = data["mp4"]
                try:
                    video_frames, video_fps = get_clip_frames(video_bytes)
                    frames_face_info = get_dwpose_output(dwpose_model, video_frames)
                    
                    sink.write({
                    "__key__": key,
                    "mp4"    : video_bytes,
                    "dwpose_result.pyd"           : frames_face_info.get("dwpose_result",          []),
                    "dwpose_score.pyd"            : frames_face_info.get("dwpose_score",           []), 
                    "faces_bbox.pyd"              : frames_face_info.get("faces_bbox",             []),
                    "exist_face_frame_index.pyd"  : frames_face_info.get("exist_face_frame_index", []),
                    })
                except Exception as e:
                    traceback.print_exc()
                    frames_face_info = {}
            sink.close()
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