import webdataset as wds
from webdataset import gopen, gopen_schemes
from tqdm import tqdm 
import imageio.v3 as iio
import imageio
import os
import cv2
import math
import numpy as np
import pickle
import scipy
import traceback
from PIL import Image
import copy
import torch

from l2cs import Pipeline

def get_face_bbox(landmark, H, W):
    offset_x, offset_y = 0.2, 0.2
    x_min, x_max = np.min(landmark[:, 0]), np.max(landmark[:, 0])
    y_min, y_max = np.min(landmark[:, 1]), np.max(landmark[:, 1]) 
    x_min = max(x_min - (x_max - x_min) * offset_x, 0)
    x_max = min(x_max + (x_max - x_min) * offset_x, W)
    y_min = max(y_min - (y_max - y_min) * offset_y, 0)
    y_max = min(y_max + (y_max - y_min) * offset_y, H)
    x_min, x_max, y_min, y_max = (int(x) for x in (x_min, x_max, y_min, y_max))
    return x_min, x_max, y_min, y_max


def get_clip_frames(video_bytes:bytes, dwpose_result_b:bytes, dwpose_score_b:bytes):
    dwpose_results = pickle.loads(dwpose_result_b)
    dwpose_scores = pickle.loads(dwpose_score_b)
    with iio.imopen(video_bytes, "r", plugin="pyav") as file:
        frames = file.read(index=...)  # np.array, [n_frames,h,w,c],  rgb, uint8

    return frames, dwpose_results, dwpose_scores


def get_gaze(gaze_pipeline, frames, dwpose_results):
    landmarks = np.array(dwpose_results, dtype="float32")[:, :, 24:92:, :]
    H, W = frames.shape[1:3]
    gaze_frames = list()
    thetas = list()
    for idx, frame in enumerate(frames):
        x_min, x_max, y_min, y_max = get_face_bbox(landmarks[idx][0], H, W)
        face = frame[y_min:y_max, x_min:x_max]
        result = gaze_pipeline.step(face)
        
        # from l2cs import render
        # for visualize 
        # face_render = render(face.copy(), result)
        # frame[y_min:y_max, x_min:x_max] = face_render

        gaze_frames.append(frame)
        pitch = result.pitch[0]
        yaw = result.yaw[0]
        tanh_theta = np.sin(pitch) * np.cos(yaw) / np.sin(yaw)
        theta = np.arctan(tanh_theta) / np.pi * 90
        thetas.append(theta)
    
    thetas = np.array(thetas)
    thetas = scipy.ndimage.maximum_filter1d(thetas, size=3)
    gaze_frames = np.array(gaze_frames)
    return gaze_frames, np.var(thetas)
        

def worker(src_tarfilepath, dst_tarfilepath, gaze_pipeline):
    
    print(f"Processing {src_tarfilepath} -> {dst_tarfilepath}")
    err_msg = None
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
                dwpose_result = data["dwpose_result.pyd"]
                dwpose_score = data["dwpose_score.pyd"]
                try:
                    frames, dwpose_results, dwpose_scores = get_clip_frames(video_bytes, dwpose_result, dwpose_score)
                    
                    length = 32
                    gaze_fames, var = get_gaze(gaze_pipeline, frames[::10][:length], dwpose_results[::10][:length])
                    if var > 200:
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
    # import debugpy
    # attach_ip='100.96.201.198'; attach_port=5678
    # debugpy.listen((attach_ip, attach_port))
    # print(f'{attach_ip}:{attach_port} is Listening')
    # debugpy.wait_for_client()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tarfile', type=str, default="")
    parser.add_argument('--dstfile', type=str, default="")
    parser.add_argument('--gaze_model', type=str, default="L2CS-Net/models/L2CSNet_gaze360.pkl")
    args = parser.parse_args()    
    
    gaze_pipeline = Pipeline(
    weights=args.gaze_model,
    arch='ResNet50',
    include_detector=True,
    device=torch.device('cuda:0') if torch.cuda.is_available() else 'cpu' # or 'gpu'
    )
    # jobs = [("s3://public-datasets/Datasets/Videos/processed/VFHQ_webdataset_20240506_dwpose_facebbox_facefusion-HQsource/group158.tar", "s3://lhd/Datasets/Videos/processed/VFHQ_webdataset_20240520_dwpose_facebbox_facefusion-HQsource_filter-eye-dynamic/test.tar")]
    print(gaze_pipeline)
    
    worker(src_tarfilepath=args.tarfile,
           dst_tarfilepath=args.dstfile,
           gaze_pipeline=gaze_pipeline
           )
    
        