from torch.utils.data import IterableDataset
import random
import cv2
# import torch.nn.functional as F
import torch
import numpy as np
from einops import rearrange, repeat
import math
import webdataset as wds
from webdataset import gopen, gopen_schemes
import imageio.v3 as iio
import json
import traceback
from typing import List
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torchaudio
import os
from animate.utils.util import get_patch, crop_and_resize_tensor_with_face_rects, crop_move_face, crop_and_resize_tensor_small_faces, get_patch_div
from animate.utils.util import save_videos_grid, pad_image, generate_random_params, apply_transforms, crop_and_resize_tensor_flex
import facer
from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor
from controlnet_aux_lib.dwpose.draw_dwpose import get_dwpose_body
from PIL import Image
import refile
import pickle
from audio_augmentations import RandomApply, PolarityInversion, Noise, Gain, Delay, HighLowPass, PitchShift, Reverb, Compose

sr = 16000
audio_transforms_list = [
    RandomApply([PolarityInversion()], p=0.2),
    RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
    RandomApply([Gain()], p=0.2),
    RandomApply([HighLowPass(sample_rate=sr)], p=0.2),
    RandomApply([PitchShift(
        n_samples=sr * 5,
        sample_rate=sr,
        pitch_cents_min=-500, pitch_cents_max=500
    )], p=0.75),
    RandomApply([Reverb(sample_rate=sr)], p=0.2)
]

audio_transforms = Compose(transforms=audio_transforms_list)

class VideoTransforms:
    def __init__(self, p_grayscale=0.1, p_color_jitter=0.25, p_flip=0.5):
        self.p_grayscale = p_grayscale
        self.p_color_jitter = p_color_jitter

    def __call__(self, x):
        # Determine if transforms should be applied
        do_grayscale = random.random() < self.p_grayscale
        do_color_jitter = random.random() < self.p_color_jitter

        x /= 255.
        # Create a ColorJitter transform with random parameters
        if do_color_jitter:
            brightness = random.uniform(0.6, 1.4)
            contrast = random.uniform(0.6, 1.4)
            saturation = random.uniform(0.6, 1.4)
            hue = random.uniform(-0.3, 0.3)
            color_jitter = transforms.ColorJitter(
                brightness=(brightness, brightness),
                contrast=(contrast, contrast),
                saturation=(saturation, saturation),
                hue=(hue, hue)
            )

        # Apply the same transform to all frames in the batch
        for i in range(x.size(0)):  # Iterate over batch
            if do_grayscale:
                x[i] = transforms.Grayscale(num_output_channels=3)(x[i])
            if do_color_jitter:
                x[i] = color_jitter(x[i])

        return x * 255.


def get_clip_frames(self, data, ref_index=0) -> torch.Tensor:
    video = data[self.main_key]
    other_video_bytes = []
    for name in self.other_frames:
        if name in data:
            other_video_bytes.append(data[name])

    video_byte_sequence = [video] + other_video_bytes
    decoded_sequences = list()
    ref_image = None
    clip_indices = None
    for idx, video_bytes in enumerate(video_byte_sequence):
        if video_bytes is None:
            frames = None
        else:
            try:
                file = iio.imopen(video_bytes, "r", plugin="pyav")
            except:
                return None
            frames = file.read(index=...)  # np.array, [n_frames,h,w,c],  rgb, uint8
            frames_real = []
            if idx == ref_index:
                ref_image = torch.tensor(random.choice(frames)).permute(2, 0, 1)[None, ...]
            for frame in frames:
                if frame.mean() > self.luma_thresh:
                    frames_real.append(frame)
            frames = np.array(frames_real, dtype=frames.dtype) 
                
            n_frames = frames.shape[0]
            if n_frames < self.video_length:
                return None
            else:
                if clip_indices is None:
                    clip_indices, frame_stride = self.get_random_clip_indices(n_frames)
                
                frames = frames[clip_indices, ...]
                frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
        if frames is not None:
            decoded_sequences.append(frames)

    if ref_image is None:
        return None

    if len(decoded_sequences) > 1 and np.random.rand() < self.use_swap_rate:
        swapped = random.choice(decoded_sequences[1:])
    else:
        swapped = decoded_sequences[0]
    return dict(
        ref_image=ref_image, 
        swapped=swapped,
        frame=decoded_sequences[0]
    )

def get_packed_frames(self, data, ref_index=0) -> torch.Tensor:
    video = data[self.main_key]
    other_video_bytes = []
    for name in self.other_frames:
        if name in data:
            other_video_bytes.append(data[name])

    video_byte_sequence = [video] + other_video_bytes

    dwpose_result_b = data["dwpose_result.pyd"]
    dwpose_score_b = data["dwpose_score.pyd"]
    faces_bbox_b = data['faces_bbox.pyd']
    audio_frames = pickle.loads(data["audio_frames.pyd"])
    audio_signal = pickle.loads(data["audio.pyd"])
    start_time = int(data["start_time.str"])
    sample_rate = int(data["sample_rate.str"])
    fps = int(data["fps.str"])
    end_time = start_time + (len(audio_frames) / fps)
    exist_face_frame_index_b = data['exist_face_frame_index.pyd']

    dwpose_results = pickle.loads(dwpose_result_b)
    dwpose_scores = pickle.loads(dwpose_score_b)
    faces_bbox = pickle.loads(faces_bbox_b)
    exist_face_frame_index = pickle.loads(exist_face_frame_index_b)

    real_select_idx = []
    cur_idx_list = []
    ref_select_idx = []
    for i, frame_is_valid in enumerate(exist_face_frame_index):
        if frame_is_valid:
            cur_idx_list.append(i)
        else:
            real_select_idx.append(cur_idx_list)
            ref_select_idx.extend(cur_idx_list)
            cur_idx_list = []
    if len(cur_idx_list) != 0:
        real_select_idx.append(cur_idx_list)
        ref_select_idx.extend(cur_idx_list)
    max_len = 0
    select_list = 0
    real_select_idx = [idx for idx in real_select_idx if len(idx) > self.frame_stride * self.video_length]
    if len(real_select_idx) == 0:
        return None
    real_select_idx = real_select_idx[random.choice(list(range(len(real_select_idx))))]
    ref_select_idx = random.choice(ref_select_idx)

    start_time = start_time + real_select_idx[0] / fps

    decoded_sequences = list()
    ref_image = None
    clip_indices = None
    for idx, video_bytes in enumerate(video_byte_sequence):
        if video_bytes is None:
            frames = None
        else:
            try:
                file = iio.imopen(video_bytes, "r", plugin="pyav")
            except:
                return None
            frames = file.read(index=...) # np.array, [n_frames,h,w,c],  rgb, uint8
            frames_real = []
            if idx == ref_index:
                ref_image = torch.tensor(frames[ref_select_idx]).permute(2, 0, 1)[None, ...]
            frames = frames[real_select_idx]
            for frame in frames:
                if frame.mean() > self.luma_thresh:
                    frames_real.append(frame)
            frames = np.array(frames_real, dtype=frames.dtype) 
                
            n_frames = frames.shape[0]
            if n_frames < self.video_length:
                return None
            else:
                if clip_indices is None:
                    clip_indices, frame_stride = self.get_random_clip_indices(n_frames)
                
                frames = frames[clip_indices, ...]
                frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
        if frames is not None:
            decoded_sequences.append(frames)

    start_time = start_time + clip_indices[0] / fps
    end_time = start_time + (clip_indices[-1] - clip_indices[0]) / fps

    if len(decoded_sequences) > 1 and np.random.rand() < self.use_swap_rate:
        swapped = random.choice(decoded_sequences[1:])
    else:
        swapped = decoded_sequences[0]
    return dict(
        ref_image=ref_image, 
        swapped=swapped,
        frames=decoded_sequences[0],
        start_time=start_time,
        end_time=end_time,
        candidates=np.array([[dwpose_results[i] for i in real_select_idx][j] for j in clip_indices]),
        subsets=np.array([[dwpose_scores[i] for i in real_select_idx][j] for j in clip_indices]),
        faces_bbox=np.array([faces_bbox[ref_select_idx]] + [[faces_bbox[i] for i in real_select_idx][j] for j in clip_indices]),
        # audio_frames=audio_frames,
        sample_rate=sample_rate,
        start_idx=real_select_idx[0] + clip_indices[0],
        audio_signal=audio_signal,
        fps=fps,
        # exist_face_frame_index=exist_face_frame_index[real_select_idx][clip_indices]
    )


def get_packed_frames(self, data, ref_index=0) -> torch.Tensor:
    video = data[self.main_key]
    other_video_bytes = []
    for name in self.other_frames:
        if name in data:
            other_video_bytes.append(data[name])

    video_byte_sequence = [video] + other_video_bytes

    dwpose_result_b = data["dwpose_result.pyd"]
    dwpose_score_b = data["dwpose_score.pyd"]
    faces_bbox_b = data['faces_bbox.pyd']
    audio_frames = pickle.loads(data["audio_frames.pyd"])
    audio_signal = pickle.loads(data["audio.pyd"])
    start_time = int(data["start_time.str"])
    sample_rate = int(data["sample_rate.str"])
    fps = int(data["fps.str"])
    end_time = start_time + (len(audio_frames) / fps)
    exist_face_frame_index_b = data['exist_face_frame_index.pyd']

    dwpose_results = pickle.loads(dwpose_result_b)
    dwpose_scores = pickle.loads(dwpose_score_b)
    faces_bbox = pickle.loads(faces_bbox_b)
    exist_face_frame_index = pickle.loads(exist_face_frame_index_b)

    real_select_idx = []
    cur_idx_list = []
    ref_select_idx = []
    for i, frame_is_valid in enumerate(exist_face_frame_index):
        if frame_is_valid:
            cur_idx_list.append(i)
        else:
            real_select_idx.append(cur_idx_list)
            ref_select_idx.extend(cur_idx_list)
            cur_idx_list = []
    if len(cur_idx_list) != 0:
        real_select_idx.append(cur_idx_list)
        ref_select_idx.extend(cur_idx_list)
    max_len = 0
    select_list = 0
    real_select_idx = [idx for idx in real_select_idx if len(idx) > self.frame_stride * self.video_length]
    if len(real_select_idx) == 0:
        return None
    real_select_idx = real_select_idx[random.choice(list(range(len(real_select_idx))))]
    ref_select_idx = random.choice(ref_select_idx)

    start_time = start_time + real_select_idx[0] / fps

    decoded_sequences = list()
    ref_image = None
    clip_indices = None
    for idx, video_bytes in enumerate(video_byte_sequence):
        if video_bytes is None:
            frames = None
        else:
            try:
                file = iio.imopen(video_bytes, "r", plugin="pyav")
            except:
                return None
            frames = file.read(index=...) # np.array, [n_frames,h,w,c],  rgb, uint8
            frames_real = []
            if idx == ref_index:
                ref_image = torch.tensor(frames[ref_select_idx]).permute(2, 0, 1)[None, ...]
            frames = frames[real_select_idx]
            for frame in frames:
                if frame.mean() > self.luma_thresh:
                    frames_real.append(frame)
            frames = np.array(frames_real, dtype=frames.dtype) 
                
            n_frames = frames.shape[0]
            if n_frames < self.video_length:
                return None
            else:
                if clip_indices is None:
                    clip_indices, frame_stride = self.get_random_clip_indices(n_frames)
                
                print(clip_indices)
                frames = frames[clip_indices, ...]
                frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
        if frames is not None:
            decoded_sequences.append(frames)

    start_time = start_time + clip_indices[0] / fps
    end_time = start_time + (clip_indices[-1] - clip_indices[0]) / fps

    if len(decoded_sequences) > 1 and np.random.rand() < self.use_swap_rate:
        swapped = random.choice(decoded_sequences[1:])
    else:
        swapped = decoded_sequences[0]
    return dict(
        ref_image=ref_image, 
        swapped=swapped,
        frames=decoded_sequences[0],
        start_time=start_time,
        end_time=end_time,
        candidates=np.array([[dwpose_results[i] for i in real_select_idx][j] for j in clip_indices]),
        subsets=np.array([[dwpose_scores[i] for i in real_select_idx][j] for j in clip_indices]),
        faces_bbox=np.array([faces_bbox[ref_select_idx]] + [[faces_bbox[i] for i in real_select_idx][j] for j in clip_indices]),
        # audio_frames=audio_frames,
        sample_rate=sample_rate,
        start_idx=real_select_idx[0] + clip_indices[0],
        audio_signal=audio_signal,
        fps=fps,
        # exist_face_frame_index=exist_face_frame_index[real_select_idx][clip_indices]
    )


def get_compact_frames(self, data, ref_index=0) -> torch.Tensor:
    video = data[self.main_key]
    other_video_bytes = []
    for name in self.other_frames:
        if name in data:
            other_video_bytes.append(data[name])

    video_byte_sequence = [video] + other_video_bytes
    use_xpose = "dwpose_result.pyd" not in data and "xpose_result.pyd" in data

    dwpose_result_b = data["dwpose_result.pyd"] if "dwpose_result.pyd" in data else data["xpose_result.pyd"]
    dwpose_score_b = data["dwpose_score.pyd"] if "dwpose_score.pyd" in data else data["xpose_score.pyd"]
    faces_bbox_b = data['faces_bbox.pyd'] if "faces_bbox.pyd" in data else data["xpose_faces_bbox.pyd"]
    audio_frames = pickle.loads(data["audio_frames.pyd"]).reshape(-1)
    sample_rate = float(data["sample_rate.str"])
    fps = float(data["fps.str"])
    start_time = 0
    end_time = start_time + (len(audio_frames) / fps)
    exist_face_frame_index_b = data['exist_face_frame_index.pyd']

    dwpose_results = pickle.loads(dwpose_result_b)
    dwpose_scores = pickle.loads(dwpose_score_b)
    faces_bbox = pickle.loads(faces_bbox_b)
    exist_face_frame_index = pickle.loads(exist_face_frame_index_b)

    real_select_idx = []
    cur_idx_list = []
    ref_select_idx = []
    for i, frame_is_valid in enumerate(exist_face_frame_index):
        if frame_is_valid:
            cur_idx_list.append(i)
        else:
            real_select_idx.append(cur_idx_list)
            ref_select_idx.extend(cur_idx_list)
            cur_idx_list = []
    if len(cur_idx_list) != 0:
        real_select_idx.append(cur_idx_list)
        ref_select_idx.extend(cur_idx_list)
    max_len = 0
    select_list = 0
    real_select_idx = [idx for idx in real_select_idx if len(idx) > self.frame_stride * self.video_length]
    if len(real_select_idx) == 0:
        return None
    real_select_idx = real_select_idx[random.choice(list(range(len(real_select_idx))))]
    ref_select_idx = random.choice(ref_select_idx)

    start_time = start_time + real_select_idx[0] / fps

    decoded_sequences = list()
    ref_image = None
    clip_indices = None
    frame_stride= None
    for idx, video_bytes in enumerate(video_byte_sequence):
        if video_bytes is None:
            frames = None
        else:
            try:
                file = iio.imopen(video_bytes, "r", plugin="pyav")
            except:
                return None
            frames = file.read(index=...) # np.array, [n_frames,h,w,c],  rgb, uint8
            frames_real = []
            if idx == ref_index:
                ref_image = torch.tensor(frames[ref_select_idx]).permute(2, 0, 1)[None, ...]
            frames = frames[real_select_idx]
            for frame in frames:
                if frame.mean() > self.luma_thresh:
                    frames_real.append(frame)
            frames = np.array(frames_real, dtype=frames.dtype) 
                
            n_frames = frames.shape[0]
            if n_frames < self.video_length:
                return None
            else:
                if clip_indices is None:
                    clip_indices, frame_stride = self.get_random_clip_indices(n_frames)
                
                frames = frames[clip_indices, ...]
                frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
        if frames is not None:
            decoded_sequences.append(frames)

    start_time = start_time + clip_indices[0] / fps
    end_time = start_time + (clip_indices[-1] - clip_indices[0]) / fps

    if len(decoded_sequences) > 1 and np.random.rand() < self.use_swap_rate:
        swapped = random.choice(decoded_sequences[1:])
    else:
        swapped = decoded_sequences[0]
    return dict(
        ref_image=ref_image, 
        swapped=swapped,
        frames=decoded_sequences[0],
        start_time=start_time,
        end_time=end_time,
        candidates=np.array([[dwpose_results[i] for i in real_select_idx][j] for j in clip_indices]),
        subsets=np.array([[dwpose_scores[i] for i in real_select_idx][j] for j in clip_indices]),
        faces_bbox=np.array([faces_bbox[ref_select_idx]] + [[faces_bbox[i] for i in real_select_idx][j] for j in clip_indices]),
        # audio_frames=audio_frames,
        sample_rate=sample_rate,
        frame_stride=frame_stride,
        start_idx=real_select_idx[0] + clip_indices[0],
        audio_signal=audio_frames,
        fps=fps,
        use_xpose=use_xpose
        # exist_face_frame_index=exist_face_frame_index[real_select_idx][clip_indices]
    )


def save_frames(self, data, ref_index=0) -> torch.Tensor:
    video = data[self.main_key]
    other_video_bytes = []
    for name in self.other_frames:
        if name in data:
            other_video_bytes.append(data[name])

    video_byte_sequence = [video] + other_video_bytes
    use_xpose = "dwpose_result.pyd" not in data and "xpose_result.pyd" in data

    dwpose_result_b = data["dwpose_result.pyd"] if "dwpose_result.pyd" in data else data["xpose_result.pyd"]
    dwpose_score_b = data["dwpose_score.pyd"] if "dwpose_score.pyd" in data else data["xpose_score.pyd"]
    faces_bbox_b = data['faces_bbox.pyd'] if "faces_bbox.pyd" in data else data["xpose_faces_bbox.pyd"]
    audio_frames = pickle.loads(data["audio_frames.pyd"]).reshape(-1)
    sample_rate = float(data["sample_rate.str"])
    fps = float(data["fps.str"])
    start_time = 0
    end_time = start_time + (len(audio_frames) / fps)
    exist_face_frame_index_b = data['exist_face_frame_index.pyd']

    dwpose_results = pickle.loads(dwpose_result_b)
    dwpose_scores = pickle.loads(dwpose_score_b)
    faces_bbox = pickle.loads(faces_bbox_b)
    exist_face_frame_index = pickle.loads(exist_face_frame_index_b)

    real_select_idx = []
    cur_idx_list = []
    ref_select_idx = []
    for i, frame_is_valid in enumerate(exist_face_frame_index):
        if frame_is_valid:
            cur_idx_list.append(i)
        else:
            real_select_idx.append(cur_idx_list)
            ref_select_idx.extend(cur_idx_list)
            cur_idx_list = []
    if len(cur_idx_list) != 0:
        real_select_idx.append(cur_idx_list)
        ref_select_idx.extend(cur_idx_list)
    max_len = 0
    select_list = 0
    real_select_idx = [idx for idx in real_select_idx if len(idx) > self.frame_stride * self.video_length and len(idx) > self.frame_stride * self.video_length * 2]
    if len(real_select_idx) == 0:
        return None
    # real_select_idx = real_select_idx[random.choice(list(range(len(real_select_idx))))]
    real_select_idx = sorted(real_select_idx, key=lambda x: len(x), reverse=True)[0]
    # print(real_select_idx)
    ref_select_idx = random.choice(ref_select_idx)

    start_time = start_time + real_select_idx[0] / fps

    decoded_sequences = list()
    ref_image = None
    clip_indices = None
    frame_stride= None
    for idx, video_bytes in enumerate(video_byte_sequence):
        if video_bytes is None:
            frames = None
        else:
            try:
                file = iio.imopen(video_bytes, "r", plugin="pyav")
            except:
                return None
            frames = file.read(index=...) # np.array, [n_frames,h,w,c],  rgb, uint8
            frames_real = []
            if idx == ref_index:
                ref_image = torch.tensor(frames[ref_select_idx]).permute(2, 0, 1)[None, ...]
            frames = frames[real_select_idx]
            for frame in frames:
                if frame.mean() > self.luma_thresh:
                    frames_real.append(frame)
            frames = np.array(frames_real, dtype=frames.dtype) 
                
            n_frames = frames.shape[0]
            if n_frames < self.video_length:
                return None
            else:
                if clip_indices is None:
                    clip_indices, frame_stride = self.get_random_clip_indices(n_frames)
                
                frames = frames[clip_indices, ...]
                # frames = frames[clip_indices, ...]
                frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
        if frames is not None:
            decoded_sequences.append(frames)

    start_time = start_time + clip_indices[0] / fps
    end_time = start_time + (clip_indices[-1] - clip_indices[0]) / fps

    if len(decoded_sequences) > 1 and np.random.rand() < self.use_swap_rate:
        swapped = random.choice(decoded_sequences[1:])
    else:
        swapped = decoded_sequences[0]
    return dict(
        ref_image=ref_image, 
        swapped=swapped,
        frames=decoded_sequences[0],
        start_time=start_time,
        end_time=end_time,
        candidates=np.array([[dwpose_results[i] for i in real_select_idx][j] for j in clip_indices]),
        subsets=np.array([[dwpose_scores[i] for i in real_select_idx][j] for j in clip_indices]),
        faces_bbox=np.array([faces_bbox[ref_select_idx]] + [[faces_bbox[i] for i in real_select_idx][j] for j in clip_indices]),
        # audio_frames=audio_frames,
        sample_rate=sample_rate,
        frame_stride=frame_stride,
        start_idx=real_select_idx[0] + clip_indices[0],
        audio_signal=audio_frames,
        fps=fps,
        use_xpose=use_xpose
        # exist_face_frame_index=exist_face_frame_index[real_select_idx][clip_indices]
    )

def crop_audio_area(self, ref_image, frames, swapped, start_time, end_time, candidates, subsets, faces_bbox, sample_rate, audio_signal, fps, start_idx, frame_stride, eval=False, use_xpose=False):
    # 首先裁剪出合适尺寸的人脸，并resize到期望的大小
    H, W = frames.shape[2:]
    l, t, r, b = W, H, 0, 0
    # for face_bbox, frame_is_valid in zip(faces_bbox, exist_face_frame_index):
    #     if not frame_is_valid:
    #         continue
    for face_bbox in faces_bbox:
        if use_xpose:
            face_bbox = face_bbox[0]
        x0, y0, x1, y1 = face_bbox
        l = min(x0, l)
        t = min(y0, t)
        r = max(r, x1)
        b = max(b, y1)
    face_mask = torch.zeros_like(frames)[:, :1, :, :]
    face_mask[:, :, int(t): int(b), int(l): int(r)] = 1.0

    # Get center and rect we need to use
    w, h = (r - l), (b - t)
    x_c, y_c = (l + r) / 2, (t + b) / 2
    expand_dis = max(w, h)
    left, right = max(x_c - expand_dis * self.left_scale * (0.8 + 0.4 * np.random.rand()), 0), min(x_c + expand_dis * self.right_scale * (0.8 + 0.4 * np.random.rand()), W)
    top, bottom = max(y_c - expand_dis * self.top_scale * (0.8 + 0.4 * np.random.rand()), 0), min(y_c + expand_dis * self.bottom_scale * (0.8 + 0.4 * np.random.rand()), H)
    
    # Get new center and new rect
    x_c, y_c = (left + right) / 2, (bottom + top) / 2
    distance_to_edge = min(x_c - left, right - x_c, y_c - top, bottom - y_c)    
    left = x_c - distance_to_edge
    right = x_c + distance_to_edge
    top = y_c - distance_to_edge
    bottom = y_c + distance_to_edge
    frames = frames[:, :, int(top):int(bottom), int(left):int(right)].float()
    target_height, target_width = self.resolution
    frames = videos = torch.nn.functional.interpolate(frames, size=(target_height, target_width), mode='bilinear', align_corners=False)

    swapped = swapped[:, :, int(top):int(bottom), int(left):int(right)].float()
    swapped = torch.nn.functional.interpolate(swapped, size=(target_height, target_width), mode='bilinear', align_corners=False)
    ref_image = ref_image[:, :, int(top):int(bottom), int(left):int(right)].float()
    ref_image = torch.nn.functional.interpolate(ref_image, size=(target_height, target_width), mode='bilinear', align_corners=False)
    
    # 再裁剪出025 的嘴巴和眼睛增强的人脸
    frames_eye_mouth = torch.zeros_like(frames)
    frames_eye = torch.zeros_like(frames)
    frames = swapped
    eps = 0.01

    # 首先将变换后的landmark进行resize
    for i in range(len(candidates)):
        for j in range(candidates.shape[2]):
            x, y = candidates[i, 0, j]
            x, y = (x - left) / (right - left), (y - top) / (bottom - top)
            x, y = x * target_width, y * target_height
            candidates[i, 0, j] = x, y
    
    patch_indices = [[1e6, 0, 1e6, 0], ] * 4
    for i in range(len(candidates)):
        xmin, xmax, ymin, ymax = patch_indices[0]
        xs, ys = [], []
        point_indexs = [41, 42, 43, 44, 45, 60, 61, 62, 63, 64, 65] if not use_xpose else [22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47]  # 左眼睛和左眉毛
        for j in point_indexs: # left eyes
            if subsets[i, 0, j] < 0.3:
                continue
            x, y = candidates[i, 0, j]
            if x < eps or y < eps:
                continue
            xs.append(x)
            ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) != 0 and len(ys) != 0:
            # x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            # left, right, top, bottom = get_patch(x0, y0, x1, y1, target_height, target_width)
            left, right, top, bottom = get_patch_div(xs.mean(), ys.mean(), self.resolution[0], self.resolution[1], random.randint(8, 16))
            xmin = min(xmin, left)
            xmax = max(xmax, right)
            ymin = min(ymin, top)
            ymax = max(ymax, bottom)
            patch_indices[0] = [xmin, xmax, ymin, ymax]

        xs, ys = [], []
        point_indexs = [46, 47, 48, 49, 50, 66, 67, 68, 69, 70, 71] if not use_xpose else [17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41] # 右眼睛和右眉毛
        xmin, xmax, ymin, ymax = patch_indices[1]
        for j in point_indexs: # left eyes
            if subsets[i, 0, j] < 0.3:
                continue
            x, y = candidates[i, 0, j]
            if x < eps or y < eps:
                continue
            xs.append(x)
            ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) != 0 and len(ys) != 0:
            # x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            left, right, top, bottom = get_patch_div(xs.mean(), ys.mean(), self.resolution[0], self.resolution[1], random.randint(8, 16))
            xmin = min(xmin, left)
            xmax = max(xmax, right)
            ymin = min(ymin, top)
            ymax = max(ymax, bottom)
            patch_indices[1] = [xmin, xmax, ymin, ymax]

        xs, ys = [], []
        xmin, xmax, ymin, ymax = patch_indices[2]
        for j in (range(72, 91 + 1) if not use_xpose else [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]): # 嘴巴
            if subsets[i, 0, j] < 0.3:
                continue
            x, y = candidates[i, 0, j]
            if x < eps or y < eps:
                continue
            xs.append(x)
            ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) != 0 and len(ys) != 0:
            # x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            left, right, top, bottom = get_patch_div(xs.mean(), ys.mean(), self.resolution[0], self.resolution[1], random.randint(8, 16))
            xmin = min(xmin, left)
            xmax = max(xmax, right)
            ymin = min(ymin, top)
            ymax = max(ymax, bottom)
            patch_indices[2] = [xmin, xmax, ymin, ymax]
    

    x1 = min(j[0] for j in patch_indices)
    x2 = max(j[1] for j in patch_indices)
    y1 = min(j[2] for j in patch_indices)
    y2 = max(j[3] for j in patch_indices)
    frames_eye_mouth[:, :, int(y1):int(y2), int(x1):int(x2)] = frames[:, :, int(y1):int(y2), int(x1):int(x2)]

    x1 = min(j[0] for j in patch_indices[:2])
    x2 = max(j[1] for j in patch_indices[:2])
    y1 = min(j[2] for j in patch_indices[:2])
    y2 = max(j[3] for j in patch_indices[:2])
    frames_eye[:, :, int(y1):int(y2), int(x1):int(x2)] = frames[:, :, int(y1):int(y2), int(x1):int(x2)]

    dwpose_results = []
    for candidate, subset in zip(candidates, subsets):
        dwpose_result_item = get_dwpose_body(candidate, subset, target_width, target_height)
        dwpose_results.append(dwpose_result_item)
    dwpose_results = np.array(dwpose_results, dtype="uint8")
    dwpose_results = torch.tensor(dwpose_results, ).permute(0, 3, 1, 2).float()

    inps = []
    ref_image, videos = torch.split(
        self.pixel_transforms(torch.cat([ref_image, videos], dim=0)), 
        (ref_image.shape[0], videos.shape[0]), dim=0)

    ref_image = ref_image.squeeze()
    swapped, frames_eye_mouth, frames_eye = torch.split(
        self.pixel_transforms(torch.cat([swapped, frames_eye_mouth, frames_eye], dim=0)), 
        ( swapped.shape[0], frames_eye_mouth.shape[0], frames_eye.shape[0]), dim=0)

    swapped = torch.sum(swapped * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)
    frames_eye_mouth = torch.sum(frames_eye_mouth * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)
    frames_eye = torch.sum(frames_eye * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)
    if np.random.rand() < self.warp_rate:
        warp_params = generate_random_params(*self.resolution)
        frames_eye_mouth = apply_transforms(frames_eye_mouth, warp_params)
        frames_eye = apply_transforms(frames_eye, warp_params)
        swapped = apply_transforms(swapped, warp_params)
        
    face_mask = torch.nn.functional.interpolate(face_mask, size=(target_height // 8, target_width // 8), mode='nearest')
    audio_length = len(audio_signal)

    clip_start_time, clip_end_time = max(0, start_time - 1.), min(audio_length / sample_rate, end_time + 1.)
    clip_start, clip_end, start_idx = int(clip_start_time * fps), int(clip_end_time * fps), int(start_time * fps)
    clip_start, clip_end, start_idx = clip_start - clip_start, clip_end - clip_start, start_idx - clip_start
    # 再根据this_start_time，这一段的开始时间，计算出相对时间
    # start_time, end_time = start_time - this_start_time, end_time - this_start_time
    # clip_start_time, clip_end_time = clip_start_time - this_start_time, clip_end_time - this_start_time
    # audio_signal_origin = audio_signal[int(sample_rate * start_time): int(sample_rate * end_time)]
    audio_signal = audio_signal[int(sample_rate * clip_start_time): int(sample_rate * clip_end_time)]
    audio_signal = torch.tensor(audio_signal)

    # audio_signal_origin = torch.tensor(audio_signal_origin)
    # audio_signal_origin = torch.cat([audio_signal_origin, audio_signal])
    # print("audio signal before is", audio_signal.shape)
    audio_signal = torchaudio.functional.resample(
        audio_signal, # shape need to be (n, ), range is [-1, 1], float32
        orig_freq=sample_rate,
        new_freq=self.standard_sample_rate,
    )
    while True:
        audio_signal_aug = audio_transforms(audio_signal.unsqueeze(0)).squeeze(0)
        if audio_signal_aug.shape == audio_signal.shape:
            break

    sample_dic = dict(
        reference=ref_image, 
        video=videos,
        swapped=swapped,
        frames_eye_mouth=frames_eye_mouth,
        frames_eye=frames_eye,
        start_time=start_time,
        face_mask=face_mask,
        end_time=end_time,
        audio_signal=audio_signal_aug,
        frame_stride=frame_stride,
        clip_start=clip_start, 
        clip_end=clip_end,
        start_idx=start_idx,
        fps=fps,
        )
    return sample_dic


def crop_audio_240728(self, ref_image, frames, swapped, start_time, end_time, candidates, subsets, faces_bbox, sample_rate, audio_signal, fps, start_idx, frame_stride, eval=False, use_xpose=False):
    # 首先裁剪出合适尺寸的人脸，并resize到期望的大小
    H, W = frames.shape[2:]
    l, t, r, b = W, H, 0, 0
    # for face_bbox, frame_is_valid in zip(faces_bbox, exist_face_frame_index):
    #     if not frame_is_valid:
    #         continue
    for face_bbox in faces_bbox:
        if use_xpose:
            face_bbox = face_bbox[0]
        x0, y0, x1, y1 = face_bbox
        l = min(x0, l)
        t = min(y0, t)
        r = max(r, x1)
        b = max(b, y1)
    face_mask = torch.zeros_like(frames)[:, :1, :, :]
    face_mask[:, :, int(t): int(b), int(l): int(r)] = 1.0

    # Get center and rect we need to use
    w, h = (r - l), (b - t)
    x_c, y_c = (l + r) / 2, (t + b) / 2
    expand_dis = max(w, h)
    left, right = max(x_c - expand_dis * self.left_scale, 0), min(x_c + expand_dis * self.right_scale, W)
    top, bottom = max(y_c - expand_dis * self.top_scale, 0), min(y_c + expand_dis * self.bottom_scale, H)
    
    # Get new center and new rect
    x_c, y_c = (left + right) / 2, (bottom + top) / 2
    distance_to_edge = min(x_c - left, right - x_c, y_c - top, bottom - y_c)    
    left = x_c - distance_to_edge
    right = x_c + distance_to_edge
    top = y_c - distance_to_edge
    bottom = y_c + distance_to_edge
    frames = frames[:, :, int(top):int(bottom), int(left):int(right)].float()
    target_height, target_width = self.resolution
    frames = videos = torch.nn.functional.interpolate(frames, size=(target_height, target_width), mode='bilinear', align_corners=False)

    swapped = swapped[:, :, int(top):int(bottom), int(left):int(right)].float()
    swapped = torch.nn.functional.interpolate(swapped, size=(target_height, target_width), mode='bilinear', align_corners=False)
    ref_image = ref_image[:, :, int(top):int(bottom), int(left):int(right)].float()
    ref_image = torch.nn.functional.interpolate(ref_image, size=(target_height, target_width), mode='bilinear', align_corners=False)
    
    # 再裁剪出025 的嘴巴和眼睛增强的人脸
    frames_eye_mouth = torch.zeros_like(frames)
    frames_eye = torch.zeros_like(frames)
    frames = swapped
    eps = 0.01

    # 首先将变换后的landmark进行resize
    for i in range(len(candidates)):
        for j in range(candidates.shape[2]):
            x, y = candidates[i, 0, j]
            x, y = (x - left) / (right - left), (y - top) / (bottom - top)
            x, y = x * target_width, y * target_height
            candidates[i, 0, j] = x, y
    
    patch_indices = [[1e6, 0, 1e6, 0], ] * 4
    for i in range(len(candidates)):
        xmin, xmax, ymin, ymax = patch_indices[0]
        xs, ys = [], []
        point_indexs = [41, 42, 43, 44, 45, 60, 61, 62, 63, 64, 65] if not use_xpose else [22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47]  # 左眼睛和左眉毛
        for j in point_indexs: # left eyes
            if subsets[i, 0, j] < 0.3:
                continue
            x, y = candidates[i, 0, j]
            if x < eps or y < eps:
                continue
            xs.append(x)
            ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) != 0 and len(ys) != 0:
            # x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            # left, right, top, bottom = get_patch(x0, y0, x1, y1, target_height, target_width)
            left, right, top, bottom = get_patch_div(xs.mean(), ys.mean(), self.resolution[0], self.resolution[1], random.randint(8, 16))
            xmin = min(xmin, left)
            xmax = max(xmax, right)
            ymin = min(ymin, top)
            ymax = max(ymax, bottom)
            patch_indices[0] = [xmin, xmax, ymin, ymax]

        xs, ys = [], []
        point_indexs = [46, 47, 48, 49, 50, 66, 67, 68, 69, 70, 71] if not use_xpose else [17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41] # 右眼睛和右眉毛
        xmin, xmax, ymin, ymax = patch_indices[1]
        for j in point_indexs: # left eyes
            if subsets[i, 0, j] < 0.3:
                continue
            x, y = candidates[i, 0, j]
            if x < eps or y < eps:
                continue
            xs.append(x)
            ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) != 0 and len(ys) != 0:
            # x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            left, right, top, bottom = get_patch_div(xs.mean(), ys.mean(), self.resolution[0], self.resolution[1], random.randint(8, 16))
            xmin = min(xmin, left)
            xmax = max(xmax, right)
            ymin = min(ymin, top)
            ymax = max(ymax, bottom)
            patch_indices[1] = [xmin, xmax, ymin, ymax]

        xs, ys = [], []
        xmin, xmax, ymin, ymax = patch_indices[2]
        for j in (range(72, 91 + 1) if not use_xpose else [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]): # 嘴巴
            if subsets[i, 0, j] < 0.3:
                continue
            x, y = candidates[i, 0, j]
            if x < eps or y < eps:
                continue
            xs.append(x)
            ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) != 0 and len(ys) != 0:
            # x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            left, right, top, bottom = get_patch_div(xs.mean(), ys.mean(), self.resolution[0], self.resolution[1], random.randint(8, 16))
            xmin = min(xmin, left)
            xmax = max(xmax, right)
            ymin = min(ymin, top)
            ymax = max(ymax, bottom)
            patch_indices[2] = [xmin, xmax, ymin, ymax]
    

    for j in patch_indices:
        x1, x2, y1, y2 = j
        frames_eye_mouth[:, :, int(y1):int(y2), int(x1):int(x2)] = frames[:, :, int(y1):int(y2), int(x1):int(x2)]
        frames_eye_mouth = torch.nn.functional.interpolate(frames_eye_mouth, size=(target_height, target_width), mode='bilinear', align_corners=False)
    
    for j in patch_indices[:2]:
        x1, x2, y1, y2 = j
        frames_eye[:, :, int(y1):int(y2), int(x1):int(x2)] = frames[:, :, int(y1):int(y2), int(x1):int(x2)]
        frames_eye = torch.nn.functional.interpolate(frames_eye, size=(target_height, target_width), mode='bilinear', align_corners=False)

    dwpose_results = []
    for candidate, subset in zip(candidates, subsets):
        dwpose_result_item = get_dwpose_body(candidate, subset, target_width, target_height)
        dwpose_results.append(dwpose_result_item)
    dwpose_results = np.array(dwpose_results, dtype="uint8")
    dwpose_results = torch.tensor(dwpose_results, ).permute(0, 3, 1, 2).float()

    inps = []
    ref_image, videos = torch.split(
        self.pixel_transforms(torch.cat([ref_image, videos], dim=0)), 
        (ref_image.shape[0], videos.shape[0]), dim=0)

    ref_image = ref_image.squeeze()
    swapped, frames_eye_mouth, frames_eye = torch.split(
        self.pixel_transforms(torch.cat([swapped, frames_eye_mouth, frames_eye], dim=0)), 
        ( swapped.shape[0], frames_eye_mouth.shape[0], frames_eye.shape[0]), dim=0)

    swapped = torch.sum(swapped * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)
    frames_eye_mouth = torch.sum(frames_eye_mouth * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)
    frames_eye = torch.sum(frames_eye * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)
    if np.random.rand() < self.warp_rate:
        warp_params = generate_random_params(*self.resolution)
        frames_eye_mouth = apply_transforms(frames_eye_mouth, warp_params)
        frames_eye = apply_transforms(frames_eye, warp_params)
        swapped = apply_transforms(swapped, warp_params)
        
    face_mask = torch.nn.functional.interpolate(face_mask, size=(target_height // 8, target_width // 8), mode='nearest')
    audio_length = len(audio_signal)

    clip_start_time, clip_end_time = max(0, start_time - 1.), min(audio_length / sample_rate, end_time + 1.)
    clip_start, clip_end, start_idx = int(clip_start_time * fps), int(clip_end_time * fps), int(start_time * fps)
    clip_start, clip_end, start_idx = clip_start - clip_start, clip_end - clip_start, start_idx - clip_start
    # 再根据this_start_time，这一段的开始时间，计算出相对时间
    # start_time, end_time = start_time - this_start_time, end_time - this_start_time
    # clip_start_time, clip_end_time = clip_start_time - this_start_time, clip_end_time - this_start_time
    # audio_signal_origin = audio_signal[int(sample_rate * start_time): int(sample_rate * end_time)]
    audio_signal = audio_signal[int(sample_rate * clip_start_time): int(sample_rate * clip_end_time)]
    audio_signal = torch.tensor(audio_signal)

    # audio_signal_origin = torch.tensor(audio_signal_origin)
    # audio_signal_origin = torch.cat([audio_signal_origin, audio_signal])
    # print("audio signal before is", audio_signal.shape)
    audio_signal = torchaudio.functional.resample(
        audio_signal, # shape need to be (n, ), range is [-1, 1], float32
        orig_freq=sample_rate,
        new_freq=self.standard_sample_rate,
    )
    while True:
        audio_signal_aug = audio_transforms(audio_signal.unsqueeze(0)).squeeze(0)
        if audio_signal_aug.shape == audio_signal.shape:
            break

    sample_dic = dict(
        reference=ref_image, 
        video=videos,
        swapped=swapped,
        frames_eye_mouth=frames_eye_mouth,
        frames_eye=frames_eye,
        start_time=start_time,
        face_mask=face_mask,
        end_time=end_time,
        audio_signal=audio_signal_aug,
        frame_stride=frame_stride,
        clip_start=clip_start, 
        clip_end=clip_end,
        start_idx=start_idx,
        fps=fps,
        )
    return sample_dic


def crop_audio_240722(self, ref_image, frames, swapped, start_time, end_time, candidates, subsets, faces_bbox, sample_rate, audio_signal, fps, start_idx, frame_stride, eval=False, use_xpose=False):
    # 首先裁剪出合适尺寸的人脸，并resize到期望的大小
    H, W = frames.shape[2:]
    l, t, r, b = W, H, 0, 0
    # for face_bbox, frame_is_valid in zip(faces_bbox, exist_face_frame_index):
    #     if not frame_is_valid:
    #         continue
    for face_bbox in faces_bbox:
        if use_xpose:
                face_bbox = face_bbox[0]
        x0, y0, x1, y1 = face_bbox
        l = min(x0, l)
        t = min(y0, t)
        r = max(r, x1)
        b = max(b, y1)
    face_mask = torch.zeros_like(frames)[:, :1, :, :]
    face_mask[:, :, int(t): int(b), int(l): int(r)] = 1.0

    # Get center and rect we need to use
    w, h = (r - l), (b - t)
    x_c, y_c = (l + r) / 2, (t + b) / 2
    expand_dis = max(w, h)
    left, right = max(x_c - expand_dis * self.left_scale, 0), min(x_c + expand_dis * self.right_scale, W)
    top, bottom = max(y_c - expand_dis * self.top_scale, 0), min(y_c + expand_dis * self.bottom_scale, H)
    
    # Get new center and new rect
    x_c, y_c = (left + right) / 2, (bottom + top) / 2
    distance_to_edge = min(x_c - left, right - x_c, y_c - top, bottom - y_c)    
    left = x_c - distance_to_edge
    right = x_c + distance_to_edge
    top = y_c - distance_to_edge
    bottom = y_c + distance_to_edge
    frames = frames[:, :, int(top):int(bottom), int(left):int(right)].float()
    target_height, target_width = self.resolution
    frames = videos = torch.nn.functional.interpolate(frames, size=(target_height, target_width), mode='bilinear', align_corners=False)

    swapped = swapped[:, :, int(top):int(bottom), int(left):int(right)].float()
    swapped = torch.nn.functional.interpolate(swapped, size=(target_height, target_width), mode='bilinear', align_corners=False)
    ref_image = ref_image[:, :, int(top):int(bottom), int(left):int(right)].float()
    ref_image = torch.nn.functional.interpolate(ref_image, size=(target_height, target_width), mode='bilinear', align_corners=False)
    
    # 再裁剪出025 的嘴巴和眼睛增强的人脸
    frames_eye_mouth = torch.zeros_like(frames)
    frames_eye = torch.zeros_like(frames)
    frames = swapped
    eps = 0.01

    # 首先将变换后的landmark进行resize
    for i in range(len(candidates)):
        for j in range(candidates.shape[2]):
            x, y = candidates[i, 0, j]
            x, y = (x - left) / (right - left), (y - top) / (bottom - top)
            x, y = x * target_width, y * target_height
            candidates[i, 0, j] = x, y
    
    for i in range(len(candidates)):
        xs, ys = [], []
        point_indexs = [41, 42, 43, 44, 45, 60, 61, 62, 63, 64, 65] if not use_xpose else [22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47]
        for j in point_indexs: # left eyes
            if subsets[i, 0, j] < 0.3:
                continue
            x, y = candidates[i, 0, j]
            if x < eps or y < eps:
                continue
            xs.append(x)
            ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) != 0 and len(ys) != 0:
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            xmin, xmax, ymin, ymax = get_patch(x0, y0, x1, y1, target_height, target_width)
            frames_eye_mouth[i, :, int(ymin):int(ymax), int(xmin):int(xmax)] = frames[i, :, int(ymin):int(ymax), int(xmin):int(xmax)]
            frames_eye[i, :, int(ymin):int(ymax), int(xmin):int(xmax)] = frames[i, :, int(ymin):int(ymax), int(xmin):int(xmax)]
    
        xs, ys = [], []
        point_indexs = [46, 47, 48, 49, 50, 66, 67, 68, 69, 70, 71] if not use_xpose else [17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41] # 右眼睛和右眉毛
        for j in point_indexs: # right eyes
            if subsets[i, 0, j] < 0.3:
                continue
            x, y = candidates[i, 0, j]
            if x < eps or y < eps:
                continue
            xs.append(x)
            ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) != 0 and len(ys) != 0:
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            xmin, xmax, ymin, ymax = get_patch(x0, y0, x1, y1, target_height, target_width)
            frames_eye_mouth[i, :, int(ymin):int(ymax), int(xmin):int(xmax)] = frames[i, :, int(ymin):int(ymax), int(xmin):int(xmax)]
            frames_eye[i, :, int(ymin):int(ymax), int(xmin):int(xmax)] = frames[i, :, int(ymin):int(ymax), int(xmin):int(xmax)]

        xs, ys = [], []
        for j in (range(72, 91 + 1) if not use_xpose else [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]): # 嘴巴
            if subsets[i, 0, j] < 0.3:
                continue
            x, y = candidates[i, 0, j]
            if x < eps or y < eps:
                continue
            xs.append(x)
            ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) != 0 and len(ys) != 0:
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            xmin, xmax, ymin, ymax = get_patch(x0, y0, x1, y1, target_height, target_width)
            frames_eye_mouth[i, :, int(ymin):int(ymax), int(xmin):int(xmax)] = frames[i, :, int(ymin):int(ymax), int(xmin):int(xmax)]

        frames_eye = torch.nn.functional.interpolate(frames_eye, size=(target_height, target_width), mode='bilinear', align_corners=False)
        frames_eye_mouth = torch.nn.functional.interpolate(frames_eye_mouth, size=(target_height, target_width), mode='bilinear', align_corners=False)

    dwpose_results = []
    for candidate, subset in zip(candidates, subsets):
        dwpose_result_item = get_dwpose_body(candidate, subset, target_width, target_height)
        dwpose_results.append(dwpose_result_item)
    dwpose_results = np.array(dwpose_results, dtype="uint8")
    dwpose_results = torch.tensor(dwpose_results, ).permute(0, 3, 1, 2).float()

    inps = []
    ref_image, videos = torch.split(
        self.pixel_transforms(torch.cat([ref_image, videos], dim=0)), 
        (ref_image.shape[0], videos.shape[0]), dim=0)

    ref_image = ref_image.squeeze()
    swapped, frames_eye_mouth = torch.split(
        self.pixel_transforms(torch.cat([swapped, frames_eye_mouth], dim=0)), 
        ( swapped.shape[0], frames_eye_mouth.shape[0]), dim=0)

    swapped = torch.sum(swapped * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)
    frames_eye_mouth = torch.sum(frames_eye_mouth * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)
    frames_eye = torch.sum(frames_eye * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)
    if np.random.rand() < self.warp_rate:
        warp_params = generate_random_params(*self.resolution)
        frames_eye_mouth = apply_transforms(frames_eye_mouth, warp_params)
        frames_eye = apply_transforms(frames_eye, warp_params)
        swapped = apply_transforms(swapped, warp_params)
        
    face_mask = torch.nn.functional.interpolate(face_mask, size=(target_height // 8, target_width // 8), mode='nearest')
    audio_length = len(audio_signal)

    clip_start_time, clip_end_time = max(0, start_time - 1.), min(audio_length / sample_rate, end_time + 1.)
    clip_start, clip_end, start_idx = int(clip_start_time * fps), int(clip_end_time * fps), int(start_time * fps)
    clip_start, clip_end, start_idx = clip_start - clip_start, clip_end - clip_start, start_idx - clip_start
    # 再根据this_start_time，这一段的开始时间，计算出相对时间
    # start_time, end_time = start_time - this_start_time, end_time - this_start_time
    # clip_start_time, clip_end_time = clip_start_time - this_start_time, clip_end_time - this_start_time
    # audio_signal_origin = audio_signal[int(sample_rate * start_time): int(sample_rate * end_time)]
    audio_signal = audio_signal[int(sample_rate * clip_start_time): int(sample_rate * clip_end_time)]
    audio_signal = torch.tensor(audio_signal)

    # audio_signal_origin = torch.tensor(audio_signal_origin)
    # audio_signal_origin = torch.cat([audio_signal_origin, audio_signal])
    # print("audio signal before is", audio_signal.shape)
    audio_signal = torchaudio.functional.resample(
        audio_signal, # shape need to be (n, ), range is [-1, 1], float32
        orig_freq=sample_rate,
        new_freq=self.standard_sample_rate,
    )
    while True:
        audio_signal_aug = audio_transforms(audio_signal.unsqueeze(0)).squeeze(0)
        if audio_signal_aug.shape == audio_signal.shape:
            break
    # print(audio_length, int(sample_rate * clip_start_time), int(sample_rate * clip_end_time), audio_signal.shape, audio_signal_aug.shape)
    # audio_signal_origin = torchaudio.functional.resample(
    #     audio_signal_origin, # shape need to be (n, ), range is [-1, 1], float32
    #     orig_freq=sample_rate,
    #     new_freq=self.standard_sample_rate,
    # )

    sample_dic = dict(
        reference=ref_image, 
        video=videos,
        swapped=swapped,
        frames_eye_mouth=frames_eye_mouth,
        frames_eye=frames_eye,
        start_time=start_time,
        face_mask=face_mask,
        end_time=end_time,
        audio_signal=audio_signal_aug,
        frame_stride=frame_stride,
        clip_start=clip_start, 
        clip_end=clip_end,
        start_idx=start_idx,
        fps=fps,
        )
    return sample_dic


def crop_small_area(self, ref_image, frame, swapped, eval=False):
    if not eval:
        batched_ref_frame = torch.cat([ref_image, frame], dim=0)
        batched_fake_swapped = torch.cat([ref_image, swapped], dim=0)
        faces = self.face_detector(batched_ref_frame)
        if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
            return None
        min_face_size, seg_area, _, batched_ref_frame  = crop_and_resize_tensor_small_faces(batched_ref_frame, faces, target_size=self.resolution)
        _, _, _, batched_fake_swapped  = crop_and_resize_tensor_small_faces(batched_fake_swapped, faces, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < 0.05 or min_face_size / seg_area < 0:
            return None
        ref_image, frame = batched_ref_frame[:1], batched_ref_frame[1:]
        _, swapped = batched_fake_swapped[:1], batched_fake_swapped[1:]

    else:
        faces = self.face_detector(frame)
        if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
            return None
        min_face_size, face_rects, bbox, frame  = crop_and_resize_tensor_small_faces(frame, faces, target_size=self.resolution)
        _, _, _, swapped  = crop_and_resize_tensor_small_faces(swapped, faces, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < self.min_face_thresh:
            return None

        faces_ref = self.face_detector(frame)
        if 'image_ids' not in faces_ref.keys() or faces_ref['image_ids'].numel() == 0:
            return None
        min_face_size, face_rects, bbox, frame  = crop_and_resize_tensor_with_face_rects(ref_image, faces_ref, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < self.min_face_thresh:
            return None

    swapped_faces = self.face_detector(frame)
    if 'image_ids' not in swapped_faces.keys() or swapped_faces['image_ids'].numel() == 0:
        return None
    swapped = crop_move_face(swapped, swapped_faces, target_size=self.resolution)
    if swapped is None:
        return None
    if not eval and np.random.rand() < self.warp_rate:
        warp_params = generate_random_params(*self.resolution)
        swapped = apply_transforms(swapped, warp_params)
    swapped = torch.sum(swapped * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)

    with torch.inference_mode():
        ref_concat_image_noises = []
        ref_img_background_masks = []
        ref_img_clips = []
        image_np = rearrange(ref_image, "b c h w -> b h w c")
        image_np = image_np.cpu().numpy().astype(np.uint8)
        for i, ref_img in enumerate(image_np):
            ref_img = Image.fromarray(ref_img)
            dwpose_model_result_dict = self.dwpose_model(ref_img)

            ref_concat_image_noise = dwpose_model_result_dict[self.concat_noise_image_type]
            ref_concat_image_noises.append(torch.tensor(ref_concat_image_noise).permute(2, 0, 1))

            ref_img_background_mask = dwpose_model_result_dict['background_mask']
            ref_img_background_masks.append(torch.tensor(ref_img_background_mask).squeeze())                                

            ref_img_clip = dwpose_model_result_dict[self.clip_image_type]
            ref_img_clips.append(torch.tensor(ref_img_clip).permute(2, 0, 1))
        concat_poses = torch.stack(ref_concat_image_noises, dim=0)
        concat_background = torch.stack(ref_img_background_masks, dim=0)
        clip_conditions = torch.stack(ref_img_clips, dim=0)

    ref_image = ref_image.squeeze()

    sample_dic = dict(
        reference=ref_image, 
        video=frame,
        swapped=swapped,
        concat_poses=concat_poses,
        concat_background=concat_background,
        clip_conditions=clip_conditions,
        )

    return sample_dic

def crop_flex_area(self, ref_image, frame, swapped, eval=False):
    if not eval:
        batched_ref_frame = torch.cat([ref_image, frame], dim=0)
        batched_fake_swapped = torch.cat([ref_image, swapped], dim=0)
        faces = self.face_detector(batched_ref_frame)
        if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
            return None
        min_face_size, _, _, batched_ref_frame  = crop_and_resize_tensor_flex(batched_ref_frame, faces, target_size=self.resolution)
        _, _, _, batched_fake_swapped  = crop_and_resize_tensor_flex(batched_fake_swapped, faces, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < 0.05:
            return None
        ref_image, frame = batched_ref_frame[:1], batched_ref_frame[1:]
        _, swapped = batched_fake_swapped[:1], batched_fake_swapped[1:]

    else:
        faces = self.face_detector(frame)
        if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
            return None
        min_face_size, face_rects, bbox, frame  = crop_and_resize_tensor_flex(frame, faces, target_size=self.resolution)
        _, _, _, swapped  = crop_and_resize_tensor_flex(swapped, faces, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < self.min_face_thresh:
            return None

        faces_ref = self.face_detector(frame)
        if 'image_ids' not in faces_ref.keys() or faces_ref['image_ids'].numel() == 0:
            return None
        min_face_size, face_rects, bbox, frame  = crop_and_resize_tensor_flex(ref_image, faces_ref, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < self.min_face_thresh:
            return None

    swapped_faces = self.face_detector(frame)
    if 'image_ids' not in swapped_faces.keys() or swapped_faces['image_ids'].numel() == 0:
        return None
    swapped = crop_move_face(swapped, swapped_faces, target_size=self.resolution)
    if swapped is None:
        return None
    if not eval and np.random.rand() < self.warp_rate:
        warp_params = generate_random_params(*self.resolution)
        swapped = apply_transforms(swapped, warp_params)
    swapped = torch.sum(swapped * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)
    # video_length = swapped.shape[1]
    # masked_indices = random.choices([])

    with torch.inference_mode():
        ref_concat_image_noises = []
        ref_img_background_masks = []
        ref_img_clips = []
        image_np = rearrange(ref_image, "b c h w -> b h w c")
        image_np = image_np.cpu().numpy().astype(np.uint8)
        for i, ref_img in enumerate(image_np):
            ref_img = Image.fromarray(ref_img)
            dwpose_model_result_dict = self.dwpose_model(ref_img)

            ref_concat_image_noise = dwpose_model_result_dict[self.concat_noise_image_type]
            ref_concat_image_noises.append(torch.tensor(ref_concat_image_noise).permute(2, 0, 1))

            ref_img_background_mask = dwpose_model_result_dict['background_mask']
            ref_img_background_masks.append(torch.tensor(ref_img_background_mask).squeeze())                                

            ref_img_clip = dwpose_model_result_dict[self.clip_image_type]
            ref_img_clips.append(torch.tensor(ref_img_clip).permute(2, 0, 1))
        concat_poses = torch.stack(ref_concat_image_noises, dim=0)
        concat_background = torch.stack(ref_img_background_masks, dim=0)
        clip_conditions = torch.stack(ref_img_clips, dim=0)

    ref_image = ref_image.squeeze()

    sample_dic = dict(
        reference=ref_image, 
        video=frame,
        swapped=swapped,
        concat_poses=concat_poses,
        concat_background=concat_background,
        clip_conditions=clip_conditions,
        )

    return sample_dic


def crop_area(self, ref_image, frame, swapped, eval=False):
    if not eval:
        batched_ref_frame = torch.cat([ref_image, frame], dim=0)
        batched_fake_swapped = torch.cat([ref_image, swapped], dim=0)
        faces = self.face_detector(batched_ref_frame)
        if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
            return None
        min_face_size, seg_area, _, batched_ref_frame  = crop_and_resize_tensor_with_face_rects(batched_ref_frame, faces, target_size=self.resolution)
        _, _, _, batched_fake_swapped  = crop_and_resize_tensor_with_face_rects(batched_fake_swapped, faces, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < 0.05 or min_face_size / seg_area < 0:
            return None
        ref_image, frame = batched_ref_frame[:1], batched_ref_frame[1:]
        _, swapped = batched_fake_swapped[:1], batched_fake_swapped[1:]

    else:
        faces = self.face_detector(frame)
        if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
            return None
        min_face_size, face_rects, bbox, frame  = crop_and_resize_tensor_with_face_rects(frame, faces, target_size=self.resolution)
        _, _, _, swapped  = crop_and_resize_tensor_with_face_rects(swapped, faces, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < self.min_face_thresh:
            return None

        faces_ref = self.face_detector(frame)
        if 'image_ids' not in faces_ref.keys() or faces_ref['image_ids'].numel() == 0:
            return None
        min_face_size, face_rects, bbox, frame  = crop_and_resize_tensor_with_face_rects(ref_image, faces_ref, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < self.min_face_thresh:
            return None

    swapped_faces = self.face_detector(frame)
    if 'image_ids' not in swapped_faces.keys() or swapped_faces['image_ids'].numel() == 0:
        return None
    swapped = crop_move_face(swapped, swapped_faces, target_size=self.resolution)
    if swapped is None:
        return None
    if not eval and np.random.rand() < self.warp_rate:
        warp_params = generate_random_params(*self.resolution)
        swapped = apply_transforms(swapped, warp_params)
    swapped = torch.sum(swapped * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)
    # video_length = swapped.shape[1]
    # masked_indices = random.choices([])

    with torch.inference_mode():
        ref_concat_image_noises = []
        ref_img_background_masks = []
        ref_img_clips = []
        image_np = rearrange(ref_image, "b c h w -> b h w c")
        image_np = image_np.cpu().numpy().astype(np.uint8)
        for i, ref_img in enumerate(image_np):
            ref_img = Image.fromarray(ref_img)
            dwpose_model_result_dict = self.dwpose_model(ref_img)

            ref_concat_image_noise = dwpose_model_result_dict[self.concat_noise_image_type]
            ref_concat_image_noises.append(torch.tensor(ref_concat_image_noise).permute(2, 0, 1))

            ref_img_background_mask = dwpose_model_result_dict['background_mask']
            ref_img_background_masks.append(torch.tensor(ref_img_background_mask).squeeze())                                

            ref_img_clip = dwpose_model_result_dict[self.clip_image_type]
            ref_img_clips.append(torch.tensor(ref_img_clip).permute(2, 0, 1))
        concat_poses = torch.stack(ref_concat_image_noises, dim=0)
        concat_background = torch.stack(ref_img_background_masks, dim=0)
        clip_conditions = torch.stack(ref_img_clips, dim=0)

    ref_image = ref_image.squeeze()

    sample_dic = dict(
        reference=ref_image, 
        video=frame,
        swapped=swapped,
        concat_poses=concat_poses,
        concat_background=concat_background,
        clip_conditions=clip_conditions,
        )

    return sample_dic


def crop_lean_patch(self, ref_image, frame, swapped, eval=False):
    control_frames = []
    if not eval:
        batched_ref_frame = torch.cat([ref_image, frame], dim=0)
        batched_fake_swapped = torch.cat([ref_image, swapped], dim=0)
        faces = self.face_detector(batched_ref_frame)
        if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
            return None
        min_face_size, seg_area, _, batched_ref_frame  = crop_and_resize_tensor_with_face_rects(batched_ref_frame, faces, target_size=self.resolution)
        _, _, _, batched_fake_swapped  = crop_and_resize_tensor_with_face_rects(batched_fake_swapped, faces, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < 0.05 or min_face_size / seg_area < 0:
            return None
        ref_image, frame = batched_ref_frame[:1], batched_ref_frame[1:]
        _, swapped = batched_fake_swapped[:1], batched_fake_swapped[1:]

    else:
        faces = self.face_detector(frame)
        if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
            return None
        min_face_size, face_rects, bbox, frame  = crop_and_resize_tensor_with_face_rects(frame, faces, target_size=self.resolution)
        _, _, _, swapped  = crop_and_resize_tensor_with_face_rects(swapped, faces, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < self.min_face_thresh:
            return None

        faces_ref = self.face_detector(ref_image)
        if 'image_ids' not in faces_ref.keys() or faces_ref['image_ids'].numel() == 0:
            return None
        min_face_size, face_rects, bbox, ref_image  = crop_and_resize_tensor_with_face_rects(ref_image, faces_ref, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < self.min_face_thresh:
            return None

    swapped_faces = self.face_detector(frame)
    swapped_np = swapped.permute(0, 2, 3, 1).cpu().numpy()
    if 'image_ids' not in swapped_faces.keys() or swapped_faces['image_ids'].numel() == 0:
        return None
    for frame_index in range(frame.shape[0]):
        cur_control = swapped_np[frame_index]
        _, __, ldm = self.dwpose_model.dwpose_model(cur_control, output_type='np', image_resolution=self.resolution[0], get_mark=True)
        ldm = ldm["faces_all"][0] * self.resolution[0]

        masked_frame = np.zeros_like(cur_control)
        for patch_idx, (kp_index_begin, kp_index_end, div_n) in enumerate([(36, 42, 8), (42, 48, 8), (48, 68, 8)]):
            xmin= ldm[kp_index_begin: kp_index_end][..., 0].min()
            xmax= ldm[kp_index_begin: kp_index_end][..., 0].max()
            xc = (xmin + xmax) / 2
            ymin= ldm[kp_index_begin: kp_index_end][..., 1].min()
            ymax= ldm[kp_index_begin: kp_index_end][..., 1].max()
            yc = (ymin + ymax) / 2

            pw = xmax - xmin
            ph = ymax - ymin
            pw = ph = max(pw, ph)

            xmin = max(0, xc - 0.5 * pw)
            xmax = max(0, xc + 0.5 * pw)
            ymin = max(0, yc - 0.5 * ph)
            ymax = max(0, yc + 0.5 * ph)

            masked_frame[int(ymin):int(ymax), int(xmin):int(xmax), :] = cur_control[int(ymin):int(ymax), int(xmin):int(xmax), :]
        control_frames.append(torch.tensor(masked_frame))
    swapped = torch.stack(control_frames, dim=0).permute(0, 3, 1, 2)

    if not eval and np.random.rand() < self.warp_rate:
        warp_params = generate_random_params(*self.resolution)
        swapped = apply_transforms(swapped, warp_params)
    swapped = torch.sum(swapped * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)

    with torch.inference_mode():
        ref_concat_image_noises = []
        ref_img_background_masks = []
        ref_img_clips = []
        image_np = rearrange(ref_image, "b c h w -> b h w c")
        image_np = image_np.cpu().numpy().astype(np.uint8)
        for i, ref_img in enumerate(image_np):
            ref_img = Image.fromarray(ref_img)
            dwpose_model_result_dict = self.dwpose_model(ref_img)

            ref_concat_image_noise = dwpose_model_result_dict[self.concat_noise_image_type]
            ref_concat_image_noises.append(torch.tensor(ref_concat_image_noise).permute(2, 0, 1))

            ref_img_background_mask = dwpose_model_result_dict['background_mask']
            ref_img_background_masks.append(torch.tensor(ref_img_background_mask).squeeze())                                

            ref_img_clip = dwpose_model_result_dict[self.clip_image_type]
            ref_img_clips.append(torch.tensor(ref_img_clip).permute(2, 0, 1))
        concat_poses = torch.stack(ref_concat_image_noises, dim=0)
        concat_background = torch.stack(ref_img_background_masks, dim=0)
        clip_conditions = torch.stack(ref_img_clips, dim=0)

    ref_image = ref_image.squeeze()

    sample_dic = dict(
        reference=ref_image, 
        video=frame,
        swapped=swapped,
        concat_poses=concat_poses,
        concat_background=concat_background,
        clip_conditions=clip_conditions,
        )

    return sample_dic

def crop_patch(self, ref_image, frame, swapped, eval=False):
    control_frames = []
    if not eval:
        batched_ref_frame = torch.cat([ref_image, frame], dim=0)
        batched_fake_swapped = torch.cat([ref_image, swapped], dim=0)
        faces = self.face_detector(batched_ref_frame)
        if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
            return None
        min_face_size, seg_area, _, batched_ref_frame  = crop_and_resize_tensor_with_face_rects(batched_ref_frame, faces, target_size=self.resolution)
        _, _, _, batched_fake_swapped  = crop_and_resize_tensor_with_face_rects(batched_fake_swapped, faces, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < 0.05 or min_face_size / seg_area < 0:
            return None
        ref_image, frame = batched_ref_frame[:1], batched_ref_frame[1:]
        _, swapped = batched_fake_swapped[:1], batched_fake_swapped[1:]

    else:
        faces = self.face_detector(frame)
        if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
            return None
        min_face_size, face_rects, bbox, frame  = crop_and_resize_tensor_with_face_rects(frame, faces, target_size=self.resolution)
        _, _, _, swapped  = crop_and_resize_tensor_with_face_rects(swapped, faces, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < self.min_face_thresh:
            return None

        faces_ref = self.face_detector(ref_image)
        if 'image_ids' not in faces_ref.keys() or faces_ref['image_ids'].numel() == 0:
            return None
        min_face_size, face_rects, bbox, ref_image  = crop_and_resize_tensor_with_face_rects(ref_image, faces_ref, target_size=self.resolution)
        if min_face_size is None:
            return None
        if min_face_size < self.min_face_thresh:
            return None

    swapped_faces = self.face_detector(frame)
    swapped_np = swapped.permute(0, 2, 3, 1).cpu().numpy()
    if 'image_ids' not in swapped_faces.keys() or swapped_faces['image_ids'].numel() == 0:
        return None

    if eval:
        cur_ref = ref_image.permute(0, 2, 3, 1)[0].cpu().numpy()
        _, __, dist_point = self.dwpose_model.dwpose_model(cur_ref, output_type='np', image_resolution=self.resolution[0], get_mark=True)
        dist_point = dist_point["faces_all"][0] * self.resolution[0]
        right, bottom = dist_point[:, :].max(axis=0)
        left, top = dist_point[:, :].min(axis=0)
        dist_point = np.array([[left, top], [right, bottom], [left, bottom]]).astype("int32")
    patch_indices = [[1e6, 0, 1e6, 0], ] * 4

    for frame_index in range(frame.shape[0]):
        cur_control = swapped_np[frame_index]
        _, __, ldm = self.dwpose_model.dwpose_model(cur_control, output_type='np', image_resolution=self.resolution[0], get_mark=True)
        ldm = ldm["faces_all"][0] * self.resolution[0]

        masked_frame = np.zeros_like(cur_control)
        for patch_idx, (kp_index_begin, kp_index_end, div_n) in enumerate([(36, 42, 8), (42, 48, 8), (48, 68, 8)]):
            xmin, xmax, ymin, ymax = patch_indices[patch_idx]
            x_mean, y_mean = np.mean(ldm[kp_index_begin: kp_index_end], axis=0) # left eyes
            left, right, top, bottom = get_patch_div(x_mean, y_mean, self.resolution[0], self.resolution[1], div_n)
            xmin = min(xmin, left)
            xmax = max(xmax, right)
            ymin = min(ymin, top)
            ymax = max(ymax, bottom)
            patch_indices[patch_idx] = [xmin, xmax, ymin, ymax]


    for frame_index in range(frame.shape[0]):
        cur_control = swapped_np[frame_index]
        # masked_frame = cur_control / 2
        masked_frame = np.zeros_like(cur_control)
        for xmin, xmax, ymin, ymax in patch_indices:
            masked_frame[int(ymin):int(ymax), int(xmin):int(xmax), :] = cur_control[int(ymin):int(ymax), int(xmin):int(xmax), :]
        if eval and frame_index == 0:
            right, bottom = ldm[:, :].max(axis=0)
            left, top = ldm[:, :].min(axis=0)
            src_point = np.array([[left, top], [right, bottom], [left, bottom]]).astype("int32")
        control_frames.append(masked_frame if eval else torch.tensor(masked_frame))
        # control_frames.append(torch.tensor(masked_frame))
    if eval:
        transform_matrix = cv2.getAffineTransform(np.float32(src_point), np.float32(dist_point))
        control_frames = [torch.Tensor(cv2.warpAffine(item, transform_matrix, self.resolution)) for item in control_frames]
    swapped = torch.stack(control_frames, dim=0).permute(0, 3, 1, 2)

    if not eval and np.random.rand() < self.warp_rate:
        warp_params = generate_random_params(*self.resolution)
        swapped = apply_transforms(swapped, warp_params)
    swapped = torch.sum(swapped * self.color_BW_weights, dim=1, keepdim=True).clamp(0, 255.).repeat(1, 3, 1, 1)

    with torch.inference_mode():
        ref_concat_image_noises = []
        ref_img_background_masks = []
        ref_img_clips = []
        image_np = rearrange(ref_image, "b c h w -> b h w c")
        image_np = image_np.cpu().numpy().astype(np.uint8)
        for i, ref_img in enumerate(image_np):
            ref_img = Image.fromarray(ref_img)
            dwpose_model_result_dict = self.dwpose_model(ref_img)

            ref_concat_image_noise = dwpose_model_result_dict[self.concat_noise_image_type]
            ref_concat_image_noises.append(torch.tensor(ref_concat_image_noise).permute(2, 0, 1))

            ref_img_background_mask = dwpose_model_result_dict['background_mask']
            ref_img_background_masks.append(torch.tensor(ref_img_background_mask).squeeze())                                

            ref_img_clip = dwpose_model_result_dict[self.clip_image_type]
            ref_img_clips.append(torch.tensor(ref_img_clip).permute(2, 0, 1))
        concat_poses = torch.stack(ref_concat_image_noises, dim=0)
        concat_background = torch.stack(ref_img_background_masks, dim=0)
        clip_conditions = torch.stack(ref_img_clips, dim=0)

    ref_image = ref_image.squeeze()

    sample_dic = dict(
        reference=ref_image, 
        video=frame,
        swapped=swapped,
        concat_poses=concat_poses,
        concat_background=concat_background,
        clip_conditions=clip_conditions,
        )

    return sample_dic
