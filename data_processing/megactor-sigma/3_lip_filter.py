import cv2
import traceback
import numpy as np
from tqdm import tqdm 
import imageio.v3 as iio
import pickle
import math

from scipy.signal import resample
from moviepy.editor import VideoClip, AudioFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from pydub import AudioSegment
import webdataset as wds
from webdataset import gopen, gopen_schemes
import torch
import python_speech_features

from syncnet_python.SyncNetInstance import *

def calc_pdist(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1

    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    return dists

def numpy_to_wav(audio_signal, sample_rate=44100, filename="output.wav"):
    # 将 NumPy 数组转换为 Pydub AudioSegment 对象
    audio_signal = (audio_signal * 32767).astype(np.int16)  # 确保信号是整数类型
    audio_segment = AudioSegment(
        audio_signal.tobytes(), 
        frame_rate=sample_rate,
        sample_width=audio_signal.dtype.itemsize, 
        channels=1
    )
    # 导出为 WAV 文件
    audio_segment.export(filename, format="wav")
    return filename


def change_fps(video_frames, target_fps, original_fps):
    """
    Change the frame rate of the video.
    
    :param video_frames: List of frames (each frame is a numpy array with shape (h, w, c))
    :param target_fps: Target frame rate (int)
    :param original_fps: Original frame rate (int)
    :return: List of frames with the new frame rate
    """
    # Calculate the ratio of the new frame rate to the old frame rate
    frame_ratio = target_fps / original_fps
    # Calculate the indices of the frames to keep
    new_frame_indices = np.arange(0, len(video_frames), 1 / frame_ratio).astype(int)
    # Select the frames
    new_video_frames = [video_frames[i] for i in new_frame_indices if i < len(video_frames)]
    return new_video_frames

def resample_audio(audio_signal, original_rate, target_rate):
    """
    Resample the audio signal to a new sample rate.
    
    :param audio_signal: 1D numpy array of the audio signal
    :param original_rate: Original sample rate (int)
    :param target_rate: Target sample rate (int)
    :return: Resampled audio signal (1D numpy array)
    """
    number_of_samples = round(len(audio_signal) * float(target_rate) / original_rate)
    resampled_signal = resample(audio_signal, number_of_samples)
    return resampled_signal


def get_clip_frames(video_bytes:bytes) -> np.ndarray:
    frames = []
    # fps = iio.immeta(video_bytes, plugin="pyav")["fps"] #有时候会报错拿不到这个属性
    # fps = int(math.floor(fps+0.5))
    with iio.imopen(video_bytes, "r", plugin="pyav") as file:
        
        frames = file.read(index=...)  # np.array, [n_frames,h,w,c],  rgb, uint8
        frames = np.array(frames, dtype=frames.dtype) 
            
        # n_frames = frames.shape[0]
        # duration = file._video_stream.duration
    return frames#, fps


def is_front_face(landmarks: np.ndarray):

    angR = npAngle(landmarks[0], landmarks[1], landmarks[2]) # Calculate the right eye angle
    angL = npAngle(landmarks[1], landmarks[0], landmarks[2])# Calculate the left eye angle
    if ((int(angR) in range(35, 65)) and (int(angL) in range(35, 65))):
        return True
    else:
        return False

def npAngle(a, b, c):
    ba = a - b
    bc = c - b 
    
    cosine_angle = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def get_dwpose_output(sync_model, frames_np, dwpose_results, dwpose_scores, faces_bboxs, audio_signal, sample_rate, fps):
    audio_frames = []
    num_frames, H, W, C = frames_np.shape
    l, t, r, b = W, H, 0, 0
    x_c_list, y_c_list = [], []
    w_list, h_list = [], []
    eps = 0.01
    direct_right_num = 0
    for frame_id in tqdm(range(num_frames)):
        frame_np = frames_np[frame_id]
        candidates = dwpose_results[frame_id]
        subsets = dwpose_scores[frame_id]
        face_bbox = faces_bboxs[frame_id]
        x0, y0, x1, y1 = face_bbox
        if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
            continue
        
        check_face_direct = True
        face_points = []
        xs = []
        ys = []
        for j in range(60, 66): # left eyes
            if subsets[0, j] < 0.3:
                continue
            x, y = candidates[0, j]
            if x < eps or y < eps:
                continue
            xs.append(x)
            ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) == 0:
            check_face_direct = False
        else:
            face_points.append([xs.mean(), ys.mean()])
        xs = []
        ys = []
        for j in range(66, 72): # right eyes
            if subsets[0, j] < 0.3:
                continue
            x, y = candidates[0, j]
            if x < eps or y < eps:
                continue
            xs.append(x)
            ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) == 0:
            check_face_direct = False
        else:
            face_points.append([xs.mean(), ys.mean()])
        xs = []
        ys = []
        for j in range(54, 60): # bose
            if subsets[0, j] < 0.3:
                continue
            x, y = candidates[0, j]
            if x < eps or y < eps:
                continue
            xs.append(x)
            ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) == 0:
            check_face_direct = False
        else:
            face_points.append([xs.mean(), ys.mean()])
        for j in [72, 78]:
            if subsets[0, j] < 0.3:
                check_face_direct = False
                continue
            x, y = candidates[0, j]
            if x < eps or y < eps:
                check_face_direct = False
                continue
            face_points.append([x, y])
        if check_face_direct and is_front_face(np.array(face_points)):
            direct_right_num += 1
        w, h = (x1 - x0), (y1 - y0)
        x_c, y_c = (x0 + x1) / 2, (y0 + y1) / 2
        w_list.append(w)
        h_list.append(h)
        x_c_list.append(x_c)
        y_c_list.append(y_c)
    direct_right_num /= num_frames
    move_ratio = 0.
    if len(w_list) != 0:
        w_mean = np.array(w_list).mean()
        h_mean = np.array(h_list).mean()
        delta_x, delta_y = np.array(x_c_list).max() - np.array(x_c_list).min(), \
            np.array(y_c_list).max() - np.array(y_c_list).min()
        move_ratio = max(delta_x / w_mean, delta_y / h_mean)
        
    
    for frame_id in tqdm(range(num_frames)):
        frame_np = frames_np[frame_id]
        candidates = dwpose_results[frame_id]
        subsets = dwpose_scores[frame_id]
        face_bbox = faces_bboxs[frame_id]
        
        x0, y0, x1, y1 = face_bbox
        if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
            x0, y0, x1, y1 = 0, 0, W, H
        w, h = (x1 - x0), (y1 - y0)
        x_c, y_c = (x0 + x1) / 2, (y0 + y1) / 2
        expand_dis = max(w, h)
        left, right = max(x_c - expand_dis * 1., 0), min(x_c + expand_dis * 1., W)
        top, bottom = max(y_c - expand_dis * 1., 0), min(y_c + expand_dis * 1.2, H)
        audio_frame = frame_np[int(top):int(bottom), int(left):int(right), :]
        audio_frames.append(cv2.resize(cv2.cvtColor(audio_frame, cv2.COLOR_RGB2BGR), (224, 224)))
    
    audio_frames = change_fps(audio_frames, 25, fps)
    audio_signal = resample_audio(audio_signal, sample_rate, 16000)
    audio = (audio_signal * 32767).astype(np.int16)

    im = np.stack(audio_frames,axis=3)
    im = np.expand_dims(im,axis=0)
    im = np.transpose(im,(0,3,4,1,2))

    imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

    # ========== ==========
    # Load audio
    # ========== ==========

    mfcc = zip(*python_speech_features.mfcc(audio , 16000))
    mfcc = np.stack([np.array(i) for i in mfcc])

    cc = np.expand_dims(np.expand_dims(mfcc,axis=0),axis=0)
    cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

    # ========== ==========
    # Check audio and video input length
    # ========== ==========

    min_length = min(len(audio_frames), math.floor(len(audio)/640))
    
    # ========== ==========
    # Generate video and audio feats
    # ========== ==========

    lastframe = min_length-5
    im_feat = []
    cc_feat = []
    batch_size = 32
    for i in range(0, lastframe, batch_size):
        # print("i is", i)
        im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i + batch_size)) ]
        im_in = torch.cat(im_batch,0)
        im_out  = sync_model.__S__.forward_lip(im_in.cuda());
        im_feat.append(im_out.data.cpu())

        cc_batch = [ cct[:,:,:,vframe*4 : vframe*4+20] for vframe in range(i,min(lastframe,i + batch_size)) ]
        # for cc_item in cc_batch:
        #     print("cc_item shape is", cc_item.shape)
        cc_in = torch.cat(cc_batch,0)
        cc_out  = sync_model.__S__.forward_aud(cc_in.cuda())
        cc_feat.append(cc_out.data.cpu())

    im_feat = torch.cat(im_feat,0)
    cc_feat = torch.cat(cc_feat,0)

    # ========== ==========
    # Compute offset
    # ========== ==========
    vshift = 15
    dists = calc_pdist(im_feat,cc_feat,vshift=vshift)
    mdist = torch.mean(torch.stack(dists,1),1)

    minval, minidx = torch.min(mdist,0)

    offset = vshift-minidx
    conf   = torch.median(mdist) - minval
    # return audio_frames, audio_signal, conf, direct_right_num, move_ratio
    return (conf > 5.5 and direct_right_num > 0.9 and move_ratio < 0.4)


def worker(job):
    src_tarfilepath, dst_tarfilepath = job
    print(f"Processing {src_tarfilepath} -> {dst_tarfilepath}")
    sync_model = SyncNetInstance()
    sync_model.loadParameters("data/syncnet_v2.model")
    err_msg = None
    try:
        # 如果字符串中有()的话，必须添加转义符号\
        src_tarfilepath = src_tarfilepath.replace("(","\(")
        src_tarfilepath = src_tarfilepath.replace(")","\)")

        dst_tarfilepath = dst_tarfilepath.replace("(","\(")
        dst_tarfilepath = dst_tarfilepath.replace(")","\)")

        dataset = wds.WebDataset(src_tarfilepath)

        with refile.smart_open(dst_tarfilepath, "wb") as wf:
            sink = wds.TarWriter(fileobj=wf,)
            for data in tqdm(dataset):
                key          = data["__key__"]
                # url          = data["__url__"]
                video_bytes  = data["mp4"]

                dwpose_result = data["dwpose_result.pyd"]
                dwpose_score = data["dwpose_score.pyd"]
                faces_bbox = data['faces_bbox.pyd']
                audio_signal = pickle.loads(data["audio_frames.pyd"]).reshape(-1,)
                sample_rate = float(data["sample_rate.str"])
                fps = float(data["fps.str"])
                dwpose_results = pickle.loads(dwpose_result)
                dwpose_scores = pickle.loads(dwpose_score)
                faces_bbox = pickle.loads(faces_bbox)
                try:
                    video_frames = get_clip_frames(video_bytes)
                    save_this_clip = get_dwpose_output(sync_model, video_frames, dwpose_results, dwpose_scores, faces_bbox, audio_signal, sample_rate, fps)
                except Exception as e:
                    traceback.print_exc()
                    save_this_clip = False
                
                if save_this_clip:
                    print("save this video")
                    sink.write(data)
                else:
                    print("Drop this video")
            sink.close()
        dataset.close()
    except Exception as e:
        traceback.print_exc()
        err_msg = e
    return src_tarfilepath, err_msg


if __name__ == "__main__":
    # python lip_fliter.py \
    #     --tarfile_dir Datasets/TalkingHead-1KH_Part2_dwpose_20240722 \
    #     --out_dir Datasets/TalkingHead-1KH_Part2FliterV2_20240816

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tarfile_dir', type=str, default="")
    parser.add_argument('--out_dir', type=str, default="")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    args = parser.parse_args()    
    
    tagfile_dir = args.tarfile_dir
    if args.end_idx == -1:
        tagfilepath_list = list(refile.smart_glob(
            refile.smart_path_join(tagfile_dir, "*.tar")))[args.start_idx:] #[args.start_idx:args.end_idx]
    else:
        tagfilepath_list = list(refile.smart_glob(
            refile.smart_path_join(tagfile_dir, "*.tar")))[args.start_idx:args.end_idx] #[args.start_idx:args.end_idx]
    out_dir = args.out_dir

    jobs = []
    for tarfilepath in tagfilepath_list:
        tarfilename = tarfilepath.split('/')[-1]
        dst_tarfilepath = refile.smart_path_join(out_dir, tarfilename)
        # skip
        if refile.smart_exists(dst_tarfilepath):
            continue
        if (tarfilepath, dst_tarfilepath) not in jobs:
            jobs.append((tarfilepath, dst_tarfilepath))

    for job in tqdm(jobs):
        src_tarfilepath, err_msg = worker(job)
        print(src_tarfilepath, err_msg)
        