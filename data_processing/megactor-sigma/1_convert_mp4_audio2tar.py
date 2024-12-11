import os
import traceback
import numpy as np
from tqdm import tqdm 

import librosa
import torchvision
import imageio.v3 as iio
from moviepy.editor import VideoFileClip

import webdataset as wds
from webdataset import gopen, gopen_schemes


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


def get_dwpose_output(frames_np, dwpose_results, dwpose_scores):
    num_frames, *_ = frames_np.shape

    frames_face_bbox = []
    frames_exist_face_index = []
    eps = 0.01

    for frame_id in tqdm(range(num_frames)):
        frame_np = frames_np[frame_id]
        candidates = dwpose_results[frame_id]
        subsets = dwpose_scores[frame_id]

        frame_is_valid = False
        face_bbox = np.array([0., 0., 0., 0.])
        if len(candidates) == 1 and len(subsets) == 1:
            # 只能有一个人脸
            xs, ys = [], []
            for j in range(24, 92): # left eyes
                if subsets[0, j] < 0.3:
                    continue
                x, y = candidates[0, j]
                if x < eps or y < eps:
                    continue
                xs.append(x)
                ys.append(y)
            
            if len(xs) < 60 or len(ys) < 60:
                # 脸部关键点要全
                frame_is_valid = False
            else:
                H, W, C = frame_np.shape
                xs, ys = np.array(xs), np.array(ys)
                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

                w, h = (x1 - x0), (y1 - y0)
                x_c, y_c = (x0 + x1) / 2, (y0 + y1) / 2
                expand_dis = max(w, h)
                left, right = max(x_c - expand_dis * 1.25, 0), min(x_c + expand_dis * 1.25, W)
                bottom, top = max(y_c - expand_dis * 1.5, 0), min(y_c + expand_dis, H)
                x_c, y_c = (left + right) / 2, (bottom + top) / 2
                distance_to_edge = min(x_c - left, right - x_c, y_c - bottom, top - y_c)
                if distance_to_edge > 200:
                    face_bbox = np.array([x0, y0, x1, y1])
                    frame_is_valid = True
                
                # 按照正方形裁剪脸后，正方形分辨率要大于512，512
                # 否则当前帧被标记为不合法

        frames_face_bbox.append(face_bbox.astype(np.float32))
        frames_exist_face_index.append(frame_is_valid)

    return {
        "dwpose_result"          : dwpose_results,
        "dwpose_score"           : dwpose_scores,
        "faces_bbox"             : frames_face_bbox,
        "exist_face_frame_index" : frames_exist_face_index,
    }


def read_video_audio(file_path):
    # 使用 moviepy 读取视频和音频数据
    clip = VideoFileClip(file_path)
    fps = clip.fps
    sample_rate = clip.audio.fps
    with iio.imopen(file_path, "r", plugin="pyav") as file:
        video = file.read(index=...)
    audio_path = file_path.replace(".mp4", ".wav")

    if not os.path.exists(audio_path):
        clip.audio.write_audiofile(audio_path, fps = sample_rate)
    audio_signal, sample_rate = librosa.load(audio_path, sr=None)

    return video, audio_signal, fps, sample_rate

def worker(job):
    src_file_list, dst_filepath = job
    print(f"Processing {len(src_file_list)} -> {dst_filepath}")
    err_msg = None
    try:
        with refile.smart_open(dst_filepath, "wb") as wf:
            sink = wds.TarWriter(fileobj=wf,)
            for file_path in tqdm(src_file_list):
                try:
                    frames, audio_signal, video_fps, audio_sampling_rate = read_video_audio(file_path)
                    key = file_path.split("/")[-2] + os.path.basename(file_path)[:-4]
                    sink.write({
                        "__key__": key,
                        "mp4": iio.imwrite("<bytes>", frames, extension='.mp4', plugin="pyav", codec="h264", fps=int(video_fps)),
                        "audio_frames.pyd": audio_signal.reshape(-1,),
                        "sample_rate.str": str(audio_sampling_rate),
                        "fps.str": str(video_fps),
                    })
                    del frames, audio_signal
                except Exception as e:
                    traceback.print_exc()
            sink.close()
    except Exception as e:
        traceback.print_exc()
        err_msg = e
    return len(src_file_list), err_msg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', type=str, default="")
    parser.add_argument('--out_dir', type=str, default="")
    parser.add_argument('--start_idx', type=int, default=0) 
    parser.add_argument('--end_idx', type=int, default=-1)
    args = parser.parse_args()    
    
    file_dir = args.file_dir
    filepath_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith('.mp4'):
                filepath_list.append(os.path.join(root, file))
    
    out_dir = args.out_dir

    jobs = []
    print("total deal file number is", len(filepath_list))

    threld_file_size = 512 * 1024 * 1024
    threld_file_number = 1000
    cur_file_size = 0
    cur_file_list = []
    files_bundle_num = args.start_idx
    for filepath in filepath_list:
        tarfilename = filepath.split('/')[-1]
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            continue
        cur_file_size += file_size

        src_filepath = filepath.replace("(","\(")
        src_filepath = src_filepath.replace(")","\)")

        cur_file_list.append(src_filepath)
        if cur_file_size > threld_file_size or len(cur_file_list) > threld_file_number:
            print(f"cur_file_size is {cur_file_size / 1024 / 1024} MB")
            out_path = os.path.join(out_dir, f"archive_{files_bundle_num}.tar")
            if not os.path.exists(out_path):
                jobs.append((cur_file_list, os.path.join(out_dir, f"archive_{files_bundle_num}.tar")))
            cur_file_size = 0
            files_bundle_num += 1
            cur_file_list = []

    if len(cur_file_list) != 0:
        out_path = os.path.join(out_dir, f"archive_{files_bundle_num}.tar")
        if not os.path.exists(out_path):
            jobs.append((cur_file_list, os.path.join(out_dir, f"archive_{files_bundle_num}.tar")))
        cur_file_size = 0
        files_bundle_num += 1
        cur_file_list = []
    
    print("number jobs is", len(jobs))

    for job in tqdm(jobs):
        src_filepath, err_msg = worker(job)
        print(src_filepath, err_msg)
    