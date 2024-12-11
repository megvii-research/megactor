import os
import glob
from tqdm import tqdm 
import numpy as np
import traceback
import imageio.v3 as iio
import webdataset as wds
from webdataset import gopen, gopen_schemes

import sys
sys.path.append('../../')

from controlnet_resource.dwpose import SimpleDWposeDetector

DISTANCE_TO_EDGE = 200


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


def get_dwpose_output(dwpose_model, frames_np):
    num_frames, *_ = frames_np.shape

    frames_dwpose_result = []
    frames_dwpose_score = []
    frames_face_bbox = []
    frames_exist_face_index = []
    eps = 0.01

    for frame_id in range(num_frames):
        frame_np = frames_np[frame_id]
        frame_dwpose_result, frame_dwpose_score, face_bbox = dwpose_model(frame_np) # 输入RGB而不是BGR
        candidates = frame_dwpose_result
        subsets = frame_dwpose_score

        frame_is_valid = False
        face_bbox = np.array([0., 0., 0., 0.])
        # 对所有帧筛选哪些帧不合法。
        # 训练时，等到数据选择时，遇到不合法的数据将直接被跳过
        # 合法帧只能有一个人脸，脸部关键点要超过60个，按照正方形裁剪脸后正方形分辨率要大于512，512
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
                if distance_to_edge > DISTANCE_TO_EDGE:
                    face_bbox = np.array([x0, y0, x1, y1])
                    frame_is_valid = True
                
                # 按照正方形裁剪脸后，正方形分辨率要大于400，400
                # 否则当前帧被标记为不合法

        frames_dwpose_result.append(frame_dwpose_result.astype(np.float32))
        frames_dwpose_score.append(frame_dwpose_score.astype(np.float32))
        frames_face_bbox.append(face_bbox.astype(np.float32))
        frames_exist_face_index.append(frame_is_valid)

    return {
        "dwpose_result"          : frames_dwpose_result,
        "dwpose_score"           : frames_dwpose_score,
        "faces_bbox"             : frames_face_bbox,
        "exist_face_frame_index" : frames_exist_face_index,
    }


def worker(job):
    src_tarfilepath, dst_tarfilepath = job
    print(f"Processing {src_tarfilepath} -> {dst_tarfilepath}")
    dwpose_model = SimpleDWposeDetector() #DWposeDetector()

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
                # print("data key is", data.keys())
                video_bytes  = data["mp4"] if "mp4" in data else data[".mp4"]

                try:
                    video_frames = get_clip_frames(video_bytes)
                    if len(video_frames) < 60:
                        continue
                    frames_face_info = get_dwpose_output(dwpose_model, video_frames)
                except Exception as e:
                    traceback.print_exc()
                    frames_face_info = {}
                
                # dwpose_result = frames_face_info.get("dwpose_result",          [])
                # dwpose_score = frames_face_info.get("dwpose_score",           [])
                # faces_bbox = frames_face_info.get("faces_bbox",             [])
                exist_face_frame_index = frames_face_info.get("exist_face_frame_index", [])
                num_frame_valid = 0
                for frame_valid in exist_face_frame_index:
                    if frame_valid:
                        num_frame_valid += 1
                
                if num_frame_valid / len(video_frames) <= 0.9:
                    # 少于一半的帧不符合要求，则当前视频不要
                    print(f"drop this video for {num_frame_valid / len(video_frames)}")
                    continue

                print("save this video")
                sink.write({
                    "__key__": key,
                    # "__url__": url,
                    "mp4"                         : video_bytes,
                    "dwpose_result.pyd"           : frames_face_info.get("dwpose_result",          []),
                    "dwpose_score.pyd"            : frames_face_info.get("dwpose_score",           []), 
                    "faces_bbox.pyd"              : frames_face_info.get("faces_bbox",             []),
                    "exist_face_frame_index.pyd"  : frames_face_info.get("exist_face_frame_index", []),
                    "fps.str": data["fps.str"] if "fps.str" in data else data[".fps.str"],
                    "sample_rate.str": data["sample_rate.str"] if "sample_rate.str" in data else data[".sample_rate.str"],
                    "audio_frames.pyd": data["audio_frames.pyd"] if "audio_frames.pyd" in data else data[".audio_frames.pyd"],
                })
            sink.close()
        dataset.close()
    except Exception as e:
        traceback.print_exc()
        err_msg = e
    return src_tarfilepath, err_msg


if __name__ == "__main__":
    # dwpose_fliter_data_strong_audio.py \
    #     --tarfile_dir Datasets/VoxCeleb_part2_20240720 \
    #     --out_dir Datasets/VoxCeleb_part2_dwpose_20240720 \
    #     --distance_to_edge 200 

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tarfile_dir', type=str, default="")
    parser.add_argument('--out_dir', type=str, default="")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1) 
    parser.add_argument('--distance_to_edge', type=int, default=200) # 筛选时的分辨率阈值，保证图像清晰，如果已知视频很清晰则可以适当降低该阈值
    args = parser.parse_args()    
    
    DISTANCE_TO_EDGE = args.distance_to_edge
    tagfile_dir = args.tarfile_dir
    if args.end_idx == -1:
        tagfilepath_list = list(glob.glob(
            os.path.join(tagfile_dir, "*.tar")))[args.start_idx:] #[args.start_idx:args.end_idx]
    else:
        tagfilepath_list = list(glob.glob(
            os.path.join(tagfile_dir, "*.tar")))[args.start_idx:args.end_idx] #[args.start_idx:args.end_idx]
    out_dir = args.out_dir

    jobs = []
    for tarfilepath in tagfilepath_list:
        tarfilename = tarfilepath.split('/')[-1]
        dst_tarfilepath = os.path.join(out_dir, tarfilename)
        # skip
        if os.path.exists(dst_tarfilepath):
            continue
        if (tarfilepath, dst_tarfilepath) not in jobs:
            jobs.append((tarfilepath, dst_tarfilepath))

    for job in tqdm(jobs):
        src_tarfilepath, err_msg = worker(job)
        print(src_tarfilepath, err_msg)
        