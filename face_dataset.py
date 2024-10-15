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
import os
from animate.utils.util import crop_and_resize_tensor, crop_and_resize_tensor_with_face_rects, crop_move_face, crop_and_resize_tensor_small_faces
from animate.utils.util import save_videos_grid, pad_image, generate_random_params, apply_transforms, save_videos_grid_audio
import facer
from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor
from PIL import Image
from _preprocess import VideoTransforms


class VideosIterableDataset(IterableDataset):
    def __init__(
        self,  
        data_dirs,
        preprocess_function,
        decode_function='get_clip_frames',
        batch_size=1,
        video_length=16,
        resolution=[512, 512],
        frame_stride=1,
        dataset_length=100000,
        shuffle = True,
        resampled = True,
        controlnet_usable = False,
        crop_face_center = False,
        return_origin = False,
        concat_noise_image_type = "origin",
        clip_image_type = "background",
        warp_rate=0.25,
        color_jit_rate=0.5,
        use_swap_rate=0.5
    ):
        self.tarfilepath_list = self.get_tarfilepath_list(data_dirs)
        self.wds_shuffle      = shuffle
        self.wds_resampled    = resampled
        self.wds_dataset = self.get_webdataset()
        self.decode_function = decode_function
        self.preprocess_function = preprocess_function

        #.batched(batch_size, collation_fn=collate_fn)
        
        self.video_length     = video_length if video_length > 1 else 1
        self.batch_size = batch_size
        self.frame_stride     = frame_stride
        self.resolution       = resolution
        self.dataset_length   = int(dataset_length)
        self.rng = np.random.default_rng()
        self.crop_face_center = crop_face_center
        self.return_origin = return_origin

        self.concat_noise_image_type = concat_noise_image_type
        self.clip_image_type = clip_image_type

        self.other_frames = [
            # "mp4_styled",
            "swapped.mp4"
        ]
        self.main_key = 'mp4'

        self.luma_thresh = 5.0
        self.min_face_thresh = 0.2

        self.scale_factor = [1.25, 1.25, 1.5, 1.0] if video_length == 1 else [0.9, 0.9, 1., 0.8]
        self.left_scale, self.right_scale, self.top_scale, self.bottom_scale = self.scale_factor

        self.pixel_transforms = VideoTransforms(p_flip=0.)
        self.standard_fps = 8
        self.standard_sample_rate = 16000

        self.warp_rate = warp_rate
        self.color_jit_rate = color_jit_rate
        self.use_swap_rate = use_swap_rate

        self.controlnet_usable = controlnet_usable
        self.color_BW_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).cpu().float()
        
        # self.pixel_transforms = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     # transforms.Resize(resolution),
        #     # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        # ])

        self.face_detector = facer.face_detector('retinaface/mobilenet', device="cpu")
        self.face_detector.requires_grad_(False)
        self.dwpose_model = DenseDWposePredictor("cpu", resolution=self.resolution)

    from _preprocess import get_compact_frames, get_clip_frames, crop_small_area, \
    crop_area, crop_patch, get_packed_frames, crop_flex_area, \
    crop_audio_240722, crop_audio_240728, crop_audio_area

    def get_tarfilepath_list(self, data_dirs):
        tarfile_path_list = []
        for data_dir in data_dirs:
            if os.path.isdir(data_dir):
                file_path_list = os.listdir(data_dir)
                tarfile_path_list += [
                    os.path.join(data_dir, file_path)
                    for file_path in file_path_list if file_path.endswith(".tar")]
            elif data_dir.endswith(".tar"):
                tarfile_path_list.append(data_dir)
            else:
                raise NotImplementedError("Only .tar and directories containing .tar ares supported.")
        assert len(tarfile_path_list)>0, "No tar file found"
        print(f'finish get tarfile_path_list len is {len(tarfile_path_list)}')
        return tarfile_path_list

    def get_webdataset(self, ):

        dataset = wds.WebDataset(self.tarfilepath_list, resampled=self.wds_resampled)
        if self.wds_shuffle:
            dataset  = dataset.shuffle(100)
        return dataset
 

    def __len__(self, ):
        return self.dataset_length


    def get_random_clip_indices(self, n_frames:int) -> List[int]:
    
        frame_stride = self.frame_stride
        all_indices = list(range(0, n_frames, frame_stride))
        if len(all_indices) < self.video_length:
            frame_stride = n_frames // self.video_length
            assert (frame_stride != 0)
            all_indices = list(range(0, n_frames, frame_stride))
        
        rand_idx = random.randint(0, len(all_indices) - self.video_length)
        clip_indices = all_indices[rand_idx:rand_idx+self.video_length]
        return clip_indices, frame_stride


    def aug_data(self, pixel_values, pixel_values_swap):
        pass


    def __iter__(self):
        while True:
            try:
                for data in self.wds_dataset:
                    # key          = data["__key__"]
                    # url          = data["__url__"]
                    try:
                        # video = data[self.main_key]
                        # other_video_bytes = []
                        # for name in self.other_frames:
                        #     if name in data:
                        #         other_video_bytes.append(data[name])

                        ret = getattr(self, self.decode_function)(data)
                        if ret is None:
                            continue
                        # if ref_image is None:
                        #     continue
                        # start_frame = self.rng.integers(0, max(len(all_frames[0]) - self.video_length +1, 0))
                        # all_frames = [frame[start_frame:start_frame + self.video_length, ...] for frame in all_frames]
                        # frame, *other_frames = all_frames
                        # if len(other_frames) > 0 and np.random.rand() < self.use_swap_rate:
                        #     swapped = random.choice(other_frames)
                        # else:
                        #     swapped = frame

                        if not isinstance(self.preprocess_function, str):
                            sample_dict = getattr(self, random.choice(self.preprocess_function))(**ret)
                        else:
                            sample_dict = getattr(self, self.preprocess_function)(**ret)
                        if sample_dict is None:
                            continue

                        yield sample_dict

                    except Exception as e:
                        traceback.print_exc()
                        print('meet error for', e)
                        continue
            except Exception as e:
                traceback.print_exc()
                print('meet break error for', e)
                continue

def train_collate_fn(examples):
    images = torch.stack([example["videos"] for example in examples])
    images = images.to(memory_format=torch.contiguous_format).float()
    masked_images = torch.stack([example["masked_image"] for example in examples])
    masked_images = masked_images.to(memory_format=torch.contiguous_format).float()
    masks = torch.stack([example["mask"] for example in examples])
    masks = masks.to(memory_format=torch.contiguous_format).float()
    caption_tokens = torch.stack([example["caption_token"] for example in examples])
    caption_tokens = caption_tokens.to(memory_format=torch.contiguous_format).long()
    caption_tokens_2 = torch.stack([example["caption_token_2"] for example in examples])
    caption_tokens_2 = caption_tokens_2.to(memory_format=torch.contiguous_format).long()
    return {
        "image"           : images, 
        "masked_image"    : masked_images, 
        "mask"            : masks, 
        "caption_token"   : caption_tokens,
        "caption_token_2" : caption_tokens_2,
    }



if __name__ == "__main__":

    from PIL import Image

    import resource
    from tqdm import tqdm

    dataset = VideosIterableDataset(
        [
            # '/data/data/VFHQ_webdataset_20240404/group410.tar'
            # "s3://radar/yangshurong/Datasets/HDTF_20240626/"
            # "s3://public-datasets/Datasets/Videos/processed/HDTF_20240704_dwpose_facebbox_facefusion-HQsource_short-video",

            "s3://radar/yangshurong/Datasets/Zoo_xpose_newKey_20240806/000.tar",

            # "s3://radar/yangshurong/Datasets/CHTF_dwpose20240726/",
            # "s3://radar/yangshurong/Datasets/LawExam_dwpose_20240725/",
            # "s3://radar/yangshurong/Datasets/TalkingHead-1KH_Part3_dwpose_20240722",
            # "s3://radar/yangshurong/Datasets/TalkingHead-1KH_Part2_dwpose_20240722",
            # "s3://radar/yangshurong/Datasets/MultiTalk_0_5_20240719/",
            # "s3://radar/yangshurong/Datasets/MultiTalk_6_11_20240719",
        ],
        "crop_audio_240728",
        # "crop_flex_area",
        # decode_function='get_packed_frames',
        decode_function='get_compact_frames',
        video_length = 32,
        resolution = [256,256],
        frame_stride = 1,
        shuffle        = True,
        resampled      = True,
    )

    dataloader = wds.WebLoader(
        dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn = None,
    ).with_length(len(dataset))
    from animate.utils.util import save_videos_grid
    cnt_num = 0
    for data in tqdm(dataloader):
        seq = [
            # data["swapped"], 
            # data["reference"][:, None].repeat(1, data["swapped"].shape[1], 1, 1, 1), 
            data["video"],
            # data["frames_eye"],
            # data["frames_eye_mouth"],
            # data["dwpose_results"],
            # data["mask_mouth"],

            # data["concat_background"][:, :, None].repeat(1, data["swapped"].shape[1], 3, 1, 1), 
            # data["concat_poses"].repeat(1, data["swapped"].shape[1], 1, 1, 1), 
            # data["clip_conditions"].repeat(1, data["swapped"].shape[1], 1, 1, 1)
            ]
        # print([i.shape for i in seq])
        samples_per_video = torch.cat(seq, dim=-2)
        samples_per_video = rearrange(samples_per_video, "b f c h w -> b c f h w")
        # print('samples_per_video shape is', samples_per_video.shape, samples_per_video.min(), samples_per_video.max())
        audio_signal = data['audio_signal']
        # print(audio_signal.shape, data['start_time'], data['end_time'])
        # audio_signal = audio_signal[:, int(max(0, dataset.standard_sample_rate * (data['start_time']))): int(min(audio_signal.shape[1], dataset.standard_sample_rate * (data['end_time'])))]
        # print(audio_signal.shape)
        # save_videos_grid(samples_per_video, f"./show_data/{cnt_num}.mp4", rescale=True if samples_per_video.min() < 0 else False, fps=15)
        save_videos_grid_audio(
                samples_per_video,  audio_signal, f"/data/show_data/{cnt_num}.mp4", fps=data['fps'])
        cnt_num += 1
        pass
 
    print("...")
    
