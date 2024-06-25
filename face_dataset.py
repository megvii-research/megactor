from torch.utils.data import IterableDataset
import random
import cv2
# import torch.nn.functional as F
import torch
import numpy as np
from einops import rearrange, repeat
import math
import webdataset as wds
import imageio.v3 as iio
import json
import traceback
from typing import List
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import os
from animate.utils.util import crop_and_resize_tensor, crop_and_resize_tensor_with_face_rects, crop_move_face
from animate.utils.util import save_videos_grid, pad_image, generate_random_params, apply_transforms
import facer
from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor
from PIL import Image
import refile


class VideosIterableDataset(IterableDataset):
    
    def __init__(
        self,  
        data_dirs,
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
            "mp4_styled",
            "swapped.mp4"
        ]
        self.main_key = 'mp4'

        self.luma_thresh = 5.0
        self.min_face_thresh = 0.2

        self.warp_rate = warp_rate
        self.color_jit_rate = color_jit_rate
        self.use_swap_rate = use_swap_rate

        self.controlnet_usable = controlnet_usable
        self.color_BW_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).cpu().float()
        
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(resolution),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.face_detector = facer.face_detector('retinaface/mobilenet', device="cpu")
        self.face_detector.requires_grad_(False)
        self.dwpose_model = DenseDWposePredictor("cpu", resolution=self.resolution)

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
                raise NotImplementedError("仅支持输入以下路径:(1)以.tar结尾的Tar包路径; (2)文件夹路径")
        assert len(tarfile_path_list)>0, "没找到任何Tar包文件"
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

        self_frame_stride = self.frame_stride
        # print('self_frame_stride is', self_frame_stride)
        all_indices = list(range(0, n_frames, self_frame_stride))
        if len(all_indices) < self.video_length:
            frame_stride = n_frames // self.video_length
            assert (frame_stride != 0)
            all_indices = list(range(0, n_frames, frame_stride))
        
        rand_idx = random.randint(0, len(all_indices) - self.video_length)
        clip_indices = all_indices[rand_idx:rand_idx+self.video_length]
        return clip_indices


    def get_clip_frames(self, video_byte_sequence, ref_index=0) -> torch.Tensor:
        frames_swap = None
        decoded_sequences = list()
        ref_image = None
        for idx, video_bytes in enumerate(video_byte_sequence):
            if video_bytes is None:
                frames = None
            else:
                try:
                    file = iio.imopen(video_bytes, "r", plugin="pyav")
                except:
                    return None, None
                frames = file.read(index=...)  
                frames_real = []
                if idx == ref_index:
                    ref_image = torch.tensor(random.choice(frames)).permute(2, 0, 1)[None, ...]
                for frame in frames:
                    if frame.mean() > self.luma_thresh:
                        frames_real.append(frame)
                frames = np.array(frames_real, dtype=frames.dtype) 
                    
                n_frames = frames.shape[0]
                if n_frames < self.video_length:
                    return None, None
                else:
                    clip_indices = self.get_random_clip_indices(n_frames)
                    
                    frames = frames[clip_indices, ...]
                    frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
            if frames is not None:
                decoded_sequences.append(frames)
        return ref_image, decoded_sequences


    def aug_data(self, pixel_values, pixel_values_swap):
        pass


    def __iter__(self):
        while True:
            try:
                for data in self.wds_dataset:
                    try:
                        video = data[self.main_key]
                        other_video_bytes = []
                        for name in self.other_frames:
                            if name in data:
                                other_video_bytes.append(data[name])

                        ref_image, all_frames = self.get_clip_frames([video] + other_video_bytes)
                        if ref_image is None:
                            continue
                        start_frame = self.rng.integers(0, max(len(all_frames[0]) - self.video_length +1, 0))
                        all_frames = [frame[start_frame:start_frame + self.video_length, ...] for frame in all_frames]
                        frame, *other_frames = all_frames
                        if len(other_frames) > 0 and np.random.rand() < self.use_swap_rate:
                            swapped = random.choice(other_frames)
                        else:
                            swapped = frame

                        batched_ref_frame = torch.cat([ref_image, frame], dim=0)
                        faces = self.face_detector(batched_ref_frame)
                        if 'image_ids' not in faces.keys() or faces['image_ids'].numel() == 0:
                            continue
                        min_face_size, face_rects, bbox, batched_ref_frame  = crop_and_resize_tensor_with_face_rects(batched_ref_frame, faces, target_size=self.resolution)
                        _, _, _, swapped  = crop_and_resize_tensor_with_face_rects(swapped, faces, target_size=self.resolution)
                        if min_face_size is None:
                            continue
                        if min_face_size < self.min_face_thresh:
                            continue
                        ref_image, frame = batched_ref_frame[:1], batched_ref_frame[1:]

                        swapped_faces = self.face_detector(frame)
                        if 'image_ids' not in swapped_faces.keys() or swapped_faces['image_ids'].numel() == 0:
                            continue
                        swapped = crop_move_face(swapped, swapped_faces, target_size=self.resolution)
                        if np.random.rand() < self.warp_rate:
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

                        yield sample_dic
                    except Exception as e:
                        traceback.print_exc()
                        print('meet error for', e)
                        continue
            except Exception as e:
                traceback.print_exc()
                print('meet break error for', e)
                continue


if __name__ == "__main__":

    from PIL import Image

    import resource
    from tqdm import tqdm

    dataset = VideosIterableDataset(
        [
            '/data/data/VFHQ_webdataset_20240404'
        ],
        video_length = 4,
        resolution = [256,256],
        frame_stride = 1,
        shuffle        = True,
        resampled      = True,
    )

    dataloader = wds.WebLoader(
        dataset, 
        batch_size=4,
        shuffle=False,
        num_workers=32,
        collate_fn = None,
    ).with_length(len(dataset))
    from animate.utils.util import save_videos_grid
    cnt_num = 0
    for data in tqdm(dataloader):
        seq = [
            data["swapped"], 
            data["reference"][:, None].repeat(1, data["swapped"].shape[1], 1, 1, 1), 
            data["video"],
            data["concat_background"][:, :, None].repeat(1, data["swapped"].shape[1], 3, 1, 1), 
            data["concat_poses"].repeat(1, data["swapped"].shape[1], 1, 1, 1), 
            data["clip_conditions"].repeat(1, data["swapped"].shape[1], 1, 1, 1)
            ]
        samples_per_video = torch.cat(seq, dim=-2)
        samples_per_video = rearrange(samples_per_video, "b f c h w -> b c f h w")
        print('samples_per_video shape is', samples_per_video.shape, samples_per_video.min(), samples_per_video.max())
        save_videos_grid(samples_per_video, f"./show_data/{cnt_num}.gif", rescale=True if samples_per_video.min() < 0 else False)
        cnt_num += 1
        pass
 
    print("...")
    