import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from animatediff.utils.util import zero_rank_print
import json
from PIL import Image
import torch
from decord import VideoReader

from einops import rearrange

import io, csv, math, random
# from util import zero_rank_print


class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
    ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']

        video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]

        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample


class PexelsDataset(Dataset):
    """
    load video-only data, and get dwpose condition
    """

    def __init__(
            self,
            json_path,
            sample_size=(768, 512), sample_stride=1, sample_n_frames=16, is_test=False
    ):
        print("load video-only data, and get dwpose condition")
        if not isinstance(json_path, list):
            zero_rank_print(f"loading annotations from {json_path} ...")
            self.dataset = json.load(open(json_path))
        else:
            self.dataset = json_path

        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames

        self.sample_size = sample_size

        if not is_test:
            self.pixel_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
        else:
            self.pixel_transforms = transforms.Compose([
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])

    def get_batch(self, idx):
        video_dir = self.dataset[idx]
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)

        image_np = video_reader.get_batch(batch_index).asnumpy()
        pixel_values = torch.from_numpy(image_np).permute(0, 3, 1, 2).contiguous()
        pixel_values = (pixel_values / 255.0 - 0.5) * 2
        del video_reader

        return pixel_values, image_np

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        while True:
            try:
                pixel_values, image_np = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)

        pixel_values = resize_and_crop(image_np, sample_size=self.sample_size)
        pixel_values = self.pixel_transforms(pixel_values)

        sample = dict(pixel_values=pixel_values)
        return sample


def resize_and_crop(images, sample_size=(768, 512)):
    image_np = []

    for image in images:
        image = Image.fromarray(image)
        # Determine if width is larger than height or vice versa
        if image.width > image.height:
            aspect_ratio = image.width / image.height
            new_width = int(sample_size[0] * aspect_ratio)
            resize = transforms.Resize((sample_size[0], new_width))
        else:
            aspect_ratio = image.height / image.width
            new_height = int(sample_size[1] * aspect_ratio)
            resize = transforms.Resize((new_height, sample_size[1]))

        # Apply the resize transformation
        image = resize(image)

        # Calculate padding
        pad_left = (sample_size[1] - image.width) // 2
        pad_right = sample_size[1] - image.width - pad_left
        pad_top = (sample_size[0] - image.height) // 2
        pad_bottom = sample_size[0] - image.height - pad_top

        # Apply padding
        padding = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0)
        image = padding(image)

        image_np.append(np.array(image))

    image_np = np.stack(image_np)

    pixel_values = torch.from_numpy(image_np).permute(0, 3, 1, 2).contiguous()
    pixel_values = pixel_values / 255.

    return pixel_values


def get_pose_conditions(image_np, dwpose_model=None):
    dwpose = dwpose_model

    num_frames = image_np.shape[0]
    dwpose_conditions = []

    for frame_id in range(num_frames):
        pil_image = Image.fromarray(image_np[0])
        dwpose_image = dwpose(pil_image, output_type='np')
        dwpose_image = torch.tensor(dwpose_image).unsqueeze(0)
        dwpose_conditions.append(dwpose_image)

    return torch.cat(dwpose_conditions, dim=0)


if __name__ == "__main__":
    from util import save_videos_grid

    dataset = PexelsDataset(
        json_path="/work00/magic_animate_unofficial/fashion_dataset.json")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16)
    for idx, batch in enumerate(dataloader):
        print(batch['pixel_values'].shape)
        for i in range(batch['pixel_values'].shape[0]):
            save_videos_grid(batch['pixel_values'][i:i + 1].permute(0, 2, 1, 3, 4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)

        break
