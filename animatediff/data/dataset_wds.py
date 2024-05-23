from torch.utils.data import IterableDataset
import os
import random
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
from glob import glob
import tarfile
from megfile import smart_open as open
from megfile import smart_glob
import msgpack
from einops import rearrange, repeat
import math
# from functools import lru_cache
import webdataset as wds
import imageio.v3 as iio
import json
import traceback
from typing import List
import torchvision.transforms.functional as F
import torch.nn.functional as TF
import megfile
import torchvision.transforms as transforms
from transformers import CLIPProcessor
from PIL import Image, ImageDraw
from torch.utils.data.dataset import Dataset
from decord import VideoReader


from controlnet_aux import DWposeDetector
det_config = '/work00/magic_animate_unofficial/controlnet_aux/src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
det_ckpt = '/models00/controlnet_aux/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
pose_config = '/work00/magic_animate_unofficial/controlnet_aux/src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py'
pose_ckpt = '/models00/controlnet_aux/dw-ll_ucoco_384.pth'

import facer

def crop_and_resize(frame, target_size=None, crop_rect=None):
    height, width = frame.size
    
    if crop_rect is not None:
        left, top, right, bottom = crop_rect
        face_w = right-left
        face_h = bottom-top
        padding = max(face_w, face_h) // 2
        if face_w < face_h:
            left = left - (face_h-face_w)//2
            right = right + (face_h-face_w)//2
        else:
            top = top - (face_h-face_w)//2
            bottom = bottom + (face_h-face_w)//2
        left, top, right, bottom = left-padding, top-padding, right+padding, bottom+padding 
    else:
        short_edge = min(height, width)
        width, height = frame.size
        top = (height - short_edge) // 2
        left = (width - short_edge) // 2
        right = (width + short_edge) // 2
        bottom = (height + short_edge) // 2
    frame_cropped = frame.crop((left, top, right, bottom))
    if target_size is not None:
        frame_resized = frame_cropped.resize(target_size, Image.ANTIALIAS)
    else:
        return frame_cropped
    return frame_resized

DEBUG = os.environ.get('DEBUG')
if DEBUG:
    print('DATASET DEBUG MODE')
else:
    DEBUG = 0

# 放到data_utils里
def load_msgpack_list(file_path: str):
    loaded_data = []
    with open(file_path, 'rb') as f:
        unpacker = msgpack.Unpacker(f,strict_map_key = False)
        for unpacked_item in unpacker:
            loaded_data.append(unpacked_item)
        return loaded_data


# @lru_cache(maxsize=128)
def load_tar(p):
    return tarfile.open(fileobj=open(p, 'rb'))


def load_img_from_tar(img_path):
    tar_fname,img_fname = img_path.rsplit("/",1)
    tar_obj = load_tar(tar_fname)
    img = Image.open(tar_obj.extractfile(img_fname)).convert("RGB")
    return np.array(img)

def read_remote_img(p):
    with open(p, 'rb') as rf:
        return Image.open(rf).convert("RGB")

def gen_landmark_control_input(img_tensor, landmarks):
    cols = torch.tensor([int(y) for x,y in landmarks])
    rows = torch.tensor([int(x) for x,y in landmarks])
    img_tensor = img_tensor.index_put_(indices=(cols, rows), values=torch.ones(106))
    return img_tensor.unsqueeze(-1)


class S3VideosIterableDataset(IterableDataset):
    def __init__(
        self,  
        data_dirs,
        video_length=16,
        resolution=[512, 512],
        frame_stride=1,
        dataset_length=100000,
        is_image=False,
        shuffle = True,
        resampled = True,
        endpoint_url = "http://oss.i.shaipower.com:80",

        clip_model_path="openai/clip-vit-base-patch32",
        is_train=True,
        is_det_face=False,
    ):
        self.tarfilepath_list = self.get_tarfilepath_list(data_dirs)
        self.wds_shuffle      = shuffle
        self.wds_resampled    = resampled
        self.endpoint_url     = endpoint_url
        self.wds_dataset      = self.get_webdataset()
        
        self.video_length     = video_length
        self.frame_stride     = frame_stride
        self.resolution       = resolution
        self.is_image         = is_image
        self.dataset_length   = int(dataset_length)
        self.is_det_face      = is_det_face

        ######################### animateanyone #########################        
        self.is_train = is_train
        self.spilt = 'train' if self.is_train else 'test'
        

        sample_size = tuple(resolution) if not isinstance(resolution, int) else (resolution, resolution)
        self.pixel_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize([sample_size[0],sample_size[0]]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
                
        # self.clip_image_processor = CLIPProcessor.from_pretrained(clip_model_path,local_files_only=False)
        # self.dwpose_model = DWposeDetector(
        #     det_config=det_config,
        #     det_ckpt=det_ckpt,
        #     pose_config=pose_config,
        #     pose_ckpt=pose_ckpt,
        # )
        if self.is_det_face:
            self.face_detector = facer.face_detector('retinaface/mobilenet', device="cpu")



    def get_tarfilepath_list(self, data_dirs):
        tarfile_path_list = []
        for data_dir in data_dirs:
            if megfile.smart_isdir(data_dir):
                file_path_list = megfile.smart_listdir(data_dir)
                tarfile_path_list += [
                    megfile.smart_path_join(data_dir, file_path)
                    for file_path in file_path_list if file_path.endswith(".tar")]
            elif data_dir.endswith(".tar"):
                tarfile_path_list.append(data_dir)
            else:
                print('data_dir:', data_dir)
                raise NotImplementedError("仅支持输入以下路径:(1)以.tar结尾的Tar包路径; (2)文件夹路径")
        assert len(tarfile_path_list)>0, "没找到任何Tar包文件"
        return tarfile_path_list


    def get_webdataset(self, ):
        url_list = []
        for tarfilepath in self.tarfilepath_list:
            if tarfilepath.startswith("s3://") or tarfilepath.startswith("tos://"):
                tarfilepath = tarfilepath.replace("tos://", "s3://")
                url_list.append(
                    f"pipe: aws --endpoint-url={self.endpoint_url} s3 cp {tarfilepath} -"
                )
            else:
                url_list.append(tarfilepath)
        
        dataset = wds.WebDataset(url_list, resampled=self.wds_resampled)
        if self.wds_shuffle:
            dataset  = dataset.shuffle(100)
        return dataset


    def __len__(self, ):
        return self.dataset_length


    def get_random_clip_indices(self, n_frames:int) -> List[int]:
        all_indices = list(range(0, n_frames, self.frame_stride))
        # all_indices = np.linspace(0, n_frames - 1, self.frame_stride, dtype=int).tolist()
        # print('all_indices', all_indices)
        if len(all_indices) < self.video_length:
            frame_stride = n_frames // self.video_length
            assert (frame_stride != 0)
            all_indices = list(range(0, n_frames, frame_stride))
        
        rand_idx = random.randint(0, len(all_indices) - self.video_length)
        clip_indices = all_indices[rand_idx:rand_idx+self.video_length]
        # print('clip_indices:', clip_indices)

        # frame_stride = self.frame_stride
        # all_frames = list(range(0, n_frames, frame_stride))
        # if len(all_frames) < self.video_length:  # recal a max fs
        #     frame_stride = n_frames // self.video_length
        #     assert (frame_stride != 0)
        #     all_frames = list(range(0, n_frames, frame_stride))

        # # select a random clip
        # rand_idx = random.randint(0, len(all_frames) - self.video_length)
        # clip_indices = all_frames[rand_idx:rand_idx+self.video_length]
        return clip_indices


    def get_clip_frames(self, video_bytes:bytes) -> torch.Tensor:
        frames = []
        with iio.imopen(video_bytes, "r", plugin="pyav") as file:
            n_frames = file.properties().shape[0]
            assert n_frames >= self.video_length, f"len(VideoClip) < {self.video_length}"
            clip_indices = self.get_random_clip_indices(n_frames)
            frames = file.read(index=...)  # np.array, [n_frames,h,w,c],  rgb, uint8
            frames = frames[clip_indices, ...]
            frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
            # for _, idx in enumerate(clip_indices):
            #     frame_np = file.read(index=idx) # np.array, hwc, rgb, uint8
            #     frames.append(frame_np)
            # frames = torch.tensor(np.array(frames)).permute(0, 3, 1, 2).float() # tchw
        return frames

    # Center Crop
    def center_crop(self, frames:torch.Tensor):
        new_height, new_width = self.resolution 
        height, width = frames.shape[-2:]
        short_edge = min(height, width)
        size = (short_edge,short_edge)
        # # RGB mode
        # t, c, h, w  = frames.shape
        crops = F.center_crop(frames, size)
        return crops


    def get_conditions(self, frames_tensors):
        # for training
        value_dict = {}
        cond_aug = 0.02
        value_dict["motion_bucket_id"] = 127
        value_dict["fps_id"] = 6
        value_dict["cond_aug"] = cond_aug
        anchor_image = frames_tensors[:1,...]
        value_dict["cond_frames_without_noise"] = anchor_image
        value_dict["cond_frames"] = anchor_image + cond_aug * torch.randn_like(anchor_image)
        value_dict["cond_aug"] = cond_aug

        keys = [ 'motion_bucket_id', 'fps_id', 'cond_aug', 'cond_frames','cond_frames_without_noise']
        N = [1, self.video_length]
        for key in keys:
            if key == "fps_id":
                value_dict[key] = (
                    torch.tensor([value_dict["fps_id"]])
                    .repeat(int(math.prod(N)))
                )
            elif key == "motion_bucket_id":
                value_dict[key] = (
                    torch.tensor([value_dict["motion_bucket_id"]])
                    .repeat(int(math.prod(N)))
                )
            elif key == "cond_aug":
                value_dict[key] = repeat(
                    torch.tensor([value_dict["cond_aug"]]),
                    "1 -> b",
                    b=math.prod(N),
                )
            elif key == "cond_frames":
                value_dict[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=math.prod(N))
            elif key == "cond_frames_without_noise":
                value_dict[key] = repeat(
                    value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=math.prod(N)
                )
            else:
                value_dict[key] = value_dict[key]
        
            value_dict['num_video_frames'] = self.video_length
            value_dict['image_only_indicator'] = torch.zeros(1, self.video_length)
        return value_dict

    def get_clip_frames_and_conditions(self, video_bytes:bytes, meta_dic:dict) -> torch.Tensor:
        frames = []
        with iio.imopen(video_bytes, "r", plugin="pyav") as file:
            n_frames = file.properties().shape[0]
            assert n_frames >= self.video_length, f"len(VideoClip) < {self.video_length}"
            clip_indices = self.get_random_clip_indices(n_frames)
            frames = file.read(index=...)  # np.array, [n_frames,h,w,c],  rgb, uint8
            # pixel_values: train objective
            # pixel_values_pose: corresponding pose
            # clip_ref_image: processed reference clip image
            # pixel_values_ref_img: ReferenceNet image
            
            # get pixel_values
            pixel_values = frames[clip_indices, ...]

            # face detect and crop
            if self.is_det_face:
                img_for_face_det = torch.tensor(pixel_values[0]).to(torch.uint8).unsqueeze(0).permute(0, 3, 1, 2)
                with torch.inference_mode():
                    faces = self.face_detector(img_for_face_det)
                    assert faces['image_ids'].numel() > 0
                    face_rect = faces['rects'][0].numpy()
                    pixel_values = [np.array(crop_and_resize(Image.fromarray(c), target_size=None, crop_rect=face_rect)) for c in pixel_values]
                    pixel_values = np.array(pixel_values)

            # get label from json
            # pixel_values_ldmk = np.zeros_like(pixel_values)
            # landmarks = meta_dic['tag']['faces'][0]['landmark_2d_106']
            # radius = 5
            # for frame_index in clip_indices:
            #     frame = pixel_values_ldmk[frame_index]  
            #     image = Image.fromarray(frame)  
            #     draw = ImageDraw.Draw(image)  
            #     for (x, y) in landmarks:
            #         draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='white', outline='white')
            #     pixel_values_ldmk[frame_index] = np.array(image)
            # print('pixel_values_ldmk', pixel_values_ldmk.shape)

            
            # pixel_values = torch.tensor(pixel_values).permute(0, 3, 1, 2).float()
            # for _, idx in enumerate(clip_indices):
            #     frame_np = file.read(index=idx) # np.array, hwc, rgb, uint8
            #     frames.append(frame_np)
            # frames = torch.tensor(np.array(frames)).permute(0, 3, 1, 2).float() # tchw

            # get dwpose
            # dwpose_conditions = []
            # for pixel_value in pixel_values:
            #     pil_image = Image.fromarray(pixel_value)
            #     dwpose_image = self.dwpose_model(pil_image, output_type='np')
            #     dwpose_conditions.append(dwpose_image)
            # pixel_values_pose = np.array(dwpose_conditions)
            
            # get clip_ref_image
            ref_img_idx = random.randint(0, n_frames - 1)
            ref_img = frames[ref_img_idx]
            # ref_img_idx = random.randint(0, self.video_length - 1)
            # ref_img = pixel_values[ref_img_idx]
            # face detect and crop
            if self.is_det_face:
                img_for_face_det = torch.tensor(ref_img).to(torch.uint8).unsqueeze(0).permute(0, 3, 1, 2)
                with torch.inference_mode():
                    faces = self.face_detector(img_for_face_det)
                    assert faces['image_ids'].numel() > 0
                    face_rect = faces['rects'][0].numpy()
                    ref_img = np.array(crop_and_resize(Image.fromarray(ref_img), target_size=None, crop_rect=face_rect))

            assert ref_img.mean() > 10
                

            # clip_ref_image = torch.zeros(pixel_values.shape[0], 1, 3, 224, 224)
            # ref_img_pil = Image.fromarray(ref_img)
            # clip_ref_image = self.clip_image_processor(images=ref_img_pil, return_tensors="pt").pixel_values

            # get pixel_values_ref_img
            pixel_values_ref_img = torch.from_numpy(ref_img).permute(2, 0, 1).contiguous() / 255.0
            
            pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous() / 255.
            # pixel_values_pose = torch.from_numpy(pixel_values_pose).permute(0, 3, 1, 2).contiguous() / 255.

            if self.is_image:
                pixel_values = pixel_values[0]
                # pixel_values_pose = pixel_values_pose[0]

        return pixel_values, pixel_values_ref_img
        # return pixel_values, pixel_values_ref_img, pixel_values_pose, clip_ref_image
        
    def __iter__(self):
        for data in self.wds_dataset:
            key          = data["__key__"]
            url          = data["__url__"]
            video_bytes  = data["mp4"]
            meta_dic     = json.loads(data["json"])

            # for debug
            # pixel_values, pixel_values_pose, clip_ref_image, pixel_values_ref_img = self.get_clip_frames_and_conditions(video_bytes)

            try:
                # frames = self.get_clip_frames(video_bytes)
                pixel_values, pixel_values_ref_img = self.get_clip_frames_and_conditions(video_bytes, meta_dic)
                # pixel_values, pixel_values_pose, clip_ref_image, pixel_values_ref_img = self.get_clip_frames_and_conditions(video_bytes)
            
                pixel_values = self.pixel_transforms(pixel_values)
                # pixel_values_pose = self.pixel_transforms(pixel_values_pose)
                
                pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
                pixel_values_ref_img = self.pixel_transforms(pixel_values_ref_img)
                pixel_values_ref_img = pixel_values_ref_img.squeeze(0)
                
                # clip_ref_image = clip_ref_image.unsqueeze(1) # [bs,1,768]
                drop_image_embeds = 1 if random.random() < 0.1 else 0
                sample = dict(
                    pixel_values=pixel_values, 
                    # pixel_values_pose=pixel_values_pose,
                    # clip_ref_image=clip_ref_image,
                    pixel_values_ref_img=pixel_values_ref_img,
                    drop_image_embeds=drop_image_embeds,
                    )

            except Exception as e:
                # traceback.print_exc()
                continue


            

            # frames_captions = [meta_dic["tag"]["caption_coca"]] * self.video_length
            # assert(len(frames_captions) > 0)

            # cond_dict = self.get_conditions(frames)
            yield sample


def interpolate(data, crop_y1, crop_y2, crop_x1, crop_x2, size):
    data = torch.tensor(np.array(data)).permute(0, 3, 1, 2).float()
    data = F.interpolate(input=data[...,crop_y1:crop_y2, crop_x1:crop_x2], size=size, mode='bilinear', align_corners=False)
    return data


def gaussian(x, y, sigma):
    exponent = -(x**2 + y**2) / (2 * sigma**2)
    return np.exp(exponent) / (2 * np.pi * sigma**2)


def generate_gaussian_response(image_shape, landmarks, sigma=3):
    height, width = image_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    for x, y in landmarks:
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            for i in range(-sigma, sigma+1):
                for j in range(-sigma, sigma+1):
                    new_x, new_y = x + i, y + j
                    if 0 <= new_x < width and 0 <= new_y < height:
                        heatmap[new_y, new_x] += gaussian(i, j, sigma)                        
    
    heatmap[np.isnan(heatmap)] = 0
    max_value = np.max(heatmap)
    if max_value != 0:
        heatmap /= max_value
    heatmap = heatmap[:,:,np.newaxis]
    return heatmap 


def get_tarfile_name_list(bucket, object_dir):
    tarfile_path_list = megfile.smart_listdir(
        megfile.smart_path_join(f"s3://{bucket}", object_dir)
    )
    tarfile_name_list = [tarfile_path for tarfile_path in tarfile_path_list if tarfile_path.endswith(".tar")]
    return tarfile_name_list


class PexelsDataset(Dataset):
    """
    load video-only data, and get dwpose condition
    """

    def __init__(
            self,
            data_dir,
            sample_size=(512, 512), sample_stride=1, sample_n_frames=16, is_test=False
    ):
        self.dataset = glob(data_dir+'/*')

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames

        self.sample_size = sample_size

        self.pixel_transforms = transforms.Compose([
            transforms.Resize([self.sample_size[0],self.sample_size[0]]),
            transforms.CenterCrop(self.sample_size),
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
        
        del video_reader

        return image_np

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        while True:
            try:
                image_np = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)
           
        pixel_values = [np.array(crop_and_resize(Image.fromarray(c), target_size=None, crop_rect=None)) for c in image_np]
        pixel_values = np.array(pixel_values)
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous() / 255.0
        pixel_values = self.pixel_transforms(pixel_values)

        ref_idx = random.randint(0, self.length - 1)
        pixel_values_ref_img = pixel_values[ref_idx]
        print('pixel_values', pixel_values.shape, pixel_values.max(), pixel_values.min())     
        print('pixel_values_ref_img', pixel_values_ref_img.shape)     
        sample = dict(
                    pixel_values=pixel_values, 
                    pixel_values_ref_img=pixel_values_ref_img,
                    )
        return sample

# def train_collate_fn(examples):
#     images = torch.stack([example["image"] for example in examples])
#     images = images.to(memory_format=torch.contiguous_format).float()
#     masked_images = torch.stack([example["masked_image"] for example in examples])
#     masked_images = masked_images.to(memory_format=torch.contiguous_format).float()
#     masks = torch.stack([example["mask"] for example in examples])
#     masks = masks.to(memory_format=torch.contiguous_format).float()
#     caption_tokens = torch.stack([example["caption_token"] for example in examples])
#     caption_tokens = caption_tokens.to(memory_format=torch.contiguous_format).long()
#     return {
#         "image"          : images, 
#         "masked_image"   : masked_images, 
#         "mask"           : masks, 
#         "caption_token"  : caption_tokens,
#     }


import imageio
import torchvision

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

if __name__ == "__main__":
    import resource
    from tqdm import tqdm
    import cv2
    TEST_TYPE = 'test'
    if TEST_TYPE == 'train':
        dataset = S3VideosIterableDataset(
            [
                's3://ljj/Datasets/Videos/processed/CelebV_webdataset_20231211',
            #  's3://ljj/Datasets/Videos/processed/hdvila100m_20231216',
            #  's3://ljj/Datasets/Videos/processed/pexels_20231217',
            #  's3://ljj/Datasets/Videos/processed/xiaohongshu_webdataset_20231212',
            ],
            video_length = 16,
            resolution = 512,
            frame_stride = 1,
        )
        dataloader = wds.WebLoader(
            dataset, 
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn = None,
        ).with_length(len(dataset))
        # pbar = tqdm()
        save_num = 1000
        for idx,data in tqdm(enumerate(dataloader)):
            # # pbar.update(1)
            # # import pdb; pdb.set_trace()
            # print(f"RAM PEAK: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3):.4f}")
            # print(data['pixel_values'].shape)
            for i in range(data['pixel_values'].shape[0]):
                save_videos_grid(data['pixel_values'][i:i + 1].permute(0, 2, 1, 3, 4), os.path.join(".", f"debug/pixel_values{idx}-{i}.gif"), rescale=True)
                # save_videos_grid(data['pixel_values_pose'][i:i + 1].permute(0, 2, 1, 3, 4), os.path.join(".", f"debug/pixel_values_pose{idx}-{i}.gif"), rescale=True)
                # pixel_values_ref_img = ((data['pixel_values_ref_img'][i:i + 1].permute(0, 2, 3, 1)[0]+1)/2*255).numpy().astype('uint8')
                # cv2.imwrite(os.path.join(".", f"debug/pixel_values_ldmk{idx}-{i}.png"), pixel_values_ref_img)
            if idx > save_num:
                break
    else:
        def collate_fn(batch):
            return {
                'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
                'pixel_values_ref_img': torch.stack([x['pixel_values_ref_img'] for x in batch])
            }
        dataset = PexelsDataset(
        '/data/work/animate_based_ap_ctrl/data/templates/pexels_crop',
        sample_size=(512, 512), sample_stride=8, sample_n_frames=16,)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=1, collate_fn=None)
        for idx, batch in enumerate(dataloader):
            print(batch['pixel_values'].shape)
            for i in range(batch['pixel_values'].shape[0]):
                save_videos_grid(batch['pixel_values'][i:i + 1].permute(0, 2, 1, 3, 4), os.path.join(".", f"debug/test/pixel_values{idx}-{i}.mp4"), rescale=True)
                pixel_values_ref_img = ((batch['pixel_values_ref_img'][i:i + 1].permute(0, 2, 3, 1)[0]+1)/2*255).numpy().astype('uint8')
                cv2.imwrite(os.path.join(".", f"debug/test/pixel_values_ref_img{idx}-{i}.png"), pixel_values_ref_img[:,:,::-1])

            break
