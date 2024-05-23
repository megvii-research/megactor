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
import refile
from libs.controlnet_aux_lib.dwpose.draw_dwpose_v1 import get_dwpose_body
from animatediff.utils.util import crop_and_resize_tensor
from PIL import Image
import pickle

# def gopen_refile(url, mode="rb", bufsize=8192, **kw):
#     """Open a URL with `curl`.

#     :param url: url (usually, http:// etc.)
#     :param mode: file mode
#     :param bufsize: buffer size
#     """
#     if mode[0] == "r":
#         return refile.smart_open(url, mode="rb", bufsize=8192, **kw)
#     elif mode[0] == "w":
#         raise NotImplementedError
#     else:
#         raise ValueError(f"{mode}: unknown mode")
# wds.gopen_schemes["s3"] = gopen_refile 


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
        endpoint_url = "http://oss.hh-b.brainpp.cn",
        image_processor = None,
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

        self.image_processor = image_processor

        # self.pixel_transforms = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     # transforms.Resize(resolution),
        #     # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        # ])
       

    def get_tarfilepath_list(self, data_dirs):
        tarfile_path_list = []
        for data_dir in data_dirs:
            if refile.smart_isdir(data_dir):
                file_path_list = refile.smart_listdir(data_dir)
                tarfile_path_list += [
                    refile.smart_path_join(data_dir, file_path)
                    for file_path in file_path_list if file_path.endswith(".tar")]
            elif data_dir.endswith(".tar"):
                tarfile_path_list.append(data_dir)
            else:
                raise NotImplementedError("仅支持输入以下路径:(1)以.tar结尾的Tar包路径; (2)文件夹路径")
        assert len(tarfile_path_list)>0, "没找到任何Tar包文件"
        print(f'finish get tarfile_path_list len is {len(tarfile_path_list)}')
        return tarfile_path_list


    def get_webdataset(self, ):
        url_list = []
        for tarfilepath in self.tarfilepath_list:
            if tarfilepath.startswith("s3://") or tarfilepath.startswith("tos://"):
                tarfilepath = tarfilepath.replace("tos://", "s3://")
                url_list.append(
                    # f"pipe: aws --endpoint-url={self.endpoint_url} s3 cp {tarfilepath} - | pv -ptrb"
                    f"pipe: aws --endpoint-url={self.endpoint_url} s3 cp {tarfilepath} - "
                )
                # url_list.append(tarfilepath)
            else:
                url_list.append(tarfilepath)

        dataset = wds.WebDataset(url_list, resampled=self.wds_resampled)
        if self.wds_shuffle:
            dataset  = dataset.shuffle(100)
        return dataset


    def __len__(self, ):
        return self.dataset_length


    def get_random_clip_indices(self, n_frames:int) -> List[int]:

        self_frame_stride = self.frame_stride
        # print('self_frame_stride is', self_frame_stride)
        if self.video_length != 1:
            all_indices = list(range(0, n_frames, self_frame_stride))
            if len(all_indices) < self.video_length:
                frame_stride = n_frames // self.video_length
                assert (frame_stride != 0)
                all_indices = list(range(0, n_frames, frame_stride))
            
            rand_idx = random.randint(0, len(all_indices) - self.video_length)
            clip_indices = all_indices[rand_idx:rand_idx+self.video_length]    
        else:
            clip_indices = list(np.random.choice(range(n_frames), size=2, replace=True))
        
        return clip_indices

    def get_clip_frames(self, video_bytes:bytes, dwpose_result_b:bytes, dwpose_score_b:bytes, faces_bbox_b:bytes) -> torch.Tensor:
        frames = []
        dwpose_result_real = []
        dwpose_score_real = []
        frames_real = []
        dwpose_results = pickle.loads(dwpose_result_b)
        dwpose_scores = pickle.loads(dwpose_score_b)
        with iio.imopen(video_bytes, "r", plugin="pyav") as file:
            
            frames = file.read(index=...)  # np.array, [n_frames,h,w,c],  rgb, uint8
            L = frames.shape[0]
            # remove black frame since most of black frame occur in the suf or pre
            for i, frame in enumerate(frames):
                if frame.mean() > 5.0:
                    frames_real.append(frame)
                    dwpose_result_real.append(dwpose_results[i][:1, ...])
                    dwpose_score_real.append(dwpose_scores[i][:1, ...])
                    
            frames = np.array(frames_real, dtype="float32")
            dwpose_results = np.array(dwpose_result_real, dtype="float32")
            dwpose_scores = np.array(dwpose_score_real, dtype="float32")

            n_frames = frames.shape[0]
            assert n_frames >= self.video_length, f"len(VideoClip): {n_frames} < {self.video_length}"
            clip_indices = self.get_random_clip_indices(n_frames)
               
            frames = frames[clip_indices, ...]
            frames = torch.tensor(frames).permute(0, 3, 1, 2).float().clip(0., 255.)

        candidates = dwpose_results[clip_indices, ...]
        subsets = dwpose_scores[clip_indices, ...]

        # Get frame height and width
        height, width = frames.shape[2:]

        # crop point and image to center
        xmin = np.min(candidates[0, :, 24:92, 0])
        xmax = np.max(candidates[0, :, 24:92, 0])
        ymin = np.min(candidates[0, :, 24:92, 1])
        ymax = np.max(candidates[0, :, 24:92, 1])
        crop_center_x, crop_center_y = (xmax + xmin) / 2., (ymax + ymin) / 2. - (ymax - ymin) / 4.
        if crop_center_x < 0 or crop_center_x > width:
            crop_center_x = width / 2.
        
        if crop_center_y < 0 or crop_center_y > height:
            crop_center_y = height / 2.
        
        crop_size = min(width, height, min(crop_center_x, crop_center_y, width - crop_center_x, height - crop_center_y) * 2)
        left = crop_center_x - crop_size // 2
        top = crop_center_y - crop_size // 2
        right = crop_center_x + crop_size // 2
        bottom = crop_center_y + crop_size // 2
        frames = frames[:, :, int(top):int(bottom), int(left):int(right)].float()
        new_width, new_height = self.resolution

        frames = torch.nn.functional.interpolate(frames, size=(new_width, new_height), mode='bilinear', align_corners=False)

        # try:
        # except Exception:
        #     # print("top bottom left right", top, bottom, left, right)
        #     # print("crop_center_x crop_center_y", crop_center_x, crop_center_y)
        #     # print("height, width", height, width)
        #     # print("crop_size is", crop_size)
        #     raise Exception

        scale = new_width / crop_size

        for i in range(len(candidates)):
            for j in range(134):
                x, y = candidates[i, 0, j]
                x, y = int((x - left) * scale), int((y - top) * scale)
                candidates[i, 0, j] = x, y

        dwpose_results = []
        for candidate, subset in zip(candidates, subsets):
            dwpose_result_item = get_dwpose_body(candidate, subset, new_width, new_height)
            dwpose_results.append(dwpose_result_item)
        dwpose_results = np.array(dwpose_results, dtype="uint8")
        # assert len(np.unique(dwpose_results)) > 1, f"ALL frame has not condition, this is {np.unique(dwpose_results)}"

        # print("dwpose_results shape", dwpose_results.shape, H, W)
        dwpose_results = torch.tensor(dwpose_results, ).permute(0, 3, 1, 2).float()

        return frames, dwpose_results

    def __iter__(self):
        while True:
            try:
                for data in self.wds_dataset:
                    # key          = data["__key__"]
                    # url          = data["__url__"]
                    try:
                        video_bytes  = data["mp4"]
                        dwpose_result = data["dwpose_result.pyd"]
                        dwpose_score = data["dwpose_score.pyd"]
                        faces_bbox = data["faces_bbox.pyd"]

                        frames, dwpose_alls = self.get_clip_frames(video_bytes, dwpose_result, dwpose_score, faces_bbox) 
                        # print("frames shape is", frames.shape, self.video_length)  
                        ref_index = np.random.randint(0, self.video_length) if self.video_length != 1 else 1
                        frames, ref_frames = frames[:self.video_length, ...], frames[ref_index, ...]
                        dwpose_alls, ref_dwpose_alls = dwpose_alls[:self.video_length, ...], dwpose_alls[ref_index, ...]
                        clip_ref_frames = Image.fromarray(ref_frames.permute(1, 2, 0).numpy().astype("uint8"))
                        if self.image_processor is not None:
                            clip_ref_frames = self.image_processor(clip_ref_frames, return_tensors="pt").pixel_values
                        drop_image_embeds = 1 if random.random() < 0.1 else 0
                        sample_dic = dict(
                            pixel_values=frames, 
                            dwpose_all=dwpose_alls,
                            pixel_values_ref=ref_frames,
                            ref_img_conditions=ref_dwpose_alls,
                            clip_ref_frames=clip_ref_frames.squeeze() if self.image_processor is not None else ref_frames,
                            drop_image_embeds=drop_image_embeds,
                            ref_index=ref_index,
                            )
                        
                        yield sample_dic
                    except Exception as e:
                        # traceback.print_exc()
                        print('meet error for', e)
                        continue
                        # raise e
            except Exception as e:
                # traceback.print_exc()
                print('meet break error for', e)
                continue

def train_collate_fn(examples):
    images = torch.stack([example["pixel_values"] for example in examples])
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
    tarfile_path_list = refile.smart_listdir(
        refile.smart_path_join(f"s3://{bucket}", object_dir)
    )
    tarfile_name_list = [tarfile_path for tarfile_path in tarfile_path_list if tarfile_path.endswith(".tar")]
    return tarfile_name_list

if __name__ == "__main__":

    from PIL import Image

    import resource
    from tqdm import tqdm

    dataset = S3VideosIterableDataset(
        [
            "s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/CelebV_webdataset_20231211_videoblip",
            "s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/hdvila100m_20231216",
            "s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/pexels_20231217",
            "s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/xiaohongshu_webdataset_20231212",
        ],
        video_length = 16,
        resolution = [256,256],
        frame_stride = 4,
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
    # pbar = tqdm()
    from animatediff.utils.util import save_videos_grid
    cnt_num = 0
    for data in tqdm(dataloader):
        samples_per_video = data["pixel_values"]
        # img = data["pixel_values"][0,0,...].numpy() # chw
        # img = img * 0.5 + 0.5
        # samples_per_video = samples_per_video * 0.5 + 0.5
        
        # samples_per_video *= 255.
        samples_per_video = rearrange(samples_per_video, "b f c h w -> b c f h w")
        print('samples_per_video shape is', samples_per_video.shape, samples_per_video.min(), samples_per_video.max())
        save_videos_grid(samples_per_video, f"./show_data/{cnt_num}.gif", rescale=True if samples_per_video.min() < 0 else False)
        cnt_num += 1
        # print('img shape is', img.shape)
        # img = img.transpose((1,2,0))
        
        # import pdb; pdb.set_trace()
        # print(f"RAM PEAK: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3):.4f}")
        pass
 
    print("...")
    

    '''
    import accelerate
    from tqdm import tqdm

    
    accelerator = accelerate.Accelerator()

    dataset = S3VideosIterableDataset(
        ["s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/wds/CelebV_webdataset_20231211",],
        video_length = 14,
        resolution = [320,512],
        frame_stride = 2,
    )

    dataloader = wds.WebLoader(
        dataset, 
        batch_size=1,
        num_workers=8,
        collate_fn = None,
    ).with_length(len(dataset))
    # pbar = tqdm()
    for data in tqdm(dataloader, disable=not accelerator.is_main_process):
        # pbar.update(1)
        # import pdb; pdb.set_trace()
        # print(f"RAM PEAK: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3):.4f}")
        pass
 
    print("...")
    '''