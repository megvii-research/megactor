from torch.utils.data import IterableDataset
import io
import boto3
import os
import random
import bisect
import pandas as pd
import torch.nn.functional as F
import omegaconf
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu
import imageio
import numpy as np
import cv2
from PIL import Image, ImageDraw
from glob import glob
import tarfile
from megfile import smart_open as open
from megfile import smart_glob
import msgpack
import pickle
import facer
import pickle as pkl
# from functools import lru_cache

DEBUG = os.environ.get('DEBUG')
if DEBUG:
    print('DEBUG MODE')

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

class S3VideosDataset(Dataset):   
    def __init__(self,
                 data_dirs,
                 deca_dir='deca_v231023',
                 valid_size = 256,
                 use_faceparsing=False,
                 use_deca=True,
                 ldmk_use_gaussian=False,
                 subsample=None,
                 video_length=16,
                 resolution=[512, 512],
                 frame_stride=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=True,
                 fps_schedule=None,
                 fs_probs=None,
                 bs_per_gpu=None,
                 trigger_word='',
                 dataname='',
                 is_image=False,
                 ):
        self.data_dirs = data_dirs
        self.deca_dir = deca_dir
        self.valid_size = valid_size
        
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(
            resolution, int) else resolution
        self.frame_stride = frame_stride
        self.fps_max = fps_max
        self.load_raw_resolution = load_raw_resolution
        self.fs_probs = fs_probs
        self.trigger_word = trigger_word
        self.dataname = dataname
        self.is_image = is_image
        self.videos_info_list = self._get_video_list()
        self.ldmk_use_gaussian = ldmk_use_gaussian
        self.use_faceparsing = use_faceparsing
        self.use_deca = use_deca
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "resize_center_crop":
                assert (self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(resolution),
                    transforms.CenterCropVideo(resolution),
                ])
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

        self.fps_schedule = fps_schedule
        self.bs_per_gpu = bs_per_gpu
        if self.fps_schedule is not None:
            assert (self.bs_per_gpu is not None)
            self.counter = 0
            self.stage_idx = 0
        
        self.face_detector = facer.face_detector('retinaface/mobilenet', device="cpu")
        # self.face_ana = load_landmarks_model()
    
    # asian_woman_v231102/59a78342-0a2f-4b40-8a47-033b4ea68ff1/
    # results/EMOCA_v2_lr_mse_20/000040_000/
    def _get_video_list(self):
        videos_info_list = []
        for data_dir in self.data_dirs:
            # print('data_dir', data_dir)
            for worker_data_dir in smart_glob(data_dir + '/*'):
                print('worker_data_dir', worker_data_dir)
                video_data_dirs = smart_glob(worker_data_dir + '/*')
                for video_data_dir in video_data_dirs:
                    video_data_dir = video_data_dir + '/results/EMOCA_v2_lr_mse_20'
                    # print('video_data_dir', video_data_dir)
                    num_frames = len(smart_glob(video_data_dir + '/*'))
                    if num_frames > 0:
                        videos_info_list.append((video_data_dir, num_frames))
                        if DEBUG:
                            print(video_data_dir, num_frames)
                        # if len(videos_info_list) > 16:
                        #     break
                        
        print(f'Detect {len(videos_info_list)} labeled video files')
        return videos_info_list

    def get_frame_number_format(self):
        return "%06d"
    
    def load_detections(self, fname):
        with open(fname, "rb" ) as f:  
            detection_fnames = pkl.load(f)
            centers = pkl.load(f)
            sizes = pkl.load(f)
            try:
                last_frame_id = pkl.load(f)
            except:
                last_frame_id = -1
            try:
                landmark_fnames = pkl.load(f)
            except:
                landmark_fnames = [None]*len(detection_fnames)

        return detection_fnames, landmark_fnames, centers, sizes, last_frame_id
    
    def load_landmark(self, fname):
        with open(fname, "rb") as f:
            landmark_type = pkl.load(f)
            landmark = pkl.load(f)
        return landmark_type, landmark

       
    def get_labels(self, video_info_dir, num_frames):
        # num_frames = len(smart_glob(video_info_dir + f'/{self.deca_dir}/*'))

        # sample strided frames
        frame_stride = self.frame_stride
        all_frames = list(range(0, num_frames, frame_stride))
        if len(all_frames) < self.video_length:  # recal a max fs
            frame_stride = num_frames // self.video_length
            assert (frame_stride != 0)
            all_frames = list(range(0, num_frames, frame_stride))

        # select a random clip
        rand_idx = random.randint(0, len(all_frames) - self.video_length)
        frame_indices = all_frames[rand_idx:rand_idx+self.video_length]

        key_filenames = ['frame.png', 'frame_deca_full.png', 'frame_deca_trans.png']

        frames_captions = []
        frames_deca_orig_inputs_pathes = []
        frames_shape_detail_pathes = []
        frames_shape_pathes = []
        frames_deca_ldmk_pathes = []

        video_root_dir = video_info_dir.replace('results/EMOCA_v2_lr_mse_20', '')
        bboxes_pkl_path = video_info_dir.replace('results/EMOCA_v2_lr_mse_20', 'detections/bboxes.pkl')
        detection_fnames, landmarks, centers, sizes, last_frame_id  = self.load_detections(bboxes_pkl_path)
        for frame_index in frame_indices:
            frame_index = str(self.get_frame_number_format() % frame_index) + '_000'
            # TODO: caption relabel
            # frames_captions.append(frame_meta['caption'])
            frames_deca_dir = os.path.join(video_info_dir,  str(frame_index))
            frames_deca_orig_inputs_pathes.append(frames_deca_dir + '/frame.png')
            frames_shape_detail_pathes.append(frames_deca_dir + '/frame_deca_full.png')
            frames_shape_pathes.append(frames_deca_dir + '/frame_deca_trans.png')
            frames_deca_ldmk_path = os.path.join(video_root_dir, 'landmarks', f'{frame_index}.pkl')
            frames_deca_ldmk_pathes.append(frames_deca_ldmk_path)
        

        new_height, new_width = self.resolution 
        size = (new_height, new_width)
        # RGB mode
        frames_imgs = [np.array(Image.open(open(x,'rb'))) for x in frames_deca_orig_inputs_pathes]
        h, w, c = frames_imgs[0].shape
        # crop and resize according face
        video_crop_center = centers[0][0]
        video_crop_center = np.array(video_crop_center).astype('uint')
        video_crop_size = int(sizes[0][0]) // 2
        x_min, x_max = int(video_crop_center[0]-video_crop_size), int(video_crop_center[0]+video_crop_size)
        y_min, y_max = int(video_crop_center[1]-video_crop_size), int(video_crop_center[1]+video_crop_size)
        # x_min = max(0, int(x_min))
        # y_min = max(0, int(y_min))
        # x_max = min(w, int(x_max))
        # y_max = min(h, int(y_max))
        h_delta = min(int(h-y_max), y_min, (y_max-y_min)//2)
        crop_y1 = max(y_min - h_delta, 0)
        crop_y2 = min(y_max + h_delta, h)
        h = crop_y2 -crop_y1
        w_delta = max(0, (h - (x_max-x_min)) / 2)
        crop_x1 = max(int(x_min - w_delta),0)
        crop_x2 = min(int(x_max + w_delta),w)

        # crop_y1, crop_y2, crop_x1, crop_x2 = y_min, y_max, x_min, x_max 
        #############################      
        frames_imgs = interpolate(frames_imgs, crop_y1, crop_y2, crop_x1, crop_x2, size)

        # filter datas
        # (224, 224, 3)
        with torch.inference_mode():
            faces = self.face_detector(frames_imgs[0].unsqueeze(0))
            assert faces['image_ids'].numel() == 1, "Image must has exactly one face!"

        # landmark
        # crop_landmarks = [self.load_landmark(x)[1] for x in frames_deca_ldmk_pathes]
        # print('crop_landmarks', len(crop_landmarks), crop_landmarks[0].shape)
        # print(crop_landmarks[0].max(), crop_landmarks[0].min())
        # frames_landmarks = []
        # new_height, new_width = self.resolution 
        # print('new_height, new_width', new_height, new_width)
            
        # for landmark in crop_landmarks:
        #     new_landmarks = [] 
        #     img_tensor = torch.zeros(new_height, new_width,1)
        #     for x,y in landmark:
        #         x,y = int(x), int(y)
        #         # print(x,y)
        #         xc = x - crop_x1 + x_min
        #         yc = y - crop_y1 + y_min
        #         # print(xc,yc)
        #         scale_x1 = (crop_x2-crop_x1) / (x_max-x_min)
        #         scale_y1 = (crop_y2-crop_y1) / (y_max-y_min)
        #         scale_x = new_width / (crop_x2-crop_x1)
        #         scale_y = new_height / (crop_y2-crop_y1)

        #         xr = int(xc * scale_x * scale_x1)
        #         yr = int(yc * scale_y * scale_y1)
        #         if xr >= 0 and xr < new_width and yr >= 0 and yr < new_height:
        #             # print('bingo')
        #             img_tensor[yr,xr] = 1
        #         new_landmarks.append((xr,yr))
        #     frames_landmarks.append(img_tensor)
        # frames_landmarks = torch.stack(frames_landmarks).permute(0, 3, 1, 2).float()

        frames_imgs = (frames_imgs / 255.0 - 0.5) * 2

        frames_shape_detail_imgs = [np.array(Image.open(open(x,'rb')))[:,:,:1] for x in frames_shape_detail_pathes]
        frames_shape_detail_imgs = interpolate(frames_shape_detail_imgs, crop_y1, crop_y2, crop_x1, crop_x2, size)
        frames_shape_detail_imgs = (frames_shape_detail_imgs / 255.0 - 0.5) * 2
        
        frames_shape_imgs = [np.array(Image.open(open(x,'rb')))[:,:,:1] for x in frames_shape_pathes]
        frames_shape_imgs = interpolate(frames_shape_imgs, crop_y1, crop_y2, crop_x1, crop_x2, size)
        frames_shape_imgs = (frames_shape_imgs / 255.0 - 0.5) * 2
        
        
        if self.is_image:
            frames_imgs = frames_imgs[0]
            frames_shape_detail_imgs = frames_shape_detail_imgs[0]
            frames_shape_imgs = frames_shape_imgs[0]
            # frames_landmarks = frames_landmarks[0]
            
        # frames_captions = ['A photo of a face, its emotion is ' + x.split(" ")[-1] for x in frames_captions]
        frames_captions = [''] * self.video_length
        assert(len(frames_captions) > 0)

        sample = dict(pixel_values=frames_imgs, 
                        texts=frames_captions,
                        shape_detail_imgs = frames_shape_detail_imgs,
                        shape_imgs = frames_shape_imgs,
                        # landmark = frames_landmarks,
                        )

        
        return sample
    
    def __getitem__(self, f_idx):
        while True:
            f_idx = f_idx % len(self.videos_info_list)
            video_info_dir, num_frames = self.videos_info_list[f_idx]
            
            if DEBUG:
                sample = self.get_labels(video_info_dir, num_frames)
                print('debug sample')
                for x,y in sample.items():
                    try:
                        print(x, y.shape)
                    except:
                        print(x, len(y))
                        print(x, y)
                        continue
                break
            else:
                try:
                    sample = self.get_labels(video_info_dir, num_frames)                        
                    break
                except:
                    f_idx += 1
                    continue
        
        return sample
    
    def __len__(self):
        return len(self.videos_info_list)

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

if __name__ == '__main__':


    data_dirs = [
        # 's3://ljj-sh/Datasets/Videos/videos4000_gen',
        # 's3://ljj-sh/Datasets/Videos/videos1600_gen'
        's3://ljj-sh/Datasets/Videos/videos_emoca_labels_v0'

    ]

    # ldmk_use_gaussian= False
    # use_faceparsing= False
    use_deca= False
    resolution=       [512,512]
    frame_stride=     1
    video_length=     16
    is_image = False
    deca_dir = 'deca_v231023'

    save_dir = f'debug{video_length}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    dataset = S3VideosDataset(data_dirs, 
                              use_deca=use_deca, 
                              resolution =resolution,
                              deca_dir=deca_dir,
                                video_length=video_length, frame_stride=frame_stride, is_image=is_image)
    print('dataset size is ', len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=1,)
    
    for idx, batch in enumerate(dataloader):
        print('gen',batch["pixel_values"].shape, batch["pixel_values"].shape, batch["texts"])
        print('gen',batch["pixel_values"].max(), batch["pixel_values"].min())
        # caption = batch["texts"][0][0].replace(' ', '_')
        caption = idx #batch["frames_deca_pathes"][0][0].replace('/', '_')
        

        if is_image:
            image_array = ((batch["pixel_values"][0].permute(1, 2, 0)+1)/2 * 255).numpy().astype(np.uint8)[...,::-1]
            shape_detail_imgs_array = ((batch["shape_detail_imgs"][0].permute(1, 2, 0)+1)/2 * 255).numpy().astype(np.uint8)
            shape_detail_imgs_array = np.repeat(shape_detail_imgs_array, repeats=3, axis=2)
            cv2.imwrite(f"{save_dir}/{caption}_image.png", image_array)
            cv2.imwrite(f"{save_dir}/{caption}_shape_detail.png", shape_detail_imgs_array)
        else:
            # batch["pixel_values"] = batch["pixel_values"] * (1-batch["landmarks"])
            video_array = ((batch["pixel_values"][0].permute(0, 2, 3, 1)+1)/2 * 255).numpy().astype(np.uint8)
            shape_detail_imgs_array = ((batch["shape_detail_imgs"][0].permute(0, 2, 3, 1)) * 255).numpy().astype(np.uint8)
            shape_imgs_array = ((batch["shape_imgs"][0].permute(0, 2, 3, 1)) * 255).numpy().astype(np.uint8)
            # ldmk_array = ((batch["landmark"][0].permute(0, 2, 3, 1)) * 255).numpy().astype(np.uint8)
            print('caption:', caption)
            print('video_array', video_array.shape, video_array.max(), video_array.min())
            print('shape_detail_imgs_array', shape_detail_imgs_array.shape, shape_detail_imgs_array.max(), shape_detail_imgs_array.min())


            # with imageio.get_writer(f"{save_dir}/{idx}_{caption}_video.mp4", fps=30) as video_writer:
            #     for frame in video_array:
            #         video_writer.append_data(frame)
            
            # with imageio.get_writer(f"{save_dir}/{idx}_{caption}_shape_detail_frames{video_length}.mp4", fps=30) as video_writer:
            #     for shape_detail in shape_detail_imgs_array:
            #         shape_detail = np.repeat(shape_detail, repeats=3, axis=2)
            #         video_writer.append_data(shape_detail)
            
            # with imageio.get_writer(f"{save_dir}/{idx}_{caption}_shape_detail_frames{video_length}.mp4", fps=30) as video_writer:
            #     for shape_detail in shape_detail_imgs_array:
            #         shape_detail = np.repeat(shape_detail, repeats=3, axis=2)
            #         video_writer.append_data(shape_detail)

            # with imageio.get_writer(f"{save_dir}/{idx}_{caption}.mp4", fps=30) as video_writer:
            #     for frame,shape_detail,shape,ldmk in zip(video_array,shape_detail_imgs_array,shape_imgs_array, ldmk_array):
            #         shape_detail = np.repeat(shape_detail, repeats=3, axis=2)
            #         shape = np.repeat(shape, repeats=3, axis=2)
            #         ldmk = np.repeat(ldmk, repeats=3, axis=2)
            #         mask = shape_detail==0
            #         frame_with_mask = frame.copy()
            #         frame_with_mask[mask] = 0
            #         res = np.hstack((frame, shape_detail, shape, ldmk, frame_with_mask))
            #         print('res', res.shape)
            #         video_writer.append_data(res)    
            
            with imageio.get_writer(f"{save_dir}/{idx}_{caption}.mp4", fps=30) as video_writer:
                for frame,shape_detail,shape in zip(video_array,shape_detail_imgs_array,shape_imgs_array):
                    shape_detail = np.repeat(shape_detail, repeats=3, axis=2)
                    shape = np.repeat(shape, repeats=3, axis=2)
                    mask = shape_detail==0
                    frame_with_mask = frame.copy()
                    frame_with_mask[mask] = 0
                    res = np.hstack((frame, shape_detail, shape, frame_with_mask))
                    print('res', res.shape)
                    video_writer.append_data(res)   

            # if use_faceparsing:
            #     with imageio.get_writer(f"{save_dir}/{idx}_{caption}_video_all.mp4", fps=30) as video_writer:
            #         for frame,ldmk,faceparsing,mask in zip(video_array,ldmks_array,face_parsings_array,masks_array):
            #             ldmk = np.repeat(ldmk, repeats=3, axis=2)
            #             faceparsing = np.repeat(faceparsing, repeats=3, axis=2)
            #             mask = np.repeat(mask, repeats=3, axis=2)
            #             frame[mask==0] = 0
            #             res = np.hstack((frame,ldmk,faceparsing))
            #             video_writer.append_data(res)    

            #     with imageio.get_writer(f"{save_dir}/{idx}_{caption}_fp_frames{video_length}.mp4", fps=30) as video_writer:
            #         for frame in face_parsings_array:
            #             video_writer.append_data(frame)
                
            #     with imageio.get_writer(f"{save_dir}/{idx}_{caption}_mask_frames{video_length}.mp4", fps=30) as video_writer:
            #         for frame in masks_array*255:
            #             video_writer.append_data(frame)

            # else:
            #     with imageio.get_writer(f"{save_dir}/{idx}_{caption}_video_all.mp4", fps=30) as video_writer:
            #         for frame,ldmk in zip(video_array,ldmks_array):
            #             ldmk = np.repeat(ldmk, repeats=3, axis=2)
            #             res = np.hstack((frame,ldmk))
            #             video_writer.append_data(res)    

        if idx >= 100:
            break
