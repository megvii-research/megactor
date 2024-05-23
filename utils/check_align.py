import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from animatediff.utils.videoreader import VideoReader
from animatediff.utils.util import save_videos_grid, pad_image, generate_random_params, apply_transforms
from einops import rearrange
import os
from PIL import Image
import numpy as np
from controlnet_resource.dense_dwpose.densedw import DenseDWposePredictor
import cv2

def crop_and_resize(frame, target_size, crop_rect=None, is_arcface=False):
    height, width = frame.size
    if is_arcface:
        target_size = (112, 112)

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
    frame_resized = frame_cropped.resize(target_size, Image.ANTIALIAS)
    return frame_resized

def save_one_control(images, savedir, source_name):
    pixel_values_pose = torch.Tensor(np.array(images))
    pixel_values_pose = rearrange(
        pixel_values_pose, "(b f) h w c -> b f h w c", b=1)
    pixel_values_pose = pixel_values_pose / 255.0
    pixel_values_pose = rearrange(pixel_values_pose, "b f h w c -> b c f h w")
    pixel_values_pose = pixel_values_pose.cpu()
    save_videos_grid(
            pixel_values_pose, f"{savedir}/{source_name}.gif")


if __name__ == "__main__":
    video_list = [
        "./me_test_data/my_data_0.gif",
        "./me_test_data/my_data_8.gif",
        "./me_test_data/my_data_12.gif",
        "./me_test_data/my_data_16.gif",
        "./me_test_data/my_data_17.gif",
        "./me_test_data/my_data_18.gif",
    ]
    source_list = [
        "./new_test_data/chenweiting.png",
        "./new_test_data/chenweiting.png",
        "./new_test_data/chenweiting.png",
        "./new_test_data/chenweiting.png",
        "./new_test_data/chenweiting.png",
        "./new_test_data/chenweiting.png",
    ]
    device = torch.device(f"cuda:{0}")
    dwpose_model = DenseDWposePredictor(device)
    os.makedirs("./align_result", exist_ok=True)
    for i, (test_video, source_image) in enumerate(zip(video_list, source_list)):
        if source_image.endswith(".mp4") or source_image.endswith(".gif"):
            origin_video = VideoReader(source_image).read()[0]
            origin_video = Image.fromarray(origin_video)
            origin_video = crop_and_resize(origin_video, (512, 512), crop_rect=None)
            origin_video = np.array(origin_video)
        else:
            source_image = Image.open(source_image)
            source_image = crop_and_resize(source_image, (512, 512), crop_rect=None)
            origin_video = np.array(source_image)

        control = VideoReader(test_video).read()
        item_list = []
        for item in control:
            item = Image.fromarray(item)
            item = crop_and_resize(item, (512, 512), crop_rect=None)
            item_list.append(item)
        control = np.array(item_list)
        


        _, __, source_landmark = dwpose_model.dwpose_model(origin_video, output_type='np', image_resolution=512, get_mark=True)
        source_landmark = source_landmark["faces_all"].squeeze(0) * 512.
        control_landmarks = []
        control_results = []
        for item in control:
            _, item_res, item_land = dwpose_model.dwpose_model(item, output_type='np', image_resolution=512, get_mark=True)
            control_landmarks.append(item_land["faces_all"].squeeze(0) * 512.)
            control_results.append(item_res)
        control_results = np.array(control_results)
        # print("src and dist is", np.float32(control_landmarks[0]).squeeze(0)[:3].shape, np.float32(source_landmark).squeeze(0)[:3].shape)
        # print("example is", np.float32([[50, 50], [150, 50], [100, 200]]).shape)
        # print("control_landmarks[0] is", control_landmarks[0].shape, source_landmark.shape)
        # print("control_landmarks[0] is", control_landmarks[0], source_landmark)
        src_point = control_landmarks[0]
        src_point_1 = src_point[37:43, :].mean(axis=0)
        src_point_2 = src_point[43:49, :].mean(axis=0)
        src_point_3 = src_point[49:, :].mean(axis=0)
        src_point = np.array([src_point_1, src_point_2, src_point_3]).astype("int32")

        dist_point = source_landmark
        dist_point_1 = dist_point[37:43, :].mean(axis=0)
        dist_point_2 = dist_point[43:49, :].mean(axis=0)
        dist_point_3 = dist_point[49:, :].mean(axis=0)
        dist_point = np.array([dist_point_1, dist_point_2, dist_point_3]).astype("int32")
        
        transform_matrix = cv2.getAffineTransform(np.float32(src_point), np.float32(dist_point))

        control_aligns = []
        for item in control_results:
            aligned_img = cv2.warpAffine(item, transform_matrix, (512, 512))
            control_aligns.append(aligned_img)
        control_aligns = np.array(control_aligns)

        origin_video = origin_video[None, ...].repeat(repeats=len(control_aligns), axis=0)
        
        control_results = torch.Tensor(np.array(control_results)) / 255.
        control_aligns = torch.Tensor(np.array(control_aligns)) / 255.
        origin_video = torch.Tensor(np.array(origin_video)) / 255.

        control_results = rearrange(control_results, "t h w c -> 1 c t h w") 
        control_aligns = rearrange(control_aligns, "t h w c -> 1 c t h w") 
        origin_video = rearrange(origin_video, "t h w c -> 1 c t h w") 

        save_videos_grid(torch.cat([control_results, control_aligns, origin_video]), f"./align_result/{i}.gif", save_every_image=False)
        save_videos_grid(control_aligns, f"./align_result/{i}_control.gif", save_every_image=False)
        print(f"finish work {i}")
