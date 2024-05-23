import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from animatediff.utils.videoreader import VideoReader
from animatediff.utils.util import save_videos_grid, pad_image, generate_random_params, apply_transforms
from einops import rearrange
import os
from PIL import Image
import numpy as np

if __name__ == "__main__":
    video_list = [
        "./new_test_data/001.gif",
        "./new_test_data/002.gif",
        "./new_test_data/003.gif",
        "./new_test_data/004.gif",
        "./new_test_data/005.gif",
        "./new_test_data/006.gif",
        "./new_test_data/007.gif"
    ]
    os.makedirs("./warp_result", exist_ok=True)
    for i, test_video in enumerate(video_list):
        torch.manual_seed(i)
        torch.cuda.manual_seed_all(i)

        control = VideoReader(test_video).read()
        dwpose_conditions = control
        control = torch.Tensor(np.array(control)) / 255.
        control = rearrange(control, "t h w c -> 1 c t h w") 
        
        B, H, W, C = dwpose_conditions.shape

        params = generate_random_params(W, H)
        print("params is", params)
        
        video_length = control.shape[0]
        ans_list = []
        for item in dwpose_conditions:
            ans_list.append(apply_transforms(Image.fromarray(item), params))
        ans_list = torch.Tensor(np.array(ans_list))
        ans_list = rearrange(ans_list, "t h w c -> 1 c t h w") / 255.

        print("control shape", control.shape, ans_list.shape)
        save_videos_grid(torch.cat([control, ans_list]), f"./warp_result/{i}.gif", save_every_image=False)
        print(f"finish work {i}")
