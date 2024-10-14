import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import clip
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import transforms as T
from .models import build_model
from .predefined_keypoints import *
from .util import box_ops
from .util.config import Config
from .util.utils import clean_state_dict
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib import transforms
from torchvision.ops import nms


def text_encoding(instance_names, keypoints_names, model, device):

    ins_text_embeddings = []
    for cat in instance_names:
        instance_description = f"a photo of {cat.lower().replace('_', ' ').replace('-', ' ')}"
        text = clip.tokenize(instance_description).to(device)
        text_features = model.encode_text(text)  # 1*512
        ins_text_embeddings.append(text_features)
    ins_text_embeddings = torch.cat(ins_text_embeddings, dim=0)

    kpt_text_embeddings = []

    for kpt in keypoints_names:
        kpt_description = f"a photo of {kpt.lower().replace('_', ' ')}"
        text = clip.tokenize(kpt_description).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)  # 1*512
        kpt_text_embeddings.append(text_features)

    kpt_text_embeddings = torch.cat(kpt_text_embeddings, dim=0)


    return ins_text_embeddings, kpt_text_embeddings




def plot_on_image(image_pil, tgt, keypoint_skeleton,keypoint_text_prompt,output_dir):
    num_kpts = len(keypoint_text_prompt)
    H, W = tgt["size"]
    fig = plt.figure(frameon=False)
    dpi = plt.gcf().dpi
    fig.set_size_inches(W / dpi, H / dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = plt.gca()
    ax.imshow(image_pil, aspect='equal')
    ax = plt.gca()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    color_kpt = [[0.00, 0.00, 0.00],
                 [1.00, 1.00, 1.00],
                 [1.00, 0.00, 0.00],
                 [1.00, 1, 00., 0.00],
                 [0.50, 0.16, 0.16],
                 [0.00, 0.00, 1.00],
                 [0.69, 0.88, 0.90],
                 [0.00, 1.00, 0.00],
                 [0.63, 0.13, 0.94],
                 [0.82, 0.71, 0.55],
                 [1.00, 0.38, 0.00],
                 [0.53, 0.15, 0.34],
                 [1.00, 0.39, 0.28],
                 [1.00, 0.00, 1.00],
                 [0.04, 0.09, 0.27],
                 [0.20, 0.63, 0.79],
                 [0.94, 0.90, 0.55],
                 [0.33, 0.42, 0.18],
                 [0.53, 0.81, 0.92],
                 [0.71, 0.49, 0.86],
                 [0.25, 0.88, 0.82],
                 [0.5, 0.0, 0.0],
                 [0.0, 0.3, 0.3],
                 [1.0, 0.85, 0.73],
                 [0.29, 0.0, 0.51],
                 [0.7, 0.5, 0.35],
                 [0.44, 0.5, 0.56],
                 [0.25, 0.41, 0.88],
                 [0.0, 0.5, 0.0],
                 [0.56, 0.27, 0.52],
                 [1.0, 0.84, 0.0],
                 [1.0, 0.5, 0.31],
                 [0.85, 0.57, 0.94],
                 [0.00, 0.00, 0.00],
                 [1.00, 1.00, 1.00],
                 [1.00, 0.00, 0.00],
                 [1.00, 1, 00., 0.00],
                 [0.50, 0.16, 0.16],
                 [0.00, 0.00, 1.00],
                 [0.69, 0.88, 0.90],
                 [0.00, 1.00, 0.00],
                 [0.63, 0.13, 0.94],
                 [0.82, 0.71, 0.55],
                 [1.00, 0.38, 0.00],
                 [0.53, 0.15, 0.34],
                 [1.00, 0.39, 0.28],
                 [1.00, 0.00, 1.00],
                 [0.04, 0.09, 0.27],
                 [0.20, 0.63, 0.79],
                 [0.94, 0.90, 0.55],
                 [0.33, 0.42, 0.18],
                 [0.53, 0.81, 0.92],
                 [0.71, 0.49, 0.86],
                 [0.25, 0.88, 0.82],
                 [0.5, 0.0, 0.0],
                 [0.0, 0.3, 0.3],
                 [1.0, 0.85, 0.73],
                 [0.29, 0.0, 0.51],
                 [0.7, 0.5, 0.35],
                 [0.44, 0.5, 0.56],
                 [0.25, 0.41, 0.88],
                 [0.0, 0.5, 0.0],
                 [0.56, 0.27, 0.52],
                 [1.0, 0.84, 0.0],
                 [1.0, 0.5, 0.31],
                 [0.85, 0.57, 0.94],
                 [0.00, 0.00, 0.00],
                 [1.00, 1.00, 1.00],
                 [1.00, 0.00, 0.00],
                 [1.00, 1, 00., 0.00],
                 [0.50, 0.16, 0.16],
                 [0.00, 0.00, 1.00],
                 [0.69, 0.88, 0.90],
                 [0.00, 1.00, 0.00],
                 [0.63, 0.13, 0.94],
                 [0.82, 0.71, 0.55],
                 [1.00, 0.38, 0.00],
                 [0.53, 0.15, 0.34],
                 [1.00, 0.39, 0.28],
                 [1.00, 0.00, 1.00],
                 [0.04, 0.09, 0.27],
                 [0.20, 0.63, 0.79],
                 [0.94, 0.90, 0.55],
                 [0.33, 0.42, 0.18],
                 [0.53, 0.81, 0.92],
                 [0.71, 0.49, 0.86],
                 [0.25, 0.88, 0.82],
                 [0.5, 0.0, 0.0],
                 [0.0, 0.3, 0.3],
                 [1.0, 0.85, 0.73],
                 [0.29, 0.0, 0.51],
                 [0.7, 0.5, 0.35],
                 [0.44, 0.5, 0.56],
                 [0.25, 0.41, 0.88],
                 [0.0, 0.5, 0.0],
                 [0.56, 0.27, 0.52],
                 [1.0, 0.84, 0.0],
                 [1.0, 0.5, 0.31],
                 [0.85, 0.57, 0.94]]
    color = []
    color_box = [0.53, 0.81, 0.92]
    polygons = []
    boxes = []
    for box in tgt['boxes'].cpu():
        unnormbbox = box * torch.Tensor([W, H, W, H])
        unnormbbox[:2] -= unnormbbox[2:] / 2
        [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
        boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(color_box)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', linestyle="--", edgecolors=color, linewidths=1.5)
    ax.add_collection(p)

    if 'keypoints' in tgt:

        sks = np.array(keypoint_skeleton)
        # import pdb;pdb.set_trace()
        if sks !=[]:
            if sks.min()==1:
                sks = sks - 1

        for idx, ann in enumerate(tgt['keypoints']):
            kp = np.array(ann.cpu())
            Z = kp[:num_kpts*2] * np.array([W, H] * num_kpts)
            x = Z[0::2]
            y = Z[1::2]
            if len(color) > 0:
                c = color[idx % len(color)]
            else:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]

            for sk in sks:
                plt.plot(x[sk], y[sk], linewidth=1, color=c)

            for i in range(num_kpts):
                c_kpt = color_kpt[i]
                plt.plot(x[i], y[i], 'o', markersize=4, markerfacecolor=c_kpt, markeredgecolor='k', markeredgewidth=0.5)
    ax.set_axis_off()
    savename = os.path.join(output_dir, "pred.jpg")
    print("savename: {}".format(savename))
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    plt.savefig(savename, dpi=dpi)
    plt.close()




def load_image(image_pil):
    # load image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = Config.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # print(load_res)
    _ = model.eval()
    return model

keypoint_dict = globals()["face"]
keypoint_text_prompt = keypoint_dict.get("keypoints")
keypoint_skeleton = keypoint_dict.get("skeleton")
instance_text_prompt = "face"
box_threshold = 0.1
iou_threshold = 0.6

model = load_model("./xpose/config_model/UniPose_SwinT.py", "./weights/swint.pth", cpu_only=False)

def get_unipose_output(model, image, instance_text_prompt,keypoint_text_prompt, box_threshold,iou_threshold,with_logits=True, cpu_only=False, size=(512, 512)):
    # instance_text_prompt: A, B, C, ...
    # keypoint_text_prompt: skeleton
    instance_list = instance_text_prompt.split(',')

    device = "cuda" if not cpu_only else "cpu"

    # clip_model, _ = clip.load("ViT-B/32", device=device)
    ins_text_embeddings, kpt_text_embeddings = text_encoding(instance_list, keypoint_text_prompt, model.clip_model, device)
    target={}
    target["instance_text_prompt"] = instance_list
    target["keypoint_text_prompt"] = keypoint_text_prompt
    target["object_embeddings_text"] = ins_text_embeddings.float()
    kpt_text_embeddings = kpt_text_embeddings.float()
    kpts_embeddings_text_pad = torch.zeros(100 - kpt_text_embeddings.shape[0], 512,device=device)
    target["kpts_embeddings_text"] = torch.cat((kpt_text_embeddings, kpts_embeddings_text_pad), dim=0)
    kpt_vis_text = torch.ones(kpt_text_embeddings.shape[0],device=device)
    kpt_vis_text_pad = torch.zeros(kpts_embeddings_text_pad.shape[0],device=device)
    target["kpt_vis_text"] = torch.cat((kpt_vis_text, kpt_vis_text_pad), dim=0)
    # import pdb;pdb.set_trace()
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], [target])


    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    keypoints = outputs["pred_keypoints"][0][:,:2*len(keypoint_text_prompt)] # (nq, n_kpts * 2)
    # filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    keypoints_filt = keypoints.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    keypoints_filt = keypoints_filt[filt_mask]  # num_filt, 4


    keep_indices = nms(box_ops.box_cxcywh_to_xyxy(boxes_filt), logits_filt.max(dim=1)[0], iou_threshold=iou_threshold)

    # Use keep_indices to filter boxes and keypoints
    filtered_boxes = boxes_filt[keep_indices].cpu().numpy()
    H, W = size
    filtered_boxes = filtered_boxes * np.array([W, H, W, H])
    filtered_boxes[:, :2] -= filtered_boxes[:, 2:] / 2
    filtered_boxes[:, 2:] += filtered_boxes[:, :2]

    # print(filtered_boxes, image.shape)

    filtered_keypoints = keypoints_filt[keep_indices].cpu().numpy().reshape(-1, 68, 2) * np.array([W, H])


    return filtered_boxes,filtered_keypoints


def detect_one_image(raw_image):
    
    image = load_image(Image.fromarray(raw_image.astype('uint8')))
    # run model
    boxes_filt,keypoints_filt = get_unipose_output(
        model, image, instance_text_prompt, keypoint_text_prompt, box_threshold, iou_threshold, cpu_only=False, size=raw_image.shape[:2]
    )
    return boxes_filt, keypoints_filt


if __name__ == '__main__':
    import cv2
    im = cv2.imread('test.png')
    detect_one_image(im)