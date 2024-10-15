import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import torch
import numpy as np
from PIL import Image

from ..util import HWC3, resize_image
from . import util
def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    # hack
    canvas = util.draw_bodypose(canvas, candidate, subset)
    canvas = util.draw_facepose(canvas, faces)

    return canvas

def get_dwpose_body(candidate, subset, H, W):
    # print("candidate and subset is", candidate.shape, subset.shape, candidate.dtype, subset.dtype)
    # print("candidate and subset unique is", np.unique(candidate), np.unique(subset))
    nums, keys, locs = candidate.shape
    candidate[..., 0] /= float(W)
    candidate[..., 1] /= float(H)
    body = candidate[:,:18].copy()
    body = body.reshape(nums*18, locs)
    score = subset[:,:18]
    
    for i in range(len(score)):
        for j in range(len(score[i])):
            if score[i][j] > 0.3:
                score[i][j] = int(18*i+j)
            else:
                score[i][j] = -1

    un_visible = subset<0.3
    candidate[un_visible] = -1

    foot = candidate[:,18:24]

    faces_all = candidate[:,24:92]
    faces_surface = candidate[:,24:41] # 只有脸部轮廓，没有五官关键点
    # hack
    faces = candidate[:,41:]
    # faces = candidate[:,41:84]

    hands = candidate[:,92:113]
    hands = np.vstack([hands, candidate[:,113:]])
    
    bodies = dict(candidate=body, subset=score)

    # hack
    # hands = []
    # bodies = []
    pose = dict(bodies=bodies, hands=hands, faces=faces_all)
    
    detected_map = draw_pose(pose, H, W)
    detected_map = HWC3(detected_map)
    
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    return detected_map