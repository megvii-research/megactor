# 这个版本的绘制DWpose，只绘制肩膀和脖子连线，以及脸部连线，和所有脸部关键点

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import torch
import numpy as np
from PIL import Image
import math

from ..util import HWC3, resize_image
from . import util

eps = 0.01

def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    draw_indexs = [0, 1, 12, 13, 14, 15, 16] 

    for i in draw_indexs:
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    # hack
    canvas = draw_bodypose(canvas, candidate, subset)
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

    # faces_all = candidate[:,24:92]
    faces_all = candidate[:,24:84]
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