import cv2
import numpy as np
import torch
from densepose import add_densepose_config
from densepose.vis.densepose_results import (
    DensePoseResultsFineSegmentationVisualizer as Visualizer,
)
from densepose.vis.extractor import DensePoseResultExtractor

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import os
import imageio

class DensePosePredictor:
    def __init__(self, device, model_weights_path, resolution = [512, 512]):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file("detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
        cfg.MODEL.WEIGHTS = model_weights_path
        cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(cfg)
    def __call__(self, img, show_results=False, convert_rgb=True):
        img = np.array(img)
        height, width = img.shape[0], img.shape[1]
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]
            results = DensePoseResultExtractor()(outputs)

        arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cv2.COLORMAP_VIRIDIS)
        out_frame = Visualizer(alpha=1, cmap=cv2.COLORMAP_VIRIDIS).visualize(arr, results)
        
        if convert_rgb:
            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)

        if show_results:
            return out_frame, results    
        else:
            return out_frame