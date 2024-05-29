from controlnet_resource.dense_aux.densepredictor import DensePosePredictor
from controlnet_aux_lib import DWposeDetector
import cv2
import numpy as np

class DenseDWposePredictor:
    def __init__(self, device, resolution = [512, 512]) -> None:
        det_config = './controlnet_resource/controlnet_aux/yolox_config/yolox_l_8xb8-300e_coco.py'
        det_ckpt = './weights/aux/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
        pose_config = './controlnet_resource/controlnet_aux/dwpose_config/dwpose-l_384x288.py'
        pose_ckpt = './weights/aux/dw-ll_ucoco_384.pth'
        dense_weight_path = "./weights/aux/densepose_model.pkl"
        self.resolution = resolution
        self.dwpose_model = DWposeDetector(
            det_config=det_config,
            det_ckpt=det_ckpt,
            pose_config=pose_config,
            pose_ckpt=pose_ckpt,
            device=device)
        self.dense_model = DensePosePredictor(device = device if isinstance(device, int) or isinstance(device, str) else device.index, model_weights_path=dense_weight_path)

    def __call__(self, img):
        dense_frame = self.dense_model(img, convert_rgb=False)
        dw_frame, dw_frame_all = self.dwpose_model(img, output_type='np', image_resolution=self.resolution[0])

        # dense_res =  dense_frame == np.array([84, 1, 68], dtype='uint8')

        dense_res =  (dense_frame[:, :, 0] != 84)[:, :, None] # foreground of densepose
        dense_res_face = (dense_frame[:, :, 0] == 24) | (dense_frame[:, :, 0] == 37)
        
        dw_res = (dw_frame[:, :, 0] != 0)[:, :, None] # pose of dwpose
        dw_res *= dense_res_face[:, :, None] # only save dwpose on body
        

        # cv2.imwrite('dense_res_face.png', (dense_res_face * 255).astype('uint8'))
        # cv2.imwrite('dense_frame.png', dense_frame)
        # exit(0)
        
        dense_frame = cv2.cvtColor(dense_frame, cv2.COLOR_BGR2RGB)
        dwpose_result = dw_frame * dw_res
        densepose_dwpose_result = dense_frame * (dw_res == 0) + dw_frame * dw_res
        dw_frame_all = dw_frame_all * (dw_res == 0) + dw_frame * dw_res
        # dw_frame_all = dw_frame_all * (dense_res_face[:, :, None] == 0) + dw_frame * dense_res_face[:, :, None]
        return {
            "origin": np.array(img), 
            "origin_control": np.concatenate([np.array(img)[None, ...], densepose_dwpose_result[None, ...]], axis=0), 
            'densepose': dense_frame,
            'dwpose': dwpose_result,
            "dwpose_all": dw_frame_all,
            'densepose_dwpose': densepose_dwpose_result,
            'foreground': dense_res * img,
            'background': (dense_res == 0) * img,
            'foreground_mask': dense_res * 1.,
            'background_mask': (dense_res == 0) * 1.,
            "densepose_dwpose_concat": np.concatenate([dense_frame, dw_frame[:, :, :1] * dw_res], axis=2)
        }
