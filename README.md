## <p align=center>MegActor: Harness the Power of Raw Video for Vivid Portrait Animation</p>

<p align=center>Shurong yang, Huadong Li, Juhao Wu, Minhao Jing, Linze Li, Renhe Ji, Jiajun Liang, Haoqiang Fan</p>

<p align=center>MEGVII Technology</p>

## News & TODO List
- **[✅2024.05.24]** Inference settings are released.

- **[❌]** Data curation pipeline to be released .

- **[❌]** Training setup to be released.
  
## Overview
![Model](https://github.com/megvii-research/MegFaceAnimate/assets/29685592/a3cf55a9-9838-400a-a2e3-281acca11b76)

MegActor is an intermediate-representation-free portrait animator that uses the original video, rather than intermediate features, as the driving factor to generate realistic and vivid talking head videos. Specifically, we utilize two UNets: one extracts the identity and background features from the source image, while the other accurately generates and integrates motion features directly derived from the original videos. MegActor can be trained on low-quality, publicly available datasets and excels in facial expressiveness, pose diversity, subtle controllability, and visual quality.


## Pre-generated results

https://github.com/megvii-research/MegFaceAnimate/assets/29685592/6d3edec7-1008-4fde-93ee-a0598114120b


## Preparation
* Environments
  
  Detailed environment settings should be found with environment.yaml
    ```
    conda env create -f environment.yaml
    pip install -U openmim
    
    mim install mmengine
    mim install "mmcv>=2.0.1"
    mim install "mmdet>=3.1.0"
    mim install "mmpose>=1.1.0"

    conda install -c conda-forge cudatoolkit-dev -y
    ```
* Dataset

  
  To be released.
  
* Pretrained weights
  
  Please find our pretrained weights at https://huggingface.co/HVSiniX/RawVideoDriven.
  Or simply use
    ```bash
    git clone https://huggingface.co/HVSiniX/RawVideoDriven && ln -s RawVideoDriven/weights weights
    ```
## Training
To be released.
## Inference
Currently only single-GPU inference is supported.

    CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/infer12_catnoise_warp08_power_vasa.yaml --source {source image path} --driver {driving video path}


## Acknowledgement
Many thanks to the authors of [mmengine](https://github.com/open-mmlab/mmengine), [MagicAnimate](https://github.com/magic-research/magic-animate), [Controlnet_aux](https://github.com/huggingface/controlnet_aux), and [Detectron2](https://github.com/facebookresearch/detectron2).



## Contact
If you have any questions, feel free to open an issue or contact us at lihuadong@megvii.com or wujuhao@megvii.com.

