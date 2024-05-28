## <p align=center>MegActor: Harness the Power of Raw Video for Vivid Portrait Animation</p>

<p align=center>Shurong yang, Huadong Li, Juhao Wu, Minhao Jing, Linze Li, Renhe Ji, Jiajun Liang, Haoqiang Fan</p>

**<p align=center>MEGVII Technology</p>**


## News & TODO List
- **[✅2024.05.24]** Inference settings are released.

- **[❌]** Data curation pipeline to be released .

- **[❌]** Training setup to be released.
  
## Overview

  ![Model](https://github.com/megvii-research/MegFaceAnimate/assets/29685592/857c7a9f-6231-4e7f-bfce-1e279ba57c89)

MegActor is an intermediate-representation-free portrait animator that uses the original video, rather than intermediate features, as the driving factor to generate realistic and vivid talking head videos. Specifically, we utilize two UNets: one extracts the identity and background features from the source image, while the other accurately generates and integrates motion features directly derived from the original videos. MegActor can be trained on low-quality, publicly available datasets and excels in facial expressiveness, pose diversity, subtle controllability, and visual quality.


## Pre-generated results

https://github.com/megvii-research/MegFaceAnimate/assets/29685592/6d3edec7-1008-4fde-93ee-a0598114120b

https://github.com/megvii-research/MegFaceAnimate/assets/29685592/b2eb4e5f-c1fc-4874-8af9-c4b14afaea4b

https://github.com/megvii-research/MegFaceAnimate/assets/29685592/212c7669-0839-4260-a7c4-94606818e61f

https://github.com/megvii-research/MegFaceAnimate/assets/29685592/ce4e5c19-cdc7-435e-83f3-8bce39f0c04e

https://github.com/megvii-research/MegFaceAnimate/assets/29685592/45043598-4f9b-4490-9686-fa3be4c361c0

https://github.com/megvii-research/MegFaceAnimate/assets/29685592/c7d71435-c98a-42b6-9f59-c72cb49851a1

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
If you have any questions, feel free to open an issue or contact us at 15066146083@163.com, lihuadong@megvii.com or wujuhao@megvii.com.

