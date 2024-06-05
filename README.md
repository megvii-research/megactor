<div  align=center><img src="https://github.com/megvii-research/MegFaceAnimate/assets/29685592/5687c444-d437-4387-8219-61392cfa0dcf" width="15%"></div>

## <p align=center>MegActor: Harness the Power of Raw Video for Vivid Portrait Animation</p>

<p align=center>Shurong Yang<sup>*</sup>, Huadong Li<sup>*</sup>, Juhao Wu<sup>*</sup>, Minhao Jing<sup>*†</sup>, Linze Li, Renhe Ji<sup>‡</sup>, Jiajun Liang<sup>‡</sup>, Haoqiang Fan</p>

**<p align=center>MEGVII Technology</p>**

  <p align=center><sup>*</sup>Equal contribution  <sup>†</sup>Lead this project <sup>‡</sup>Corresponding author</p>

  <br>
  <a href='https://arxiv.org/abs/2405.20851'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
  <a href='https://megactor.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
  <br>

## News & TODO List
- **[✅2024.05.24]** Inference settings are released.

- **[❌]** Data curation pipeline to be released .

- **[❌]** Training setup to be released.



## **MegActor Features:**

**Usability**: animates a portrait with video while ensuring **consistent motion**.

**Reproducibility**: fully open-source and trained on **publicly available** datasets.

**Efficiency**: ⚡**200 V100 hours** of training to achieve pleasant motions on portraits.


## Overview
  ![Model](https://github.com/megvii-research/MegFaceAnimate/assets/29685592/857c7a9f-6231-4e7f-bfce-1e279ba57c89)

MegActor is an intermediate-representation-free portrait animator that uses the original video, rather than intermediate features, as the driving factor to generate realistic and vivid talking head videos. Specifically, we utilize two UNets: one extracts the identity and background features from the source image, while the other accurately generates and integrates motion features directly derived from the original videos. MegActor can be trained on low-quality, publicly available datasets and excels in facial expressiveness, pose diversity, subtle controllability, and visual quality.


## Pre-generated results
https://github.com/megvii-research/MegFaceAnimate/assets/29685592/1b9dc77c-50da-48bd-bb16-8b2dd56d703f

https://github.com/megvii-research/MegFaceAnimate/assets/29685592/ce4e5c19-cdc7-435e-83f3-8bce39f0c04e

https://github.com/megvii-research/MegFaceAnimate/assets/29685592/c7d71435-c98a-42b6-9f59-c72cb49851a1

## Preparation
* Environments
  
  Detailed environment settings should be found with environment.yaml
  * Linux
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

## Demo
For gradio interface, please run

    python demo/run_gradio.py

## BibTeX
```
@misc{yang2024megactor,
      title={MegActor: Harness the Power of Raw Video for Vivid Portrait Animation}, 
      author={Shurong Yang and Huadong Li and Juhao Wu and Minhao Jing and Linze Li and Renhe Ji and Jiajun Liang and Haoqiang Fan},
      year={2024},
      eprint={2405.20851},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
Many thanks to the authors of [mmengine](https://github.com/open-mmlab/mmengine), [MagicAnimate](https://github.com/magic-research/magic-animate), [Controlnet_aux](https://github.com/huggingface/controlnet_aux), and [Detectron2](https://github.com/facebookresearch/detectron2).


## Contact
If you have any questions, feel free to open an issue or contact us at yangshurong6894@gmail.com, lihuadong@megvii.com or wujuhao@megvii.com.







