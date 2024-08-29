<div  align=center><img src="https://github.com/megvii-research/MegFaceAnimate/assets/29685592/5687c444-d437-4387-8219-61392cfa0dcf" width="15%"></div>

## <p align=center>MegActor: Harness the Power of Raw Video for Vivid Portrait Animation</p>

<p align=center>Shurong Yang<sup>*</sup>, Huadong Li<sup>*</sup>, Juhao Wu<sup>*</sup>, Minhao Jing<sup>*†</sup>, Linze Li, Renhe Ji<sup>‡</sup>, Jiajun Liang<sup>‡</sup>, Haoqiang Fan</p>

**<p align=center>MEGVII Technology</p>**

  <p align=center><sup>*</sup>Equal contribution  <sup>†</sup>Lead this project <sup>‡</sup>Corresponding author</p>

<div align="center">
  <br>
  <a href='https://arxiv.org/abs/2405.20851'><img src='https://img.shields.io/badge/MegActor-Arxiv-red'></a>
  <a href='https://arxiv.org/abs/2408.14975'><img src='https://img.shields.io/badge/MegActorSigma-Arxiv-red'></a>
  <a href='https://megactor.github.io/'><img src='https://img.shields.io/badge/MegActor-ProjectPage-Green'></a>
  <a href='https://megactor-ops.github.io/'><img src='https://img.shields.io/badge/MegActorSigma-ProjectPage-Green'></a>
  <a href='https://f4c5-58-240-80-18.ngrok-free.app/'><img src='https://img.shields.io/badge/DEMO-RUNNING-<COLOR>.svg'></a>
  <a href='https://openbayes.com/console/public/tutorials/3IphFlojVlO'><img src='https://img.shields.io/badge/CONTAINER-OpenBayes-blue.svg'></a>
  <br>
</div>

## News & TODO List
- **[TODO]** The code of **MegActor-Sigma** will be cooming soon.
- **[🔥🔥🔥 2024.08.28]** [Arxiv](https://arxiv.org/abs/2408.14975) **MegActor-Sigma** paper are released.
- **[✨✨✨ 2024.07.02]** For ease of replication, we provide a 10-minute dataset available on [Google Drive](https://drive.google.com/drive/folders/1GVhCd3syxl2-oqF7TiPyoy7VrWJXbrQs?usp=drive_link), which should yield satisfactory performance..
- **[🔥🔥🔥 2024.06.25]** **Training setup released.** Please refer to [Training](https://github.com/megvii-research/megactor/edit/main/README.md#training) for details.
- **[🔥🔥🔥 2024.06.25]** Integrated into [OpenBayes](https://openbayes.com/), see the [demo](https://openbayes.com/console/public/tutorials/3IphFlojVlO). Thank **OpenBayes** team!
- **[🔥🔥🔥 2024.06.17]** [Demo Gradio Online](https://f4c5-58-240-80-18.ngrok-free.app/) are released .
- **[🔥🔥🔥 2024.06.13]** [Data curation pipeline](https://github.com/megvii-research/megactor/tree/main/data_processing) are released .
- **[🔥🔥🔥 2024.05.31]** [Arxiv](https://arxiv.org/abs/2405.20851) **MegActor**  paper are released.
- **[🔥🔥🔥 2024.05.24]** Inference settings are released.


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
* Dataset.
   * For a detailed description of the data processing procedure, please refer to the accompanying below. [Data Process Pipeline](https://github.com/megvii-research/megactor/tree/main/data_processing)
   * You may refer to a 10-min dataset in this format at [Google Drive](https://drive.google.com/drive/folders/1GVhCd3syxl2-oqF7TiPyoy7VrWJXbrQs?usp=drive_link).
  
* Pretrained weights
  
  Please find our pretrained weights at https://huggingface.co/HVSiniX/RawVideoDriven.
  Or simply use
  
    ```bash
    git clone https://huggingface.co/HVSiniX/RawVideoDriven && ln -s RawVideoDriven/weights weights
    ```
    
## Training
We currently support two-stage training on single node machines.

Stage1(Image training):
```
bash train.sh train.py ./configs/train/train_stage1.yaml {number of gpus on this node}
```
Stage2(Video training):
```
bash train.sh train.py ./configs/train/train_stage2.yaml {number of gpus on this node}
```

## Inference
Currently only single-GPU inference is supported. We highly recommend that you use ```--contour-preserve``` arg the better preserve the shape of the source face.

    CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/inference/inference.yaml --source {source image path} --driver {driving video path} --contour-preserve


## Demo
For gradio interface, please run

    python demo/run_gradio.py

## BibTeX
```
@misc{yang2024megactorsigmaunlockingflexiblemixedmodal,
      title={MegActor-$\Sigma$: Unlocking Flexible Mixed-Modal Control in Portrait Animation with Diffusion Transformer}, 
      author={Shurong Yang and Huadong Li and Juhao Wu and Minhao Jing and Linze Li and Renhe Ji and Jiajun Liang and Haoqiang Fan and Jin Wang},
      year={2024},
      eprint={2408.14975},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.14975}, 
}
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

**If you're seeking an internship and are interested in our work, please send your resume to wujuhao@megvii.com or lihuadong@megvii.com.**

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=megvii-research/MegActor&type=Date)](https://star-history.com/#megvii-research/MegActor&Date)







