<div  align=center><img src="https://github.com/megvii-research/MegFaceAnimate/assets/29685592/5687c444-d437-4387-8219-61392cfa0dcf" width="15%"></div>

## <p align=center>MegActor-Σ: Unlocking Flexible Mixed-Modal Control in Portrait Animation with Diffusion Transformer</p>

<p align=center>Shurong Yang<sup>*</sup>, Huadong Li<sup>*</sup>, Juhao Wu<sup>*</sup>, Minhao Jing<sup>*</sup>, Linze Li, Renhe Ji<sup>‡</sup>, Jiajun Liang<sup>‡</sup>, Haoqiang Fan</p>

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
- **[🔥🔥🔥 2024.10.14]** The code of **MegActor-Sigma** is released.
- **[🔥🔥🔥 2024.08.28]** [Arxiv](https://arxiv.org/abs/2408.14975) **MegActor-Sigma** paper are released.
- **[✨✨✨ 2024.07.02]** For ease of replication, we provide a 10-minute dataset available on [Google Drive](https://drive.google.com/drive/folders/1GVhCd3syxl2-oqF7TiPyoy7VrWJXbrQs?usp=drive_link), which should yield satisfactory performance..
- **[🔥🔥🔥 2024.06.25]** **Training setup released.** Please refer to [Training](https://github.com/megvii-research/megactor/edit/main/README.md#training) for details.
- **[🔥🔥🔥 2024.06.25]** Integrated into [OpenBayes](https://openbayes.com/), see the [demo](https://openbayes.com/console/public/tutorials/3IphFlojVlO). Thank **OpenBayes** team!
- **[🔥🔥🔥 2024.06.17]** [Demo Gradio Online](https://f4c5-58-240-80-18.ngrok-free.app/) are released .
- **[🔥🔥🔥 2024.06.13]** [Data curation pipeline](https://github.com/megvii-research/megactor/tree/main/data_processing) are released .
- **[🔥🔥🔥 2024.05.31]** [Arxiv](https://arxiv.org/abs/2405.20851) **MegActor**  paper are released.
- **[🔥🔥🔥 2024.05.24]** Inference settings are released.



https://github.com/user-attachments/assets/5b5b4ac4-67df-4397-9982-5b91e196097a

## Overview
Diffusion models have demonstrated superior performance in the field of portrait animation. However, current approaches relied on either visual or audio modality to control character movements, failing to exploit the potential of mixed-modal control. This challenge arises from the difficulty in balancing the weak control strength of audio modality and the strong control strength of visual modality.

To address this issue, we introduce MegActor-Σ: a mixed-modal conditional diffusion transformer (DiT), which can flexibly inject audio and visual modality control signals into portrait animation. Specifically, we make substantial advancements over its predecessor, MegActor, by leveraging the promising model structure of DiT and integrating audio and visual conditions through advanced modules within the DiT framework. To further achieve flexible combinations of mixed-modal control signals, we propose a "Modality Decoupling Control" training strategy to balance the control strength between visual and audio modalities, along with the "Amplitude Adjustment" inference strategy to freely regulate the motion amplitude of each modality.

Finally, to facilitate extensive studies in this field, we design several dataset evaluation metrics to filter out public datasets and solely use this filtered dataset to train MegActor-Σ.

Extensive experiments demonstrate the superiority of our approach in generating vivid portrait animations, outperforming previous closed-source methods.

The training code, model checkpoint and filtered dataset will be released, hoping to help further develop the open-source community.



## Preparation
* Environments
  
  Detailed environment settings should be found with env_sigma.yml
  * Linux
    ```
    conda env create -f env_sigma.yml
    pip install -U openmim
    
    mim install mmengine
    mim install "mmcv>=2.0.1"
    mim install "mmdet>=3.1.0"
    mim install "mmpose>=1.1.0"

    conda install -c conda-forge cudatoolkit-dev -y
    ```

* Dataset.
   * For a detailed description of the data processing procedure, please refer to the accompanying below. [Data Process Pipeline](https://github.com/megvii-research/megactor/tree/main/data_processing)
  
* Pretrained weights
  
  Please find our pretrained weights at https://huggingface.co/HVSiniX/RawVideoDriven.
  Or simply use
  
    ```bash
    git clone https://huggingface.co/HVSiniX/RawVideoDriven && ln -s RawVideoDriven/weights weights
    ```
    
## Training
We support 3-stage training on single node machines.

Stage1(Image training):
```
bash train.sh train.py ./configs/train/megactor-sigma/train_stage1.yaml {number of gpus on this node}
```
Stage2(Video training):
```
bash train.sh train.py ./configs/train/megactor-sigma/train_stage2.yaml {number of gpus on this node}
```
Stage3(Video training):
```
bash train.sh train.py ./configs/train/megactor-sigma/train_stage3.yaml {number of gpus on this node}
```

## Inference
  ### single-pair generation
    python eval_audio.py --config configs/inference/unet_attn_whis/inference.yaml --output-path ./generated_result/--num-steps 25 --guidance-scale 2 --source {source_path} --driver {driver_path}

  ### multi-pair generation
  Specify source and driver paths in corresponding config file.

    python eval_audio.py --config configs/inference/unet_attn_whis/inference.yaml --output-path ./generated_result/--num-steps 25 --guidance-scale 2


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






