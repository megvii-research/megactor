## <p align=center>MegActor: Harness the Power of Raw Video for Vivid Portrait Animation</p>

<p align=center>Shurong yang, Huadong Li, Juhao Wu, Minhao Jing, Linze Li, Renhe Ji, Jiajun Liang, Haoqiang Fan</p>

<p align=center>MEGVII Technology</p>

## News & TODO List
- **[2024.05.24]** Inference settings are released.

- **[TBD]** Data curation pipeline to be released .

- **[TBD]** Training setup to be released.

## Pre-generated results

<table class="center">
<tr>
    <td width=100% style="border: none">
        <video controls autoplay loop src="https://github.com/megvii-research/MegFaceAnimate/assets/29685592/90959508-42b2-4657-a879-dcff22df700f
" muted="true"></video>
    </td>
</tr>
</table>

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
  
  TBD
  
* Pretrained weights
  
  Please find our pretrained weights at https://huggingface.co/HVSiniX/RawVideoDriven.
  Or simply use
    ```bash
    git clone https://huggingface.co/HVSiniX/RawVideoDriven && ln -s RawVideoDriven/weights weights
    ```
## Training
TBD
## Inference
Currently only single-GPU inference is supported.

    CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/infer12_catnoise_warp08_power_vasa.yaml --source {source image path} --driver {driving video path}


## Acknowledgement
Many thanks to the authors of [mmengine](https://github.com/open-mmlab/mmengine), [MagicAnimate](https://github.com/magic-research/magic-animate), [Controlnet_aux](https://github.com/huggingface/controlnet_aux), and [Detectron2](https://github.com/facebookresearch/detectron2).



## Contact
If you have any questions, feel free to open an issue or contact us at lihuadong@megvii.com or wujuhao@megvii.com.

