
## Requirements

see 'environment.yaml'

you need to do this below
```
conda env create -f environment.yaml
pip install -U openmim
# https://download.pytorch.org/whl/torch_stable.html
# wget https://download.pytorch.org/whl/cu117/torchvision-0.15.1%2Bcu117-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.15.1%2Bcu117-cp310-cp310-linux_x86_64.whl

mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"

conda install -c conda-forge cudatoolkit-dev -y
pip3 install deepspeed
```

```bash
git clone https://huggingface.co/HVSiniX/RawVideoDriven && ln -s RawVideoDriven/weights weights
```

## Inference example

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/infer12_catnoise_warp08_power_vasa.yaml --source new_test_data/vasa_case1.mp4 --driver new_test_data/vasa_case1.mp4
```
