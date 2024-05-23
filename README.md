## 更新逻辑
尽可能把可复用的定义与方法扔到animateddiff文件夹中，仅使用yaml文件进行配置

## Requirements

see 'environment.yaml'

此版本支持最新版的diffusers库
需要按照以下步骤配置环境
```
conda env create -f environment.yaml
pip install -U openmim
# 自行安装torchvision cuda版本
# https://download.pytorch.org/whl/torch_stable.html
# wget https://download.pytorch.org/whl/cu117/torchvision-0.15.1%2Bcu117-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.15.1%2Bcu117-cp310-cp310-linux_x86_64.whl

mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"

自行安装cudatoolkit和deepspeed
conda install -c conda-forge cudatoolkit-dev -y
pip3 install deepspeed
```


## Training example
使用make train命令开始训练
```bash
make train num_nodes={node数量} num_gpus={每个node的gpu数量} local_rank={当前node的rank} config={config路径}
```
训练2D部分
```bash
make train num_nodes=1 num_gpus=8 local_rank=0 config=configs/training_me/train1_magic_catnoise_codeback_apt0_ZT_OffNoi.yaml
```
训练3D部分
```bash
make train num_nodes=1 num_gpus=8 local_rank=0 config=configs/training_me/train12_magic_catnoise_codeback_apt0_from2D25000step_warp05_DiffMotion_ZT_OffNoi.yaml
```

## Evalutaion
```bash 
make eval devices={用于eval的gpu ranks} config={config路径}
```
如
```bash 
make eval devices=0 config=./configs/prompts_me/infer12_magic_catnoise_CrossRefTemp_codeback_apt0_ZT_OffNoi.yaml
```
