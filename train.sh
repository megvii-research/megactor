# eval $(curl -s http://deploy.i.shaipower.com/httpproxy)
NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
# [ -z "$NCCL_IB_HCA"] && NCCL_IB_HCA=mlx4_1;
export NCCL_IB_HCA=$(echo $NCCL_IB_HCA | tr ' ' ',')
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_NET_GDR_READ=1
export NCCL_TREE_THRESHOLD=0
export OMP_NUM_THREADS=4
export CUDA_CACHE_MAXSIZE=2147483647
export CUDA_CACHE_PATH=/data/.cuda_cache

MASTER_ADDR=$1
TRAINPY=$2
TRAIN_CONFIG=$3

NUM_MACHINE=$4
NUM_GPU=$5
NODE_RANK=$6

# using
# !!! first "hostname -i"
# bash train.sh $(hostname -i) train2d_for_3d.py  1 8 0

PORT=${PORT:-"29500"}

# debug using
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_IB_HCA==mlx5_0

accelerate launch \
    --config_file configs/accelerate_deepspeed.yaml \
    --gpu_ids all --use_deepspeed \
    --num_machines $NUM_MACHINE \
    --num_processes $NUM_GPU \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR --main_process_port $PORT \
    --deepspeed_multinode_launcher standard \
    $TRAINPY \
    --config $TRAIN_CONFIG \

#!/usr/bin/env bash
# sudo apt install gawk
# eval $(curl -s http://deploy.i.shaipower.com/httpproxy)
# NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
# # [ -z "$NCCL_IB_HCA"] && NCCL_IB_HCA=mlx4_1;
# export NCCL_IB_HCA=$(echo $NCCL_IB_HCA | tr ' ' ',')
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TC=106
# export NCCL_SOCKET_IFNAME="eth0"
# export NCCL_IB_DISABLE=0
# export NCCL_IB_CUDA_SUPPORT=1
# export NCCL_NET_GDR_READ=1
# export NCCL_TREE_THRESHOLD=0
# export OMP_NUM_THREADS=4
# export CUDA_CACHE_MAXSIZE=2147483647
# export CUDA_CACHE_PATH=/data/.cuda_cache

# NNODES=${RLAUNCH_REPLICA_TOTAL:-2} ##Node nums
# NODE_RANK=${RLAUNCH_REPLICA:-1} ##Node rank of different machine
# ACC_CONFIG=$1
# CONFIG=$2
# GPUS=$3
# TRAINPY=$4
# PORT=${PORT:-"29500"}
# echo "node rank .. $NODE_RANK"
# MASTER_ADDR=${MASTER_ADDR:-"100.96.56.34"}
# if [[ $NODE_RANK == 0 ]];
# then
#   echo "Write the ip address of node 0 to the hostfile.txt"
#   ip addr | awk '/^[0-9]+: / {}; /inet.*global/ {print gensub(/(.*)\/(.*)/, "\\1", "g", $2)}' > hostfile.txt
#   # hostname -I > hostfile.txt
# else
#   sleep 5
# fi

# MASTER_ADDR=$(cat hostfile.txt)
# echo "MASTER_ADDR is : $MASTER_ADDR"
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# export NCCL_IB_HCA==mlx5_0

# accelerate launch --config_file $ACC_CONFIG --gpu_ids all --use_deepspeed --num_processes $GPUS \
# --num_machines $NNODES --machine_rank $NODE_RANK --main_process_ip $MASTER_ADDR --main_process_port $PORT \
# --deepspeed_multinode_launcher standard $TRAINPY --config $CONFIG
