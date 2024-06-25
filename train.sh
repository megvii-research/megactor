export OMP_NUM_THREADS=4
export CUDA_CACHE_MAXSIZE=2147483647
export CUDA_CACHE_PATH=/data/.cuda_cache

TRAINPY=$1
TRAIN_CONFIG=$2
NUM_GPU=$3


accelerate launch \
    --config_file configs/accelerate_deepspeed.yaml \
    --gpu_ids all --use_deepspeed \
    --num_processes $NUM_GPU \
    --deepspeed_multinode_launcher standard \
    $TRAINPY \
    --config $TRAIN_CONFIG \