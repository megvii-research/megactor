export CUDA_VISIBLE_DEVICES=0

# python create_control_from_file.py \
#     --config ./configs/prompts_me/create_data.yaml \
#     > infer_cuda0.log
INFER_CONFIG=$1
python eval_fix.py \
    --config $INFER_CONFIG 

