train:
	bash train.sh $(shell hostname -i) train.py ${config} $(num_nodes) $(num_gpus) $(local_rank) > logs/$(shell date +"%Y-%m-%d-%T" ).log

eval:
	CUDA_VISIBLE_DEVICES=${devices} python3 eval_fix.py  --config ${config}