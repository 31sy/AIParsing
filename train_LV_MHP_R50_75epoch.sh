export NGPUS=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
		--config-file "configs/LV_MHP/LV_R50_75epoch_IoUloss_IoUscore_parsing.yaml" 