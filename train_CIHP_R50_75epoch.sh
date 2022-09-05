

export NGPUS=1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
		--config-file "configs/CIHP/CIHP_AIParsing_R50_75epoch_serialGE_parsingiou_iouloss.yaml" 


