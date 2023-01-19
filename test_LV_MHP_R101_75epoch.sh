CUDA_VISIBLE_DEVICES=0 python -u tools/test_net_parsing.py \
    --config-file "configs/LV_MHP/LV_R101_75epoch_IoUloss_IoUscore_parsing.yaml" \
    TEST.IMS_PER_BATCH 1 \
    MODEL.WEIGHT ./checkpoints/LV_MHP_R101_75epoch/model_final.pth