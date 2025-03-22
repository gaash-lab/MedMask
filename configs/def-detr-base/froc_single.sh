BATCH_SIZE=2
DATA_ROOT=/home/suhaib/Research/Drive/Datasets/BCD_DDSM

CUDA_VISIBLE_DEVICES=0 python -u froc_metric.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 2 \
--data_root ${DATA_ROOT} \
--source_dataset ddsm \
--target_dataset ddsm \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--resume /home/suhaib/Research/Drive/Outputs/outputs_DDSM/DMASTER/medmask_/model_best.pth \

