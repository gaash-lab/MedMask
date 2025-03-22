BATCH_SIZE=1
DATA_ROOT=/home/suhaib/Research/Drive/Datasets/BCD_RSNA

CUDA_VISIBLE_DEVICES=0 python -u froc_predictions.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 2 \
--data_root ${DATA_ROOT} \
--source_dataset rsna \
--target_dataset rsna \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--resume /home/suhaib/Research/Drive/Outputs/outputs_RSNA/DMASTER/medmask/model_best.pth \
# --output_dir /home/suhaib/Research/Drive/Outputs/outputs_RSNA/DMASTER/medmask/froc_results