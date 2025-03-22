BATCH_SIZE=1
DATA_ROOT=/home/suhaib/Research/Drive/Datasets/BCD_INBreast
OUTPUT_DIR=/home/suhaib/Research/fusion/D-MASTER/configs/weights

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 2 \
--conf-threshold 0.05 \
--data_root ${DATA_ROOT} \
--source_dataset inbreast \
--target_dataset inbreast \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/ddsm2inhouse_teaching.pth \
