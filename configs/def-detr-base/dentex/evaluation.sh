BATCH_SIZE=1
DATA_ROOT=/home/suhaib/Research/Drive/Datasets/dentex_enumeration
OUTPUT_DIR=/home/suhaib/Research/Drive/Outputs/outputs_DENTEX/DMASTER/deformable_detr

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 9 \
--data_root ${DATA_ROOT} \
--source_dataset dentex \
--target_dataset dentex \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/model_best.pth \
