N_GPUS=1
BATCH_SIZE=2
DATA_ROOT=/home/suhaib/Research/Drive/Datasets/BCD_RSNA
OUTPUT_DIR=/home/suhaib/Research/Drive/Outputs/outputs_RSNA/DMASTER/deformable_detr
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 2 \
--dropout 0.1 \
--data_root ${DATA_ROOT} \
--source_dataset rsna \
--target_dataset rsna \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--epoch 20 \
--epoch_lr_drop 40 \
--mode single_domain \
--output_dir ${OUTPUT_DIR} \
--resume /home/suhaib/Research/fusion/D-MASTER/configs/weights/source_only.pth \
