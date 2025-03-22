N_GPUS=1
BATCH_SIZE=1
DATA_ROOT=/home/suhaib/Research/Drive/mydataset/dentex_enumeration
OUTPUT_DIR=/home/suhaib/Research/Drive/OurOutputs/dentex/DMASTER/corss_domain 
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26507 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 8 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset dentex \
--target_dataset dentex \
--mae_source_dataset dentex \
--mae_target_dataset dentex \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-5 \
--lr_backbone 2e-6 \
--lr_linear_proj 2e-6 \
--epoch 20 \
--epoch_lr_drop 20 \
--mode cross_domain_mae \
--output_dir ${OUTPUT_DIR} \
--resume /home/suhaib/Research/Drive/OurOutputs/dentex/DMASTER/source_only/model_best.pth \
