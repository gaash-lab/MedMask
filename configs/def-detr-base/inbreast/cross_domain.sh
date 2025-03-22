N_GPUS=1
BATCH_SIZE=2
DATA_ROOT=/home/suhaib/Research/Drive/Datasets/BCD_INBreast
OUTPUT_DIR=/home/suhaib/Research/Drive/Outputs/outputs_INBreast/DMASTER/medmask 
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26507 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 2 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset inbreast \
--target_dataset inbreast \
--mae_source_dataset inbreast \
--mae_target_dataset inbreast \
--embeddings_dir /home/suhaib/Research/Drive/Outputs/embeddings/train_inbreast \
--train_annotations_path /home/suhaib/Research/Drive/Datasets/BCD_INBreast/coco_uniform_ts/annotations/instances_train2017.json \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-5 \
--lr_backbone 2e-6 \
--lr_linear_proj 2e-6 \
--epoch 20 \
--epoch_lr_drop 20 \
--mode cross_domain_mae \
--output_dir ${OUTPUT_DIR} \
--resume /home/suhaib/Research/fusion/D-MASTER/configs/weights/ddsm2inhouse_teaching.pth \
