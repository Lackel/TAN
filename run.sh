#!/usr/bin bash
for d in 'banking' 'clinc' 'stackoverflow'
do 
python TAN.py \
    --dataset $d \
    --gpu_id 0 \
    --known_cls_ratio 0.75 \
    --cluster_num_factor 1 \
    --seed 0 \
    --freeze_bert_parameters \
    --save_model \
    --pretrain
done