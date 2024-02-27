#!/usr/bin bash
for s in 0 1 2
do 
python TAN.py \
    --dataset banking \
    --known_cls_ratio 0.75 \
    --cluster_num_factor 1 \
    --seed $s \
    --gpu_id 0 \
    --freeze_bert_parameters \
    --save_model \
    --pretrain
done