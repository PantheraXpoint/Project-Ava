#!/bin/bash

video_ids=(874 615 732 899 688 777 820 601 845 693)

for vid in "${video_ids[@]}"
do
    echo "Running with video_id=$vid"
    
    CUDA_VISIBLE_DEVICES=2 python graph_construction.py \
        --model qwenvl \
        --dataset videomme \
        --video_id $vid \
        --gpus 1
done
