#!/bin/bash

model=qwen
device=0
data=okvqa
split=train
input=results/okvqa/qwen/train/dual_path_3.josnl
model_path=model/qwen
data_path=data/okvqa
  CUDA_VISIBLE_DEVICES=${device} python src/gen_dual_path/gen_rule_path.py \
  --input ${input} \
  --model ${model} \
  --data ${data} \
  --split ${split} \
  --n_beam 3 \
  --model_path ${model_path}\
  --data_path ${data_path} \
  --output_path "results/path"



