#!/bin/bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
  CUDA_VISIBLE_DEVICES=0 python src/gen_rule_path/gen_rule_path.py \
              --model qwen \
              --data okvqa \
              --split train \
              --n_beam 3 \
              --output_path "results/path"