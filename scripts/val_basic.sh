model=qwen
device=0
data=okvqa
split=train
model_path=model/qwen
data_path=data/okvqa
output_path=results/basic

CUDA_VISIBLE_DEVICES=${device} python ./src/validate/validate_basic.py \
  --split ${split} \
  --output_path ${output_path} \
  --data ${data} \
  --data_path ${data_path} \
  --model ${model} \
  --model_path ${model_path}\

