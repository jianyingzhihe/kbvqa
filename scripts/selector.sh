model=qwen
device=0
data=okvqa
split=train
input=results/okvqa/qwen/train/dual_path_3.josnl
model_path=model/qwen
data_path=data/okvqa
output_path=results

CUDA_VISIBLE_DEVICES=${device} python ./src/selector/selector.py \
  --split ${split} \
  --output_path ${output_path} \
  --data ${data} \
  --input ${input} \
  --model ${model} \
  --model_path ${model_path}\
  --data_path ${data_path} \
  --input_num 3 \
