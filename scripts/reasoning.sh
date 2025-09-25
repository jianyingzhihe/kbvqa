model=qwen
device=0
data=okvqa
total_parts=1
split=train
for ((i=1;i<6;i++))
do
  CUDA_VISIBLE_DEVICES=${device} python ./src/predict/reason_path.py \
  --model ${model} \
  --data ${data} \
  --split ${split} \

done
