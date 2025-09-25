# Quick start

# step 1 :download dataset & model 
## (a)download okvqa,fvqa dataset 
```shell
modelscope download --dataset OmniData/OK-VQA
```

fvqa can be downloaded [here](https://github.com/wangpengnorman/FVQA)
## (b)download model :Qwen/Qwen2.5-VL-7B-Instruct,
```shell
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct
```

## (c) download model :LLM-Research/Llama-3.2-11B-Vision-Instruct
```shell
modelscope download --model LLM-Research/Llama-3.2-11B-Vision-Instruct
```

## (d) download model :LLM-Research/gemma-3-12b-it
```shell
modelscope download --model LLM-Research/gemma-3-12b-it
```

# step 2: inference
You can utilize the script `scripts/validate_basic.py` to perform inference on FVQA and OKVQA. We provide data loading and preprocessing scripts for both datasets located at `src/fileloader/dataloader.py`, as well as preprocessing scripts and inference interfaces for three models: `qwen.py`, `llama.py`, and `google.py`, which can be found in `src/fileloader/`.

```shell
bash ./scripts/val_basic.sh
```
# Step 3: Initiate KBVQA

Below is the complete workflow for our KBVQA process.

## (a) Generate Dual Paths

<pre style="background: none"><code class="language-shell" data-language="shell" identifier="edd5d104e335495788b50cf4e75cabcf-0" index="0" total="5">bash ./scripts/gen_path.sh</code></pre>

In this step, `n_beams` specifies the number of candidate paths to generate.

## (b) Generate Explanations Corresponding to the Paths

<pre style="background: none"><code class="language-shell" data-language="shell" identifier="edd5d104e335495788b50cf4e75cabcf-1" index="1" total="5">bash ./scripts/reasoning.sh</code></pre>

The input to this script should be the output file generated in the previous step.

## (c) Perform Preference Alignment and Select the Optimal Answer Using the Selector

<pre style="background: none"><code class="language-shell" data-language="shell" identifier="edd5d104e335495788b50cf4e75cabcf-2" index="2" total="5">bash ./scripts/selector.sh</code></pre>

The input to this script remains the output from the preceding step. The `input_num` parameter in this script must match or be less than the `n_beams` value set in step (a), which means selecting the corresponding top-k explanations generated in step (a) for further processing.

This step produces two output files: one recording the model selection decisions, and another intended for use in the subsequent training phase.

## (d) Conduct Training Using the Generated Script

<pre style="background: none"><code class="language-shell" data-language="shell" identifier="edd5d104e335495788b50cf4e75cabcf-3" index="3" total="5">bash ./scripts/train.sh</code></pre>

The input to this step is the training JSONL file output by the selector in the previous step.

## (e) Perform Inference Using the Trained Model

To run inference, specify the target model, dataset, and the path to the generated LoRA weights:

<pre style="background: none"><code class="language-shell" data-language="shell" identifier="edd5d104e335495788b50cf4e75cabcf-4" index="4" total="5">bash ./scripts/val_lora.sh</code></pre>

## Evaluate the Model's Generated Results

All generated output files from the models can be evaluated using the `dataset.evaluate_jsonl()` method. Here, `dataset` refers to either `datas` or `dataf`, which are the preprocessing classes for the OKVQA and FVQA datasets, respectively. We provide two evaluation metrics for OKVQA—Hit Rate and Accuracy—and one metric, Accuracy, for FVQA (due to the FVQA dataset providing only a single ground-truth answer per question).
