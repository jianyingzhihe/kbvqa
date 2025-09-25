import tqdm
import os
import argparse
os.environ["TORCH_COMPILE_DISABLE"] = "1"
from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
from ..fileloader.dataloader import *
from ..fileloader.qwen import *
from ..fileloader.google import *
from ..fileloader.llama import *

def solve(daset, engine,output_file):

    td=[]
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
    with open(output_file, 'r') as f:
        for line in f:
            temp = json.loads(line)
            id = temp["id"]
            td.append(id)
    fw = open(output_file, 'a', encoding='utf-8')
    cnt=0
    for each in tqdm.tqdm(daset.combined):
        cnt+=1
        if each.id in td:
            continue
        infer_request = InferRequest(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a visual reasoning assistant. Given an image and a question, first identify relevant visual concepts and generate a plausible visual reasoning path (vision path). Then, retrieve or infer related background knowledge to construct a textual reasoning path (text path). Next, merge these two paths into a coherent fusion path that connects perception to knowledge. Finally, explain the fusion path step by step and use it to answer the question clearly and accurately."
                    },
                    {"role": "user", "content": f"<image>{each.question}"}
                ],
                images=[each.image]
            )

            # 推理
        try:
            resp_list = engine.infer([infer_request], request_config)
            response_text = resp_list[0].choices[0].message.content
            result = {
                    "id": each.id,
                    "question": each.question,
                    "image_path": each.image,
                    "predicted_answer": response_text,
                }
            fw.write(f"{json.dumps(result, ensure_ascii=False)}\n")
            fw.flush()
        except:
            print(each.id)

    fw.close()


#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path")
    parser.add_argument("--model_path")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--data_path")
    parser.add_argument("--split")
    parser.add_argument("--output_path")
    args = parser.parse_args()
    if args.data.lower()=="fvqa":
        ds=dataf(args.data_path,args.split)
    elif args.data.lower()=="okvqa":
        ds=datas(args.data_path,args.split)

    lora_checkpoint = safe_snapshot_download(args.lora_path)
    output_file = os.path.join(args.output_path,f"lora_res/{args.data}/{args.model}/{args.split}/{args.lora_path}.jsonl")
    template_type = None
    default_system = None
    model, tokenizer = get_model_tokenizer(args.model_path)
    model = Swift.from_pretrained(model, lora_checkpoint)
    template_type = template_type or model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=default_system)
    engine = PtEngine.from_model_template(model, template, max_batch_size=2)
    request_config = RequestConfig(max_tokens=2048, temperature=0)

    solve(ds, engine,output_file)
    # ds.evaluate_jsonl(output_file)