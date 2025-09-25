import os
import sys
import json
import time
import traceback

import tqdm
from ..fileloader.dataloader import *
from ..fileloader.qwen import *
from ..fileloader.google import *
from ..fileloader.llama import *

import argparse
def extract_prediction_paths(data):
    prediction_paths = []
    for prediction in data['prediction']:
        try:
            paths = prediction.split('\n')
            text_path = paths[0].replace("<PATH>", "").replace("</PATH>", "").replace("<SEP>", " -> ").strip().removeprefix("text_path: ").replace(" ","")
            image_path = paths[1].replace("<PATH>", "").replace("</PATH>", "").replace("<SEP>", " -> ").strip().removeprefix("image_path: ").replace(" ","")
            prediction_paths.append({'text_path': text_path, 'vision_path': image_path})
        except:
            traceback.print_exc()
            prediction_paths.append({'text_path': "", 'vision_path': ""})
    return prediction_paths



def generate_answer_list(inputfile,outputfile,model,ds):
    content=[]

    with open(inputfile) as f:
        for line in f:
            data=json.loads(line)
            content.append(data)

    with open(outputfile,"a") as f:
        cnt=0
        print(len(content))
        for line in tqdm.tqdm(content):
            cnt+=1
            if cnt==100:
                break
            res=generate_answer(line,model,ds)
            if res==None:
                continue
            f.write(json.dumps(res))
            f.write("\n")
            f.flush()



def generate_answer(line,model,ds):
    res = extract_prediction_paths(line)
    text = [item['text_path'] for item in res]
    vision=[item['vision_path'] for item in res]
    id = line["id"]
    item = ds.getitem(id)
    image_path = item.image
    predictions = []
    cnt=0
    for each in zip(text,vision):
        cnt+=1
        print(each)
        # prompt = f"""
        # Based on the text reasoning path "{each[0]}" and the visual reasoning path "{each[1]}", analyze the image to answer the question: "{item.question}".
        #
        # Use both paths as your primary guide for reasoning. Interpret the text path to understand the semantic relationship needed, and apply the visual path to identify relevant objects or features in the image. Combine both to reach a clear conclusion.
        #
        # Do not say the instruction is unclear or incomplete. Assume the paths are valid and sufficient. Avoid emojis, disclaimers, or speculative language. Be factual, concise, and directly derive the answer from the two reasoning paths.
        # """
        prompt = f""""Based on the text path {each[0]} and vision path {each[1]}, analyze the image step by step and answer the question: {item.question}.You must mention my aforementioned reasoning paths and derive the answer based on these reasoning paths."""

        messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        res = model.inf_with_message(messages)
        predictions.append({"text path": each[0], "explanation": res,"vision path":each[1]})
    result = {
            "id": item.id,
            "question": item.question,
            "prediction": predictions,
        }
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, required=True, )
    parser.add_argument("--total_parts", type=int, default=8)
    parser.add_argument("--data", type=str, default="fvqa",
                        choices=["fvqa", "aokvqa", "okvqa"],
                        help="Type of dataset to use")
    parser.add_argument("--model", type=str, default="qwen",
                        choices=["qwen", "gemma", "google", "llama", "intern"],
                        help="Type of model to use")
    parser.add_argument("--split")
    parser.add_argument("--test_num")
    args = parser.parse_args()
    generated=[]
    output_file=f"/root/autodl-tmp/RoG/qwen/src/superargs/dual_path/{args.data}/{args.model}/{args.split}/reasons.jsonl"
    if not os.path.exists(output_file):
        with open(output_file,"w") as f:
            pass

    with open(output_file,"r") as f:
        for line in f:
            try:
                generated.append(int(json.loads(line)["id"]))
            except:
                traceback.print_exc()

    if args.data == "fvqa":
        qapath = "/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/all_qs_dict_release.json"
        image_path = "/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/images"
        ds = dataf(qapath, image_path,"train")
    elif args.data == "okvqa":
        ds = datas("/root/autodl-tmp/RoG/qwen/data/OKVQA",split="train")

    if args.model == "qwen":
        model = qwenmod(modelpath="/root/autodl-tmp/RoG/qwen/multimodels/Qwen/qwenvl",)
    elif args.model == "gemma" or args.model == "google":
        model = googlemod(modelpath="/root/autodl-tmp/RoG/qwen/multimodels/google/gemma")
    elif args.model == "llama":
        model = llamamod(modelpath="/root/autodl-tmp/RoG/qwen/multimodels/meta-llama/llama")

    input_file=f"/root/autodl-tmp/RoG/qwen/src/superargs/dual_path/{args.data}/{args.model}/{args.split}/predictions_5_train.jsonl"
    cnt=0
    args.part=int(args.part)
    print(len(ds.combined))

    generate_answer_list(input_file,output_file,model,ds)



