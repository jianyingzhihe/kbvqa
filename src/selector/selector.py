import sys
import os
import json
import time
import traceback
import warnings

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import tqdm
from ..fileloader.dataloader import *
from ..fileloader.qwen import *
from ..fileloader.google import *
from ..fileloader.llama import *
import argparse


prompt_text = f""""Based on the text path , analyze the image step by step and answer the question.You must mention my aforementioned reasoning paths and derive the answer based on these reasoning paths."""
prompt_vision= f""""Based on the vision path , analyze the image step by step and answer the question.You must mention my aforementioned reasoning paths and derive the answer based on these reasoning paths."""
prompt_dual=f""""Based on the vision path and the text path, analyze the image step by step and answer the question.You must mention my aforementioned reasoning paths and derive the answer based on these reasoning paths."""

INSTRUCTION_CHOOSE = """You are a machine that only outputs one letter: a, b, or c.
You compare three answers to the ground truth and pick the best one.
No reasoning. No explanation. No uncertainty.

Rules:
- Only output 'a', 'b', or 'c'.
- Never output words, punctuation, or newlines.
- If all are wrong, pick the least wrong.
- Output exactly one character.

Example:
{
  "a": "The sky is green.",
  "b": "The sky is blue.",
  "c": "The sky is red.",
  "ground_truth": "The sky is blue on a clear day."
}
output:
"b"

Now:
"""
def get_output_file(path,force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "a")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(str(results["id"]))
        fout = open(path, "w")
        return fout, processed_results


def extract_prediction_paths(data):
    prediction_paths = []
    for prediction in data['prediction']:
        try:
            paths = prediction.split('\n')
            text_path = paths[0].replace("<PATH>", "").replace("</PATH>", "").replace("<SEP>", " -> ").strip().removeprefix("text_path: ")
            image_path = paths[1].replace("<PATH>", "").replace("</PATH>", "").replace("<SEP>", " -> ").strip().removeprefix("image_path: ")
            prediction_paths.append({'text_path': text_path, 'vision_path': image_path})
        except:
            traceback.print_exc()
            prediction_paths.append({'text_path': "", 'vision_path': ""})
    return prediction_paths



def find_prediction(id,preditions):
    for prediction in preditions:
        if str(id) == str(prediction["id"]):
            return prediction["prediction"]
    warnings.warn(f"no find fusion_Path of id : {str(id)}")
    return [[],[],[]]

def get_best(prediction,model,item):
    if len(prediction) == 1:
        for each in item.duplicatedanswer:
            if remove_punctuation_whitespace(each) in remove_punctuation_whitespace(prediction[0]["explanation"][0]):
                return prediction[0]
        return None
    if len(prediction) >=2:
        hit_pre=[]
        for pre in prediction:
            for each in item.duplicatedanswer:
                if remove_punctuation_whitespace(each) in remove_punctuation_whitespace(pre["explanation"][0]):
                    hit_pre.append(pre)
                    break
        if len(hit_pre)==0:
            return None
        elif len(hit_pre)==1:
            return hit_pre[0]
        elif len(hit_pre)>1:
            if len(hit_pre)==2:
                prediction_input = {"a": (hit_pre[0]), "b": hit_pre[1],"ground_truth": item.answer}
            elif len(hit_pre)==3:
                prediction_input = {"a": hit_pre[0], "b": hit_pre[1],"c":hit_pre[2],"ground_truth": item.answer}
            elif len(hit_pre)==4:
                prediction_input={"a":hit_pre[0],"b":hit_pre[1],"c":hit_pre[2],"d":hit_pre[3],"ground_truth": item.answer}
            elif len(hit_pre)==5:
                prediction_input = {"a": hit_pre[0], "b": hit_pre[1], "c": hit_pre[2], "d": hit_pre[3],"e":hit_pre[4],"ground_truth": item.answer}

            path_prompt = INSTRUCTION_CHOOSE + json.dumps(prediction_input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": item.image,
                        },
                        {"type": "text", "text": path_prompt},
                    ],
                }
            ]

            choices_of_model = model.inf_with_message(messages)
            choices_of_model=remove_punctuation_whitespace(choices_of_model)
            try:
                if choices_of_model=="a":
                    print(choices_of_model)
                    return hit_pre[0]
                if choices_of_model=="b":
                    print(choices_of_model)
                    return hit_pre[1]
                if choices_of_model=="c":
                    print(choices_of_model)
                    return hit_pre[2]
                if choices_of_model=="d":
                    print(choices_of_model)
                    return hit_pre[3]
                if choices_of_model=="e":
                    print(choices_of_model)
                    return hit_pre[4]
                else:
                    return hit_pre[0]
            except:
                    return hit_pre[0]

def resolve(item):
    item["vision path"]=item["text path"][1]
    item["text path"]=item["text path"][0]
    return item

def main(args):
    print("Loading dataset...")
    output_file_choice = os.path.join(args.output_path, f"{args.data}/{args.model}/{args.split}/{args.input_num}_choice.jsonl")
    output_file_train=os.path.join(args.output_path, f"{args.data}/{args.model}/{args.split}/{args.input_num}_train.jsonl")
    fout_choice, processed_list = get_output_file(output_file_choice, force=False)
    fout_train,processed_list=get_output_file(output_file_train, force=True)
    predictions=[]

    if args.data.lower()=="fvqa":
        ds=dataf(args.data_path,args.split)
    elif args.data.lower()=="okvqa":
        ds=datas(args.data_path,args.split)

    if args.model == "qwen":
        model = qwenmod(modelpath=args.model_path)
    elif args.model == "gemma" or args.model == "google":
        model = googlemod(modelpath=args.model_path)
    elif args.model == "llama":
        model = llamamod(modelpath=args.model_path)

    print("Save results to:", args.output_path)

    with open(args.input,"r") as f:
        for line in f:
            data=json.loads(line.strip())
            predictions.append(data)
    for each in tqdm.tqdm(predictions):
        each["prediction"]=[resolve(item) for item in each["prediction"]]
        print(each["id"])
        item=ds.getitem(each["id"])
        raw_question = item.question
        exp_path_pair=each["prediction"]

        res=get_best(exp_path_pair[0:args.input_num],model,item)

        if not res:
            continue
        res={"text path":res["text path"],"vision path":res["vision path"],"explanation":res["explanation"]}

        result = {
                "id": item.id,
                "question": raw_question,
                "prediction": res,
                "answer":item.answer if args.data.lower()=="fvqa" else ",".join(item.duplicatedanswer)
            }
        answ=item.answer if args.data.lower()=="fvqa" else ",".join(item.duplicatedanswer)
        fout_choice.write(json.dumps(result) + "\n")
        fout_choice.flush()
        messages = {"messages": [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"<image>{item.question}"},
            {"role": "assistant",
             "content": json.dumps(res) + " Therefore, the possible answers include: " + answ}],
            "images": [item.image]
        }
        fout_train.write(json.dumps(messages) + "\n")
    fout_choice.close()
    fout_train.close()
    print("Prediction finished.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-modal QA prediction with Qwen2.5-VL")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split (train or val)")
    parser.add_argument("--output_path", type=str, default="results/multimodal", help="Directory to save predictions")
    parser.add_argument("--data", type=str, default="OKVQA", help="Dataset name")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct", help="Model name for saving results")
    parser.add_argument("--model_path")
    parser.add_argument("--data_path")
    parser.add_argument("--input_num",type=int)
    parser.add_argument("--input",type=str)
    args = parser.parse_args()

    main(args)