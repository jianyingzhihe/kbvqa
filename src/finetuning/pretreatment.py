import argparse
import ast
import json
import sys
import traceback


sys.path.append('/root/autodl-tmp/RoG/qwen/src/')
from fileloader.dataloader import *


SYSTEM_PROMPT_EN=("You are a visual reasoning assistant. Please first identify the objects and their attributes in the image, then construct a reasonable relational path based on the question, and finally provide the answer.")
SYSTEM_PROMPT_V2="""You are a visual reasoning assistant that generates two reasoning paths: a text path based on linguistic knowledge and common sense, and a vision path built from observable visual content. You must construct both paths yourself, then use them to perform step-by-step reasoning, aligning what is known with what is seen, before arriving at a well-justified answer."""
ROG_PROMPT = ("Please generate a valid relation path of the question that can be helpful for answering the following question: ")

IMAGE_PATH_TEMPLATE = "/root/autodl-tmp/RoG/qwen/data/OKVQA/train2014/COCO_train2014_{:012d}.jpg"


def extract_predicted_paths(input_text):

    start_marker = "Predicted Paths:\n"
    end_marker = "]]"

    start_index = input_text.find(start_marker) + len(start_marker)
    end_index = input_text.find(end_marker)

    res=ast.literal_eval(input_text[start_index:end_index+2])
    # print(res)
    # a=input()

    return res

def get_worst_answer(explain,ground_answer):
    for each in explain:
        flag=0
        for ans in ground_answer:
            if remove_punctuation_whitespace(ans) in remove_punctuation_whitespace(each):
                flag=1
        if flag==0:
            return each
    return None

def get_best_answer(explain,ground_answer):
    for each in explain:
        for ans in ground_answer:
            if remove_punctuation_whitespace(ans) in remove_punctuation_whitespace(each):
                return each
    return None

def convert_jsonl_line(predicted_path,line, ground_answer,dataset,image_path,question):
    data = json.loads(line)
    generated_answers = data.get("prediction")
    best_answer=get_worst_answer(generated_answers,ground_answer)
    if not best_answer:
        return None
    # predicted_path={"vision path":[item["vision_path"] for item in predicted_path],"text path":[item["text_path"] for item in predicted_path]}

    predicted_path={"vision path":predicted_path[0]["vision_path"],"text path":predicted_path[0]["text_path"]}
    assistant_content = "\n\nPredicted Paths:\n" + json.dumps(predicted_path)+". " + best_answer
    if dataset.datatype == "okvqa":
        messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {"role": "user", "content": f"<image>{question}"},
        {"role": "assistant", "content": assistant_content +" Therefore, the possible answers include: " +",".join(ground_answer)},
    ]
    elif dataset.datatype == "fvqa":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V2},
            {"role": "user", "content": f"<image>{question}"},
            {"role": "assistant","content": assistant_content + " Therefore, the possible answers include: " + ground_answer[0]},
        ]
    return {
        "messages": messages,
        "images": [image_path]
    }


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


def process_jsonl_to_json(input_path, path_jsonl,output_path, dataset):
    results = []
    generated_rule_path=[]
    with open(path_jsonl, "r") as f:
        for line in f:
            data=json.loads(line.strip())
            generated_rule_path.append(data)
    with open(input_path, 'r', encoding='utf-8') as fin:
        i=0
        for idx, line in enumerate(fin):
            try:
                i += 1
                if dataset.datatype == "okvqa":
                    data=json.loads(line)
                    id=data["id"]
                    for each in generated_rule_path:
                        if str(each["id"])==str(id):
                            prediction_path = extract_prediction_paths(each)
                    item=dataset.getitem(id)
                    converted = convert_jsonl_line(prediction_path,line, item.duplicatedanswer,dataset,item.image,item.question)
                    if not converted:
                        continue

                    results.append(converted)

                elif dataset.datatype == "fvqa":
                    data=json.loads(line)
                    id=data["id"]
                    for each in generated_rule_path:
                        if str(each["id"])==str(id):
                            prediction_path = extract_prediction_paths(each)


                    item=dataset.getitem(id)
                    converted = convert_jsonl_line(prediction_path, line, [item.answer], dataset, item.image,
                                                   item.question)
                    if not converted:
                        continue
                    results.append(converted)



            except Exception as e:

                traceback.print_exc()

    with open(output_path, 'w', encoding='utf-8') as fout:
        for each in results:
            json.dump(each, fout, ensure_ascii=False)
            fout.write("\n")



# 示例调用
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run multi-modal QA prediction with Qwen2.5-VL")
    # parser.add_argument("--model", type=str, default="qwen")
    # parser.add_argument("--dataset", type=str, default="fvqa")
    # parser.add_argument("--split", type=str, default="val")
    # args = parser.parse_args()
    ds_okvqa=datas("/root/autodl-tmp/RoG/qwen/data/OKVQA",split="train")
    qapath = "/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/all_qs_dict_release.json"
    image = "/home/z_wen/Kbvqa/data/FVQA/new_dataset_release/images"
    ds_fvqa = dataf(qapath, image, "train")  #
    models=["gemma","qwen","llama"]
    datasets=["fvqa","okvqa"]
    for mod in models:
        for dataset in datasets:
            path_jsonl=f"/root/autodl-tmp/RoG/qwen/romg/{dataset}/{mod}/train/predictions_3_False_train_1.jsonl"
            input_jsonl = f"/root/autodl-tmp/RoG/qwen/results/multimodal/{dataset}/{mod}/train/predictions.jsonl"  # 替换为你的输入文件路径
            output_json = f"/root/autodl-tmp/RoG/qwen/src/ablation/2train/{dataset}_{mod}.jsonl"  # 输出为 .json 文件
            if dataset=="fvqa":
                ds=ds_fvqa
            elif dataset=="okvqa":
                ds=ds_okvqa
            process_jsonl_to_json(input_jsonl,path_jsonl, output_json, ds)