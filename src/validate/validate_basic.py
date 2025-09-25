import os
import traceback
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import argparse
from ..fileloader.dataloader import *
from ..fileloader.qwen import *
from ..fileloader.google import *
from ..fileloader.llama import *


def generate(model, dataset, outputdir):
    output_path = outputdir

    td = []
    if os.path.exists(outputdir):
        with open(outputdir) as f:
            for line in f:
                temp = json.loads(line)
                id = int(temp["id"])
                td.append(id)
        print(f"Found {len(td)} already processed items")
    else:
        print("Output file does not exist, starting from scratch")

    with open(output_path, 'a', encoding='utf-8') as f:

        for each in tqdm.tqdm(dataset.combined, desc="Processing images"):
            id = each.id
            if id in td:
                continue
            try:
                image = each.image
                question = each.question
                ref_answer=",".join(each.duplicatedanswer) if dataset.datatype=="okvqa" else each.duplicatedanswer
                instruction=f"Based on the image answer the question: {question}"

                prompt=f"""Below are an instruction that describes a task along with a reference answer. Using the reference answer as a guide, write your  own response.
                                ### Instruction:
                                {instruction}
                                ### Reference Answer:
                                {ref_answer}
                                ### Response:"""

                messages = [
                            {"role": "system",
                             "content": [
                                 {"type": "text",
                                  "text": prompt},
                             ]},
                            {"role": "user",
                             "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}
                        ]

                result = model.inf_with_message(messages)
                output_dict = {"messages":
                                   [{"role": "user", "content": f"<image>{question}"},
                                    {"role": "assistant", "content": result}],
                               "images": [image]}
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                traceback.print_exc()
        print(f"\nProcessed {len(td)} items")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal model validation script")
    parser.add_argument("--data", type=str, default="fvqa",
                        choices=["fvqa", "aokvqa", "okvqa"],
                        help="Type of dataset to use")
    parser.add_argument("--model", type=str, default="qwen",
                        choices=["qwen", "gemma", "google", "llama", "intern"],
                        help="Type of model to use")
    parser.add_argument("--output_path", type=str,
                        default="/root/autodl-tmp/RoG/qwen/res/temp.jsonl",
                        help="Output directory for results")
    parser.add_argument("--split",type=str,default="train")
    parser.add_argument("--model_path")
    parser.add_argument("--data_path")
    args = parser.parse_args()


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

    outputpath=os.path.join(args.output_path, f"{args.data}/{args.model}/{args.split}.jsonl")
    generate(model=model, dataset=ds, outputdir=outputpath)

