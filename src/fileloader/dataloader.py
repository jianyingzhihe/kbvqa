import json
import os
import warnings
import re
import time
import pandas
import tqdm
from PIL import Image
import io
from collections import Counter
import string

def remove_punctuation_whitespace(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = re.sub(r'\s+', '', text)
    return text.lower()


class qapair():
    def __init__(self,question,answer,image,id):
        self.question=question
        self.answer=answer
        self.image=image
        self.id=id

    def choice(self,choice):
        self.choice=choice

    def duplicate(self,answer):
        self.duplicatedanswer=answer

def format_image_name(image_id,split="val"):
    return f"COCO_{split}2014_{image_id:012d}.jpg"

class datas():
    def __init__(self,datapath,split="val"):
        self.datatype="okvqa"
        self.split=split
        print("initialing the datas")
        question_path = os.path.join(datapath, f"OpenEnded_mscoco_{split}2014_questions.json")
        answer_path = os.path.join(datapath, f"mscoco_{split}2014_annotations.json")
        image_path = os.path.join(datapath, f"{split}2014")
        # ip="/home/z_wen/Kbvqa/data/okvqa_data/train/train2014"
        self.image_path=image_path
        self.question,self.answer=self.load_json(question_path, answer_path)
        self.processdatas(split=split)
        print("finish loading datas")

    def load_json(self, question_path, answer_path):
        with open(question_path) as f:
            data = json.load(f)
            question = data["questions"]
            with open(answer_path, encoding="utf8") as an:
                data = json.load(an)
                answer = data["annotations"]
            return question, answer

    def processdatas(self, split="val"):
        """
        {'id': 297147,
        'question': 'What sport can you use this for?',
        'image_path': '../data/OKVQA/val2014/COCO_val2014_000000297147.jpg',
        'answer': [{'answer_id': 1, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 2, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 3, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 4, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 5, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 6, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 7, 'raw_answer': 'motocross', 'answer_confidence': 'yes', 'answer': 'motocross'}, {'answer_id': 8, 'raw_answer': 'motocross', 'answer_confidence': 'yes', 'answer': 'motocross'}, {'answer_id': 9, 'raw_answer': 'riding', 'answer_confidence': 'yes', 'answer': 'ride'}, {'answer_id': 10, 'raw_answer': 'riding', 'answer_confidence': 'yes', 'answer': 'ride'}]}
        """
        self.combined = []
        i = 0
        for each in self.question:
            id = each["question_id"]
            for ans in self.answer:
                if ans["question_id"] == id:
                    temp = qapair(each["question"], ans["answers"],
                                  os.path.join(self.image_path, format_image_name(each["image_id"], split=self.split)),
                                  each["question_id"])
                    self.combined.append(temp)
                    break
        self.solve_answer()
        # print(self.combined[0].id,self.combined[0].question,self.combined[0].answer,self.combined[0].image)

    def solve_answer(self):
        self.depulicated_answer = []
        for each in self.combined:
            answer = each.answer
            answer_list = []
            raw_answer_list = []
            for a in answer:
                if a["answer"] not in answer_list:
                    answer_list.append(a["answer"])
                if a["raw_answer"] not in answer_list:
                    raw_answer_list.append(a["raw_answer"])
            each.duplicate(answer_list+raw_answer_list)
            self.depulicated_answer.append({"id": each.id, "answer": answer_list, "raw_answer": raw_answer_list})

    def getanswer(self, id):
        for each in self.combined:
            if each.id == id:
                return each.answer
        warnings.warn("didn't find answer which match the id")
        return None

    def getduplicatedanswer(self, id):
        for each in self.depulicated_answer:
            if each["id"] == id:
                return each
        warnings.warn("didn't find answer which match the id")

    def getquestion(self, id):
        for each in self.combined:
            if each.id == id:
                return each.question
        warnings.warn("didn't find question which match the id")
        return None

    def getimage(self, id):
        for each in self.combined:
            if each.id == id:
                return each.image
        warnings.warn("didn't find image which match the id")
        return None

    def get_name(self, data, name):
        res = data.get(name, '')
        if res == None:
            return str(None)
        elif type(res) == str:
            return res
        elif type(res) == list:
            return ",".join(res)

    def get_names(self, data, names):
        res = ""
        for each in names:
            res = res + self.get_name(data, each)
        return res

    def getitem(self, id):
        for each in self.combined:
            if str(each.id) == str(id):
                return each
        warnings.warn("didn't find image which match the id")
        return None

    def calculate_score_dict(self, input_list):
        counts = Counter(input_list)
        score_dict = {element: min(count / 3, 1) for element, count in counts.items()}
        score_dict = {k: round(v, 2) for k, v in score_dict.items()}
        return score_dict

    def calc_answer_score(self):
        self.answer_score = []
        for item in self.combined:
            answer = [each["answer"] for each in item.answer]
            raw_answer = [each["raw_answer"] for each in item.answer]
            score_answer = self.calculate_score_dict(answer)
            score_raw_answer = self.calculate_score_dict(raw_answer)
            self.answer_score.append(
                {"id": item.id, "answer_score": score_answer, "raw_answer_score": score_raw_answer})

    def getanswerscore(self, id):
        for each in self.answer_score:
            if str(each["id"]) == str(id):
                return each
        warnings.warn("didn't find answer which match the id")
        return None

    def hit_rate(self, data, id):
        res = self.getduplicatedanswer(id)
        answer = res["answer"]
        raw_answer = res["raw_answer"]
        names = ["predicted_answer", "prediction", "answer"]
        processed_pred = remove_punctuation_whitespace(data)
        for ans in answer:
            ans = remove_punctuation_whitespace(ans)
            if ans in processed_pred:
                return True
        for ans in raw_answer:
            ans = remove_punctuation_whitespace(ans)
            if ans in processed_pred:
                return True
        return False

    def acc_rate(self, prediction, id):

        scorelist = self.getanswerscore(id)
        answer_score = scorelist["answer_score"]
        raw_answer_score = scorelist["raw_answer_score"]
        prediction = remove_punctuation_whitespace(prediction)
        sc1 = 0
        sc2 = 0
        for key in answer_score:
            temp = remove_punctuation_whitespace(key)
            if temp in prediction:
                sc1 += answer_score[key]
        for key in raw_answer_score:
            temp = remove_punctuation_whitespace(key)
            if temp in prediction:
                sc2 += raw_answer_score[key]
        return min(1, max(sc1, sc2))

    def evaluate_jsonl(self, jsonl_path):
        total_count = 0
        right = []
        false = []
        hit = 0
        acc = 0
        self.calc_answer_score()
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                id = data.get("id")
                names = ["predicted_answer", "prediction", "answer"]
                predicted_answer = self.get_names(data, names)
                acc_score = self.acc_rate(predicted_answer, id)
                hit_score = self.hit_rate(predicted_answer, id)

                acc += acc_score

                if hit_score:
                    hit += 1
                    right.append(id)
                else:
                    false.append(id)
                total_count += 1

        accuracy_rate = (acc / total_count * 100) if total_count > 0 else 0.0
        hit_rate = (hit * 100 / total_count) if total_count > 0 else 0.0
        print(f"hit:{hit_rate}, {hit} / {total_count}")
        print(f"acc: {accuracy_rate:.2f}%")
        intersection = set(right) & set(false)
        if intersection:
            print(f"warning {intersection}")
        return accuracy_rate, hit_rate, hit, total_count, right, false




def format(image_id, split="val"):
        return f"abstract_v002_{split}2015_{image_id:012d}.png"


class datap():
    def __init__(self,datapath,split="val"):
        self.datatype="aokvqa"
        print("initializing data")
        ds=pandas.read_parquet(datapath)
        self.lenth=len(ds)
        # print(self.lenth)
        # print(type(ds))
        # print(ds.iloc[0])
        self.combined=[]
        for i in range(self.lenth):
            imgpath=os.path.join("/root/autodl-tmp/RoG/qwen/data/AOKVQA/img",ds.iloc[i]["question_id"]+".png")
            if os.path.exists(imgpath):
                temp = qapair(ds.iloc[i]["question"], ds.iloc[i]["direct_answers"], imgpath, ds.iloc[i]["question_id"])
                temp.choice(ds.iloc[i]["choices"])
            else:
                print("find nothing")
                image_bytes = ds.iloc[i]["image"]['bytes']
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                temp=qapair(ds.iloc[i]["question"],ds.iloc[i]["direct_answers"],image,ds.iloc[i]["question_id"])
                temp.choice(ds.iloc[i]["choices"])
            self.combined.append(temp)
        print("loading finish")

    def createimg(self):
        for each in tqdm.tqdm(self.combined):
            imgpath=os.path.join("/root/autodl-tmp/RoG/qwen/data/AOKVQA/img",each.id+".png")
            each.image.save(imgpath)
            each.image=imgpath


    def getanswer(self,image_id):
        for each in self.combined:
            if each.id == image_id:
                return each.answer
        warnings.warn("didn't find answer whitch match the id")
        return None

    def evaluate_jsonl(self, jsonl_path):
        correct_count = 0
        total_count = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                id=data.get("id")
                answers=self.getanswer(id)

                predicted_answer = data.get('predicted_answer', '')
                if not isinstance(predicted_answer, str):
                    predicted_answer = str(predicted_answer)

                processed_pred = re.sub(r'\s+', '', predicted_answer.lower())
                answers=answers.replace("[","").replace("]","").split(",")


                for each in answers:
                    temp=each.replace("'", "").replace(" ", "")
                    # print(temp)
                    if temp in processed_pred :
                        correct_count += 1
                        break
                total_count += 1

        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
        print(f" {correct_count} / {total_count}")
        print(f" 准确率: {accuracy:.2f}%")

        return correct_count, total_count, accuracy

class datav():#给vqa用
    def __init__(self, datapath, split="val"):
        self.ans_path = f"abstract_v002_{split}2017_annotations.json"
        self.que_path = f"OpenEnded_abstract_v002_{split}2017_questions.json"
        self.image_path = os.path.join(datapath,f"scene_img_abstract_v002_{split}2017")
        inputanswer = os.path.join(datapath, self.ans_path)
        inputquestion = os.path.join(datapath, self.que_path)
        self.combined = []
        print("data initial")
        with open(inputanswer, "r") as f1:
            with open(inputquestion, "r") as f2:
                data = json.load(f1)
                que = json.load(f2)
                answers = data["annotations"]
                question = que["questions"]
                # print(len(question))
                # print(len(answers))
                time1 = time.time()
                for que in question:
                    question_id = que["question_id"]
                    for ans in answers:
                        if ans["question_id"] == question_id:
                            temp = qapair(que["question"], ans["answers"],
                                          os.path.join(self.image_path, format(ans["image_id"])), question_id)
                            self.combined.append(temp)
                            break
                print(f"dataload finish,cost {time.time() - time1}s")
                print(len(self.combined))
                # for each in self.combined:
                #     print(each.answer[0]["answer"])


class dataf():
    def __init__(self, qapath,imagepath, split="val"):
        self.datatype="fvqa"
        with open(qapath, "r") as f1:
            self.all = []
            data = json.load(f1)
            # print(type(data))
            for each in data:
                temp=qapair(
                        data[each]["question"],
                    data[each]["answer"],
                    os.path.join(imagepath,data[each]["img_file"]),
                    each
                    )
                temp.duplicate([data[each]["answer"]])
                self.all.append(temp)
        self.length=len(self.all)
        self.num_train=self.length*4/5
        self.num_val=self.length-self.num_train
        self.train=[]
        self.val=[]
        for i in range(self.length):
            if i< self.num_train:
                self.train.append(self.all[i])
            else:
                self.val.append(self.all[i])
        if split=="val":
            self.combined=self.val
        elif split=="train":
            self.combined=self.train

    def getquestion(self,id):
        for each in self.all:
            if each.id == id:
                return each.question
        warnings.warn(f"didn't find question whitch match the {id}")
        return None

    def getanswer(self,id):
        for each in self.all:
            if each.id == id:
                return each.answer
        warnings.warn("didn't find answer whitch match the id")
        return None

    def getimage(self,id):
        for each in self.all:
            if each.id == id:
                return each.image
        warnings.warn("didn't find image whitch match the id")
        return None

    def getitem(self,id):
        for each in self.all:
            if each.id == id:
                return each
        warnings.warn("didn't find answer whitch match the id")
        return None

    def evaluate_jsonl(self, jsonl_path,split="val"):
        correct_count = 0
        total_count = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                id=data.get("id")
                answers=self.getanswer(id)
                y=0
                for each in self.val:
                    if each.id == id:
                        y=1
                if y==0:
                    continue
                predicted_answer = data.get('answer', '')+" ".join(data.get('prediction', ''))+data.get("predicted_answer", "")
                if not isinstance(predicted_answer, str):
                    predicted_answer = str(predicted_answer)

                processed_pred = re.sub(r'\s+', '', predicted_answer.lower())
                answers=answers.replace("[","").replace("]","").split(",")

                flag=1
                temp=answers[0]
                temp=temp.replace("'", "").replace("a ", "").replace(" ", "")
                if temp.endswith("s"):
                    temp=temp[:-1]
                temp=temp.lower()
                if temp in processed_pred :
                    correct_count += 1
                    flag=0
                if flag==1:
                    # print([temp,processed_pred])
                    pass

                total_count += 1

        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
        print(f" {correct_count} /  {total_count}, acc: {accuracy:.2f}%")

        return correct_count, total_count, accuracy


if __name__ == "__main__":
    ds=datas("/root/autodl-tmp/RoG/qwen/data/OKVQA")

    ds.solve_answer()
    for each in ds.combined:
        print(each.answer)