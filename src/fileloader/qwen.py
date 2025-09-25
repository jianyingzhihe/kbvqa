# multimodal/qwenmod.py
import os
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import Qwen2_5_VLForConditionalGeneration
from .dataloader import *
from .multi import BaseMultiModalModel
INSTRUCTION = """Given the image and question below, please generate a valid relation path (in <PATH>...</PATH> format) that can help answer the question using a knowledge graph.

Example:
Question: What sport can you use this for?
Answer: <PATH> vehicle.brand <SEP> vehicle.type <SEP> sports.use </PATH>

Now answer:
"""

class qwenmod(BaseMultiModalModel):
    def _load_model(self,type="hf"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.modelpath,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            repetition_penalty=1.5,
        )
        self.processor = AutoProcessor.from_pretrained(self.modelpath)

    def inf_question_image(self, question: str, image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text":f"<image>{question}"}
                ]
            }
        ]
        return self.inf_with_message(messages)

    def inf_with_message(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=384,repetition_penalty=3.0)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


    def inf_with_messages(self, messages: list):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = [], []
        for message in messages:
            for content in message.get('content', []):
                if content['type'] == 'image':
                    if isinstance(content['image'], str):
                        image = Image.open(content['image']).convert('RGB')
                    elif isinstance(content['image'], Image.Image):
                        image = content['image']
                    else:
                        raise ValueError("Unsupported image type")
                    image_inputs.append(image)
        print(text)
        print(image_inputs)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda").to(torch.bfloat16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512,repetition_penalty=3.0)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return output_text

    def inf_with_score(self, question: str, pictpath: str, max_new_tokens=128, num_beams=3):
        if pictpath=="0":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pictpath},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to("cuda").to(torch.bfloat16)

            print("Generating with beam search...")

            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_beam_groups=num_beams,
                diversity_penalty=1.0,
                num_return_sequences=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True
            )

            print("Generate finished.")

            generated_ids_trimmed = [out_ids[len(inputs.input_ids[0]):] for out_ids in output.sequences]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

            scores = output.sequences_scores.tolist()
            norm_scores = torch.softmax(output.sequences_scores, dim=0).tolist()

            results = [
                {
                    "answer": output_text[i].strip(),
                    "score": scores[i],
                    "normalized_score": norm_scores[i]
                } for i in range(len(output_text))
            ]

            return results
        else:
            messages = [
                {"role": "system","content": [{"type": "text", "text":INSTRUCTION }]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pictpath},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)



            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda").to(torch.bfloat16)

            print("Generating with beam search...")

            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_beam_groups=num_beams,
                diversity_penalty=1.0,
                num_return_sequences=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True
            )

            print("Generate finished.")

            generated_ids_trimmed = [out_ids[len(inputs.input_ids[0]):] for out_ids in output.sequences]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

            scores = output.sequences_scores.tolist()
            norm_scores = torch.softmax(output.sequences_scores, dim=0).tolist()

            results = [
                {
                    "answer": output_text[i].strip(),
                    "score": scores[i],
                    "normalized_score": norm_scores[i]
                } for i in range(len(output_text))
            ]

            return results