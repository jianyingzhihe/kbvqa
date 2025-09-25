import torch
from transformers import AutoProcessor

from modelscope import Gemma3ForConditionalGeneration
from .dataloader import *
from .multi import BaseMultiModalModel
INSTRUCTION = """Given the image and question below, please generate a valid relation path (in <PATH>...</PATH> format) that can help answer the question using a knowledge graph.

Example:
Question: What sport can you use this for?
Answer: <PATH> vehicle.brand <SEP> vehicle.type <SEP> sports.use </PATH>

Now answer:
"""

INSTRUCTION_R="""Please generate a valid relation path of the question that can be helpful for answering the following question: """

class googlemod(BaseMultiModalModel):
    def _load_model(self,type="hf",max_tokens=512):
        self.type=type
        if type=="hf":
            print(os.getcwd())
            print(self.modelpath)
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.modelpath,
                torch_dtype=torch.bfloat16,
                device_map="auto",

            )
            self.processor = AutoProcessor.from_pretrained(self.modelpath)

    def inf_question_image(self, question: str, image: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        return self.inf_with_messages(messages)
    def inf_with_messages(self, messages: list):
        return None

    def inf_with_message(self, messages: list):
        if self.type=="hf":
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            return decoded

    def inf_with_score(self, question: str, pictpath: str, max_new_tokens=512, num_beams=3):

        if pictpath == "0":
            messages = [
                {"role": "system", "content": [{"type": "text", "text": INSTRUCTION_R}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question}
                    ]
                }
            ]
            inputs = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # image = Image.open(pictpath).convert("RGB")
            # image.thumbnail((512, 512))

            inputs = self.processor(
                text=[inputs],
                # images=image,
                return_tensors="pt"
            ).to(self.model.device).to(torch.bfloat16)

            print("Generating with beam search...")

            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True
            )

            print("Generate finished.")

            generated_ids_trimmed = [out_ids[len(inputs.input_ids[0]):] for out_ids in output.sequences]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

            if num_beams == 1:
                results = [
                    {
                        "answer": output_text[0].strip(),
                        "score": 1,
                        "normalized_score": 1
                    }
                ]
            else:
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
                {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pictpath},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image = Image.open(pictpath).convert("RGB")
            image.thumbnail((512, 512))

            inputs = self.processor(
                text=[inputs],
                images=image,
                return_tensors="pt"
            ).to(self.model.device).to(torch.bfloat16)

            print("Generating with beam search...")

            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True
            )

            print("Generate finished.")

            generated_ids_trimmed = [out_ids[len(inputs.input_ids[0]):] for out_ids in output.sequences]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

            if num_beams == 1:
                results = [
                    {
                        "answer": output_text[0].strip(),
                        "score": 1,
                        "normalized_score": 1
                    }
                ]
            else:

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