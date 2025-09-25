# multimodal/base.py
from abc import ABC, abstractmethod

class BaseMultiModalModel(ABC):
    def __init__(self, modelpath: str = "./",type="hf"):
        self.modelpath = modelpath
        self.type = type
        self._load_model(type=self.type)

    @abstractmethod
    def _load_model(self,type="vllm"):

        pass

    @abstractmethod
    def inf_question_image(self, question: str, image: str):

        pass

    @abstractmethod
    def inf_with_messages(self, messages: list):

        pass

    def inf_with_messages_llama(self, iamge,messages: list):

        pass

    def inf_with_score(self, question: str, image: str, max_new_tokens=128, num_beam=3):

        raise NotImplementedError("infwithscore is not implemented for this model.")