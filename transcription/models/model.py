from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class FewShotExample:
    image_path: str
    transcription: str

@dataclass
class Transcription:
    text: str
    boxes: list[dict] = field(default_factory=list)
    raw_output: object = None

class OCRModel(ABC):
    DEFAULT_MODEL_ID: str # HuggingFace id of base checkpoint
    returns_boxes = False # True: Transcription.boxes are filled
    supports_prompting = False # True: accepts prompt and fewshot examples

    def __init__(self, model_id=None, prompt=None, examples=()):
        # model id is HuggingFace id or path of fine-tuned checkpoint, default is base model
        if (prompt or examples) and not self.supports_prompting:
            raise ValueError(f"{type(self).__name__} does not support custom prompts or few-shot examples.")

        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.prompt = prompt
        self.examples = list(examples)

    @abstractmethod
    def transcribe(self, image_path:str) -> Transcription:
        """ Read text on one image """