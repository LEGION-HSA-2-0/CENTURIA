from .florence2_model import Florence2Model
from .trocr_model import TrocrModel
from .olmocr_model import OlmocrModel
from .olmocr2_model import Olmocr2Model
from .lightonocr_model import LightonOcrModel

class ModelFactory:
    models = {
        "florence2": Florence2Model,
        "trocr": TrocrModel,
        "olmocr": OlmocrModel,
        "olmocr2": Olmocr2Model,
        "lightonocr": LightonOcrModel
    }

    @classmethod
    def get_model(cls, model_name, finetuned_path=None, prompt=None, examples=()):
        model_class = cls.models.get(model_name.lower())
        
        if not model_class:
            raise ValueError(f"Model {model_name!r} not found. Available models: {', '.join(cls.models)}")
        
        return model_class(model_id=finetuned_path, prompt=prompt, examples=examples)
