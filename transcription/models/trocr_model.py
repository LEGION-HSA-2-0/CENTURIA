import easyocr
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .model import OCRModel, Transcription

class TrocrModel(OCRModel):

    DEFAULT_MODEL_ID = "microsoft/trocr-large-handwritten"
    returns_boxes = True

    MAX_NEW_TOKENS = 128

    def __init__(self, model_id=None, prompt=None, examples=()):
        super().__init__(model_id, prompt, examples)  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = TrOCRProcessor.from_pretrained(self.model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id).to(self.device).eval()
        
        # EasyOCR uses CRAFT
        self.detector = easyocr.Reader(["de"], gpu = self.device == "cuda")

    def transcribe(self, image_path: str) -> Transcription:
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        line_boxes = self.detect_lines(image)
        if not line_boxes:
            return Transcription("", [], {})

        crops = [image.crop(clip(box)) for box in line_boxes]
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True
            )

        texts = self.processor.batch_decode(output_ids, skip_special_tokens=True)

        boxes = [
            to_box(text.strip(), box)
            for box, text in zip(line_boxes, texts)
            if text.strip()
        ]
        text = " ".join(box["bbox_text"] for box in boxes)
        raw_output = {"craft_boxes": line_boxes, "trocr_texts": texts}
        return Transcription(text, boxes, raw_output)

    def detect_lines(self, image: Image.Image) -> list:
        horizontal_boxes, _ = self.detector.detect(np.array(image))
        return list(horizontal_boxes[0]) if horizontal_boxes else []

def clip(box) -> tuple[int, int, float, float]:
    x_min, x_max, y_min, y_max = box
    return max(0, int(x_min)), max(0, int(y_min)), x_max, y_max

def to_box(text:str, box) -> dict:
    x, y, x_max, y_max = clip(box)
    return {"bbox_text": text, "x": x, "y": y, "w": int(x_max - x), "h": int(y_max - y)}
        

