import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from .model import OCRModel, Transcription # Abstract Model

class Florence2Model(OCRModel):
    """
    Florence2-large in OCR-with-region mode: one text box per detected line 

    Uses the community checkpoint because it works with current transformers versions.
    
    Sources: 
    https://blog.roboflow.com/florence-2-ocr/
    https://huggingface.co/microsoft/Florence-2-large

    OCR with region output format: 
    {'<OCR_WITH_REGION>': {'quad_boxes': [[x1, y1, x2, y2, x3, y3, x4, y4], ...], 'labels': ['text1', ...]}}
    """
    DEFAULT_MODEL_ID = "florence-community/Florence-2-large"
    returns_boxes = True

    TASK = "<OCR_WITH_REGION>"
    MAX_NEW_TOKENS = 1024

    def __init__(self, model_id=None, prompt=None, examples=()):
        super().__init__(model_id, prompt, examples)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = Florence2ForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=self.dtype, 
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def transcribe(self, image_path:str) -> Transcription:
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        inputs = self.processor(text=self.TASK, images=image, return_tensors="pt").to(device=self.device)

        with torch.inference_mode(): 
            generated_ids = self.model.generate(
                input_ids = inputs["input_ids"],
                pixel_values = inputs["pixel_values"].to(self.dtype),
                max_new_tokens = self.MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task = self.TASK,
            image_size = image.size
        )

        regions = parsed_answer.get(self.TASK) or {}
        boxes = []
        for quad, label in zip(regions.get("quad_boxes", []), regions.get("labels", [])):
            x_coords, y_coords = quad[0::2], quad[1::2]

            x_min, y_min = min(x_coords), min(y_coords)

            boxes.append({
                "bbox_text": label,
                "x": int(x_min),
                "y": int(y_min),
                "w": int(max(x_coords) - x_min),
                "h": int(max(y_coords) - y_min)
            })

        text = " ".join(box["bbox_text"] for box in boxes)
        return Transcription(text, boxes, raw_output=parsed_answer)