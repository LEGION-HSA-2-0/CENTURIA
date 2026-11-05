import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from .model import OCRModel, Transcription

class OlmocrModel(OCRModel):
    """
    olmOCR v1 (Qwen2-VL 7B): fixed prompt only, no few-shot examples.
    """

    DEFAULT_MODEL_ID = "allenai/olmocr-7b-0225-preview"
    DEFAULT_PROMPT = "Transcribe the document."
    MAX_NEW_TOKENS = 1024
    
    def __init__(self, model_id=None, prompt=None, examples=()):
        super().__init__(model_id, prompt, examples)

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)


    def transcribe(self, image_path: str) -> Transcription:
        # Use the standard Qwen2-VL message format, process_vision_info can process img path directly
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}, 
                    {"type": "text", "text": self.DEFAULT_PROMPT}
                ],
            }
        ]
        
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        images = process_vision_info(messages)[0]

        inputs = self.processor(
            text=[chat_text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False, # greedy search
                use_cache=True
            )

        new_ids = output_ids[0, inputs["input_ids"].shape[1]:] # without repeated prompt
        
        return Transcription(self.processor.decode(new_ids, skip_special_tokens=True).strip())