import torch
from PIL import Image

from transformers import (
    AutoProcessor, 
    AutoConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Config      
)

from .model import OCRModel, Transcription

class Olmocr2Model(OCRModel):
    """
    olmOCR v2 (Qwen2.5-VL-7B-Instruct) supports custom prompts, few-shot examples and both combined.
    """

    DEFAULT_MODEL_ID = "allenai/olmOCR-2-7B-1025"
    supports_prompting = True
    
    DEFAULT_PROMPT = "Transcribe the document."
    MAX_IMAGE_SIZE = 1288
    MAX_NEW_TOKENS = 4096

    def __init__(self, model_id=None, prompt=None, examples=()):
        super().__init__(model_id, prompt, examples)

        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        
        if isinstance(getattr(config, "text_config", None), dict):
            config.text_config = Qwen2Config(**config.text_config)
            
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        self.example_images = [self.load_image(ex.image_path) for ex in self.examples]

    def transcribe(self, image_path: str) -> Transcription:

        messages, images = [], []
        for example, example_image in zip(self.examples, self.example_images):
            messages.append(self.user_message(example_image))
            messages.append({"role": "assistant", "content": example.transcription})
            images.append(example_image)

        image = self.load_image(image_path)
        messages.append(self.user_message(image))
        images.append(image)

        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[chat_text],
            images=images,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False, # greedy search
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True
            )

        new_ids = output_ids[0, inputs["input_ids"].shape[1]:] # without repeated prompt
        
        return Transcription(self.processor.decode(new_ids, skip_special_tokens=True).strip())

    def load_image(self, image_path):
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        image.thumbnail((self.MAX_IMAGE_SIZE, self.MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        return image

    def user_message(self, image) -> dict:
        prompt = self.prompt or self.DEFAULT_PROMPT
        return {"role": "user", 
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    
