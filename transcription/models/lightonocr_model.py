import torch
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

from .model import OCRModel, Transcription

class LightonOcrModel(OCRModel):
    """
    LightOnOCR-2 (1B): native-resolution ViT encoder (Mistral-Small-3.1) with a Qwen3 decoder. 
    Supports custom prompts and few-shot examples.

    Difference to the paper's baseline run:
    It used a longer instruction ("You are a text extractor. Read the handwritten text in this image.
    Output the text as a single, continuous string. Separate all words and lines with a single space
    only. Completely ignore any sketches, drawings and holes in the paper. Do not output any comments,
    notes or image tags whatsoever") and full-size images. It gave SpACER-M 18.25 %, BoW-F1 50.98 % and 
    509 characters of raw output on average.
    """

    DEFAULT_MODEL_ID = "lightonai/LightOnOCR-2-1B"
    supports_prompting = True

    DEFAULT_PROMPT = "Transcribe the document."
    MAX_IMAGE_SIZE = 1288
    MAX_NEW_TOKENS = 512
    REPETITION_PENALTY = 1.2

    def __init__(self, model_id=None, prompt=None, examples=()):
        super().__init__(model_id, prompt, examples)

        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"        
        self.dtype = torch.float32 if self.device == "mps" else torch.bfloat16

        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=self.dtype
            ).to(self.device).eval()

        self.processor = LightOnOcrProcessor.from_pretrained(self.model_id)

        self.example_images = [self.load_image(ex.image_path) for ex in self.examples]

    def transcribe(self, image_path: str) -> Transcription:
     
        messages, images = [], []
        for example, example_image in zip(self.examples, self.example_images):
            messages.append(self.user_message())
            messages.append({"role": "assistant", "content": example.transcription})
            images.append(example_image)

        messages.append(self.user_message())
        images.append(self.load_image(image_path))

        chat_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=chat_text,
            images=images[0] if len(images) == 1 else images,
            return_tensors="pt"
        )

        inputs = {k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.MAX_NEW_TOKENS, 
                do_sample=False, # greedy search
                repetition_penalty=self.REPETITION_PENALTY,
                use_cache=True
            )

        #torch.cuda.empty_cache()

        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        return Transcription(output_text)
    
    
    def load_image(self, image_path):
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        image.thumbnail((self.MAX_IMAGE_SIZE, self.MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        return image

    def user_message(self) -> dict:
        prompt = self.prompt or self.DEFAULT_PROMPT
        return {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    
