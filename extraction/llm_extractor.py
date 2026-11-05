"""
Field extraction with a chat LLM (default Qwen3-8B).

The same prompt and examples are used for ground-truth text and OCR text:
system prompt: data/prompts/extraction_prompt.txt
few-shot examples from correct transcriptions: data/fewshot/extraction_examples.json

Few-shot file format, a JSON list of:
    {"text": "<transcription>", "fields": {...}}
"""

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extraction.schema import empty_result, fix_llm_result, parse_json

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
MAX_NEW_TOKENS = 400

class LlmExtractor:
    def __init__(self, prompt: Path, fewshot: Path, model_id: str = DEFAULT_MODEL_ID):
        self.system_prompt = Path(prompt).read_text(encoding="utf-8")
        self.examples = json.loads(Path(fewshot).read_text(encoding="utf-8"))

        print(f"Loading {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()

    def extract(self, text: str) -> dict:
        text = remove_markup(text)
        if not text.strip():
            return empty_result()

        chat_text = self.tokenizer.apply_chat_template(
            build_messages(self.system_prompt, self.examples, text),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]   # without the repeated prompt
        answer = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return fix_llm_result(parse_json(answer))

def build_messages(system_prompt: str, examples: list[dict], text: str) -> list[dict]:
    """System prompt, then each few-shot example as a user/assistant turn, then the text."""
    messages = [{"role": "system", "content": system_prompt}]

    for example in examples:
        messages.append({"role": "user", "content": f"prediction:\n{example['text']}"})
        messages.append({
            "role": "assistant",
            "content": json.dumps(example["fields"], ensure_ascii=False, indent=2),
        })

    messages.append({"role": "user", "content": f"prediction:\n{text}"})
    return messages

def remove_markup(text: str) -> str:
    """Remove markup some OCR models add (images, tables, HTML, code blocks, sketches)."""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text, flags=re.DOTALL) # markdown images
    text = re.sub(r"<table>.*?</table>", "", text, flags=re.DOTALL) # HTML tables
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL) # HTML comments
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL) # code blocks
    text = re.sub(r"<[^>]+>", "", text) # other HTML tags
    text = re.sub(r"\[?(?:A simple|Hand-drawn|Sketch|Diagram)[^\]\n]*\]?", "", text, flags=re.IGNORECASE) # sketch descriptions
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()