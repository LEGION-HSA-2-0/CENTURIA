"""Stage 1: transcribe every image of one split (default: test) with one OCR model.

Writes results/transcriptions/<run name>.json with the text of every image.
No metrics are computed here. That is done by `python main.py evaluate <file>`.

Output format:
    {
      "stage": "transcription",
      "settings": {"model": ..., "prompt": ..., "fewshot": ..., "finetuned": ...,
                   "dataset_dir": ..., "subset": ..., "split": ...},
      "predictions": {
        "<image id>": {"text": cleaned text, "raw_text": model output,
                             "boxes": [...], "raw_output": ..., "duration": s,
                             "peak_memory_gb": GB}
      }
    }
"""

import json
import time
from pathlib import Path

import torch

from dataset import (
    DATASET_DIR, SUBSET, check_images_exist, check_no_test_leakage, load_fewshot_examples, load_records,
)
from transcription.cleaning import clean_output
from transcription.models import ModelFactory

def transcribe(
    model: str,
    prompt: Path | None = None,
    fewshot: Path | None = None,
    finetuned: Path | None = None,
    split: str = "test",
    subset: str = SUBSET,
    dataset_dir: Path = DATASET_DIR,
    output_dir: Path = Path("results/transcriptions"),
) -> Path:
    """Transcribe all images of one split and return the path of the written file."""
    prompt_text = prompt.read_text(encoding="utf-8") if prompt else None
    examples = load_fewshot_examples(fewshot, dataset_dir) if fewshot else []
    test_set = load_records(split, subset, dataset_dir)
    check_images_exist(test_set)
    check_no_test_leakage(examples, test_set.index)

    # Create the model (base model, or a fine-tuned checkpoint)
    ocr = ModelFactory.get_model(
        model, finetuned_path=str(finetuned) if finetuned else None,
        prompt=prompt_text, examples=examples,
    )
    if finetuned:
        warn_if_settings_differ_from_training(ocr, finetuned)

    # Transcribe every image
    predictions = {}
    for i, (image_id, row) in enumerate(test_set.iterrows(), 1):
        print(f"[{i}/{len(test_set)}] {image_id}")
        predictions[image_id] = transcribe_image(ocr, model, row["image_path"])

    # Save
    settings = {
        "model": model,
        "prompt": str(prompt) if prompt else None,
        "fewshot": str(fewshot) if fewshot else None,
        "finetuned": str(finetuned) if finetuned else None,
        "dataset_dir": str(dataset_dir),
        "subset": subset,
        "split": split
    }
    path = output_dir / f"{run_name(model, prompt, fewshot, finetuned)}.json"
    save({"stage": "transcription", "settings": settings, "predictions": predictions}, path)
    return path

def transcribe_image(ocr, model_name: str, image_path: str) -> dict:
    """Transcribe one image and measure time and GPU memory."""
    for gpu in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(gpu)

    start = time.perf_counter()
    transcription = ocr.transcribe(image_path)
    duration = time.perf_counter() - start

    peak_memory = sum(torch.cuda.max_memory_allocated(g) for g in range(torch.cuda.device_count()))

    return {
        "text": clean_output(model_name, transcription.text),   # used for evaluation
        "raw_text": transcription.text,                          # kept to see what cleaning did
        "boxes": transcription.boxes,                            # only Florence-2 and TrOCR
        "raw_output": transcription.raw_output,
        "duration": duration,
        "peak_memory_gb": peak_memory / 1024**3,
    }

def warn_if_settings_differ_from_training(ocr, finetuned: Path) -> None:
    """A fine-tuned model should see the same prompt and image size as in training."""
    info_file = Path(finetuned) / "training_info.json"
    if not info_file.is_file():
        return
    info = json.loads(info_file.read_text(encoding="utf-8"))

    prompt = ocr.prompt or getattr(ocr, "DEFAULT_PROMPT", None)
    if info.get("prompt") and prompt != info["prompt"]:
        print("Warning: the prompt differs from the prompt used for fine-tuning")

    image_size = getattr(ocr, "MAX_IMAGE_SIZE", None)
    if info.get("max_image_size") and image_size != info["max_image_size"]:
        print(f"Warning: transcribing with {image_size} px images, "
              f"but fine-tuned with {info['max_image_size']} px")

def run_name(model: str, prompt, fewshot, finetuned) -> str:
    """File name from the settings, so different runs never overwrite each other."""
    parts = [model]
    if prompt:
        parts.append(f"prompt-{Path(prompt).stem}")
    if fewshot:
        parts.append(f"fewshot-{Path(fewshot).stem}")
    if finetuned:
        parts.append(f"ft-{Path(finetuned).name}")
    return "__".join(parts)

def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_to_json),
                    encoding="utf-8")
    print(f"Saved transcriptions to {path}")

def _to_json(obj):
    """numpy numbers/arrays and torch tensors -> plain Python values."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")