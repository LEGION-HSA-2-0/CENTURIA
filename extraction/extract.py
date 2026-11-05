"""
Stage 2: extract fields from every text of a transcription file (or the ground truth).

python main.py extract regex results/transcriptions/olmocr2.json
python main.py extract llm   results/transcriptions/olmocr2.json
python main.py extract regex ground_truth          # oracle: fields from the correct test text

Writes results/fields/<input>__<extractor>.json:
    {
      "stage": "fields",
      "settings": {"extractor": ..., "input": ..., "prompt": ..., "fewshot": ...,
                   "dataset_dir": ..., "subset": ..., "split": ...},
      "predictions": {"<image id>": {"text": ..., "fields": {...}}}
    }
"""

import json
from pathlib import Path

from dataset import DATASET_DIR, SUBSET, load_records

GROUND_TRUTH = "ground_truth"

def extract(
    extractor: str,
    source: str,
    prompt: Path = Path("data/prompts/extraction_prompt.txt"),
    fewshot: Path = Path("data/fewshot/extraction_examples.json"),
    output_dir: Path = Path("results/fields"),
) -> Path:
    """Extract fields from all texts and return the path of the written file."""
    texts, dataset_settings = load_texts(source)
    field_extractor = create_extractor(extractor, prompt, fewshot)

    predictions = {}
    for i, (image_id, text) in enumerate(texts.items(), 1):
        print(f"[{i}/{len(texts)}] {image_id}")
        predictions[image_id] = {"text": text, "fields": field_extractor.extract(text)}

    uses_prompt = extractor == "llm"
    settings = {
        "extractor": extractor,
        "input": str(source),
        "prompt": str(prompt) if uses_prompt else None,
        "fewshot": str(fewshot) if uses_prompt else None,
        **dataset_settings,
    }
    source_name = GROUND_TRUTH if source == GROUND_TRUTH else Path(source).stem
    name = f"{source_name}__{extractor}" + (f"__{Path(fewshot).stem}" if uses_prompt else "")
    path = output_dir / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": "fields", "settings": settings,
                                "predictions": predictions}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"Saved fields to {path}")
    return path

def create_extractor(name: str, prompt: Path, fewshot: Path):
    # Imported here: the LLM needs torch and a GPU, the regex extractor does not.
    if name == "regex":
        from extraction.regex_extractor import RegexExtractor
        return RegexExtractor()
    if name == "llm":
        from extraction.llm_extractor import LlmExtractor
        return LlmExtractor(prompt, fewshot)
    raise ValueError(f"Unknown extractor {name!r}, use 'regex' or 'llm'")

def load_texts(source: str) -> tuple[dict[str, str], dict]:
    """Image id -> text, and the dataset settings (dataset_dir, subset, split).

    source is "ground_truth" (the test transcriptions) or a transcription file
    {"stage": "transcription", "settings": {...}, "predictions": {id: {"text": ...}}}.
    """
    if source == GROUND_TRUTH:
        settings = {"dataset_dir": str(DATASET_DIR), "subset": SUBSET, "split": "test"}
        return load_records("test")["text"].to_dict(), settings

    data = json.loads(Path(source).read_text(encoding="utf-8"))
    if data.get("stage") != "transcription":
        raise ValueError(f"{source} is not a transcription file (missing \"stage\": \"transcription\")")
    settings = {key: data["settings"][key] for key in ("dataset_dir", "subset", "split")}
    return {image_id: p["text"] for image_id, p in data["predictions"].items()}, settings
