"""
Loading the CENTURIA dataset: data/CENTURIA_DATASET/{annotations.json, images/}.

annotations.json has one entry per image with its id ("00042.jpg"), subset
(CENTURIA, COHORS, ...), split (train/test), source campaign, transcription,
text boxes and fields. All images share one folder.

Used by transcription and training (which images), extraction (ground-truth
text) and evaluation (what is correct).
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATASET_DIR = Path("data/CENTURIA_DATASET")
SUBSET = "CENTURIA"

def load_records(split: str, subset: str = SUBSET, dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """
    One row per image of the given split ("train" or "test") and subset.

    Index: image id ("00042.jpg"), the key in all result files.
    Columns: image_path, campaign, text, boxes, fields.
    """
    entries = [entry for entry in _read_annotations(str(dataset_dir))
               if entry["split"] == split and entry["subset"] == subset]
    if not entries:
        raise ValueError(f"No {subset} images with split {split!r} in {dataset_dir}")

    images_dir = Path(dataset_dir) / "images"
    return pd.DataFrame(
        {
            "image_path": [str(images_dir / entry["id"]) for entry in entries],
            "campaign": [campaign_name(entry["source"]) for entry in entries],
            "text": [entry["text"] for entry in entries],
            "boxes": [entry.get("text_boxes", []) for entry in entries],
            "fields": [entry.get("fields", {}) for entry in entries],
        },
        index=pd.Index([entry["id"] for entry in entries], name="id"),
    )

def campaign_name(source: dict) -> str:
    """Campaign, plus the sector if there is one: "Excavation Gstettenbreite 1976 Sektor 1-2"."""
    sector = source.get("sector")
    return f"{source['campaign']} Sektor {sector}" if sector else source["campaign"]

def check_images_exist(records: pd.DataFrame) -> None:
    """Stop before a model is loaded if images are missing."""
    missing = [image_id for image_id, path in records["image_path"].items() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images are missing, e.g. {missing[:3]}")

def load_fewshot_examples(path: Path, dataset_dir: Path = DATASET_DIR) -> list:
    """JSON list of {"id": "00001.jpg", "transcription": ...} -> FewShotExample objects."""
    # Imported here, so evaluation can use this module without the model libraries.
    from transcription.models.model import FewShotExample

    images_dir = Path(dataset_dir) / "images"
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [FewShotExample(str(images_dir / e["id"]), e["transcription"]) for e in raw]

def check_no_test_leakage(examples: list, test_ids) -> None:
    """Stop if a few-shot example is also a test image."""
    test_ids = set(test_ids)
    leaked = [e.image_path for e in examples if Path(e.image_path).name in test_ids]
    if leaked:
        raise ValueError(f"{len(leaked)} few-shot examples are test images, e.g. {leaked[:3]}")

@lru_cache(maxsize=None)
def _read_annotations(dataset_dir: str) -> list[dict]:
    """annotations.json, read once per dataset folder."""
    return json.loads((Path(dataset_dir) / "annotations.json").read_text(encoding="utf-8"))
