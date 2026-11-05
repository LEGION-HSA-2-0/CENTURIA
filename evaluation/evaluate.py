"""
Compare a results file with the ground truth and write a report.

python main.py evaluate results/transcriptions/olmocr2.json
python main.py evaluate results/fields/olmocr2__llm__extraction_examples.json

The stage is read from the file ("stage": "transcription" or "fields").
"""

import json
from collections import Counter
from itertools import chain
from pathlib import Path
from statistics import fmean

from cotescore import spacer, spacer_micro

import evaluation_metrices as metrics
from dataset import DATASET_DIR, SUBSET, load_records
from evaluation.field_metrics import (
    ACCURACY_SCORES,
    COVERAGE_SCORES,
    accuracy_record,
    coverage_record,
    summarize
)

def evaluate(results: Path, output_dir: Path = Path("results/reports")) -> Path:
    """Compare a results file with the ground truth of the split it was made for."""
    data = json.loads(Path(results).read_text(encoding="utf-8"))
    stage = data.get("stage")
    ground_truth = load_ground_truth(data["settings"])

    if stage == "transcription":
        report = evaluate_transcription(data, ground_truth)
    elif stage == "fields":
        report = evaluate_fields(data, ground_truth)
    else:
        raise ValueError(f"{results}: unknown stage {stage!r} (expected 'transcription' or 'fields')")

    report = {"source": str(results), "settings": data["settings"], **report}
    path = output_dir / stage / Path(results).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=4, ensure_ascii=False, default=_to_json),
                    encoding="utf-8")
    print(f"Saved report to {path}")
    return path

def load_ground_truth(settings: dict):
    """Ground truth of the split, subset and dataset recorded in the results file."""
    return load_records(settings.get("split", "test"), settings.get("subset", SUBSET),
                        Path(settings.get("dataset_dir", DATASET_DIR)))

# Transcription: CER, WER, BoW-F1, SpACER, field coverage
def evaluate_transcription(data: dict, ground_truth) -> dict:
    """Text metrics per image and for the whole test set.

    The metric functions and how they are called are the same as in the
    former evaluate_ocr.py, so the numbers are comparable.
    """
    settings = data["settings"]
    predictions = data["predictions"]
    has_boxes = any(p["boxes"] for p in predictions.values())   # Florence-2 and TrOCR

    per_image = {}
    for image_id, prediction in predictions.items():
        gt = ground_truth.loc[image_id]
        per_image[image_id] = score_text(gt["text"], prediction["text"])
        if has_boxes:
            per_image[image_id] |= score_boxes(gt["boxes"], prediction["boxes"])

    results = list(per_image.values())
    summary = {
        "avg_cer": fmean(r["cer"] for r in results),
        "avg_wer": fmean(r["wer"] for r in results),
        "avg_bow_f1": fmean(r["bow_f1"] for r in results),
        "dataset_macro_spacer": spacer(sum((r["_gt_chars"] for r in results), Counter()),
                                       sum((r["_pred_chars"] for r in results), Counter())),
        "avg_image_macro_spacer": fmean(r["image_macro_spacer"] for r in results),
        "dataset_micro_spacer": None,      # only for models with text boxes
        "avg_image_micro_spacer": None,
        **cost_summary(predictions.values()),
    }
    if has_boxes:
        summary["dataset_micro_spacer"] = spacer_micro(
            list(chain.from_iterable(r["_gt_counters"] for r in results)),
            list(chain.from_iterable(r["_pred_counters"] for r in results)),
        )
        summary["avg_image_micro_spacer"] = fmean(r["image_micro_spacer"] for r in results)

    image_results = {
        image_id: {"ground_truth": ground_truth.loc[image_id, "text"],
                   "prediction": predictions[image_id]["text"],
                   **{k: v for k, v in r.items() if not k.startswith("_")}}
        for image_id, r in per_image.items()
    }
    if has_boxes:
        for image_id, result in image_results.items():
            result["prediction_list"] = predictions[image_id]["boxes"]

    # "model", "metrices" and "image_results" as before, so the plots keep working.
    report = {"model": settings["model"], "metrices": summary, "image_results": image_results}
    texts = {image_id: p["text"] for image_id, p in predictions.items()}
    report["field_coverage"] = field_coverage(ground_truth["fields"].to_dict(), texts)
    return report


def score_text(ground_truth: str, prediction: str) -> dict:
    gt = metrics.clean_text_for_evaluation(ground_truth)
    pred = metrics.clean_text_for_evaluation(prediction)
    return {
        "cer": metrics.character_error_rate([pred], [gt]),
        "wer": metrics.word_error_rate([pred], [gt]),
        "bow_f1": metrics.bag_of_words_f1([pred], [gt]),
        "image_macro_spacer": metrics.calculate_macro_spacer(gt, pred),
        "_gt_chars": Counter(gt),       # for the pooled SpACER over all images
        "_pred_chars": Counter(pred),
    }

def score_boxes(gt_boxes: list, predicted_boxes: list) -> dict:
    """Micro SpACER: compares text box by text box. Counters are kept for the dataset value."""
    micro, gt_counters, pred_counters = metrics.calculate_micro_spacer(gt_boxes, predicted_boxes)
    return {"image_micro_spacer": micro, "_gt_counters": gt_counters, "_pred_counters": pred_counters}

def cost_summary(predictions) -> dict:
    durations = [p["duration"] for p in predictions]
    memory = [p["peak_memory_gb"] for p in predictions]
    return {
        "total_duration": sum(durations),
        "avg_duration_per_image": fmean(durations),
        "avg_peak_memory_gb": fmean(memory),
        "max_peak_memory_gb": max(memory),
    }

def field_coverage(gt_fields: dict, texts: dict) -> dict:
    """Is each ground-truth field value in the OCR text? (EMR exact, ANLS fuzzy)."""
    records = [coverage_record(fields, texts[image_id])
               for image_id, fields in gt_fields.items() if image_id in texts]
    report = summarize(records, COVERAGE_SCORES)
    missing = [image_id for image_id in gt_fields if image_id not in texts]
    if missing:
        report["missing_in_predictions"] = missing
    return report

# Fields: accuracy per field
def evaluate_fields(data: dict, ground_truth) -> dict:
    """Is each extracted field equal to the ground truth (after normalization)?"""
    gt_fields = ground_truth["fields"].to_dict()
    predictions = data["predictions"]

    records, errors, missing = [], {}, []
    for image_id, fields in gt_fields.items():
        if image_id not in predictions:
            missing.append(image_id)
            continue
        record = accuracy_record(fields, predictions[image_id]["fields"])
        records.append(record)
        for field, result in record.items():
            if not result["correct"]:
                errors.setdefault(field, []).append(
                    {"id": image_id, "gt": result["gt"], "pred": result["pred"]})

    report = summarize(records, ACCURACY_SCORES)
    if missing:
        report["missing"] = missing
        print(f"Warning: {len(missing)} ground-truth images are not in the predictions")
    report["errors"] = dict(sorted(errors.items(), key=lambda item: -len(item[1])))
    return report

def _to_json(obj):
    """numpy numbers/arrays and torch tensors -> plain Python values."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")