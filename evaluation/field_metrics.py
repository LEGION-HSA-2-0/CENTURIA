"""
Field metrics.
"""

import re
from collections import defaultdict

import Levenshtein

SINGLE_VALUE_FIELDS = ["pottery_form", "artifact_category", "surface_treatment", "publication_type"]
ANLS_THRESHOLD = 0.5   # fuzzy matches below this similarity count as 0

# Accuracy: extracted fields vs. ground-truth fields
def accuracy_record(gt_fields: dict, pred_fields: dict) -> dict:
    """Per field: {"correct": 0|1, "gt": ..., "pred": ...}."""

    results = {}
    for name, gt_value, pred_value in _field_pairs(gt_fields, pred_fields):
        results[name] = {
            "correct": 1 if normalize(gt_value) == normalize(pred_value) else 0,
            "gt": gt_value,
            "pred": pred_value,
        }
    return results

def normalize(value) -> str:
    """Make two notations of the same value equal before comparing them.

      parens          '(Adler-Wölfl …)'          -> 'adler-wölfl …'
      | vs /          'Ki 1|76'                  -> 'ki 1/76'
      bzw. separator  'Taf. 98/KE … bzw. Taf. …' -> '… | taf. …'
      % notation      '% 10' / '10 %' / '10%'    -> '10%'
      space vs / (KE) 'Taf. 102 KE 2937'         -> 'taf. 102/ke 2937'
      SE = sign       '19 = 21'                  -> '19-21'
      quadrant dash   'Q - 40 - 12'              -> 'q 40-12'
      decimal sep.    '1,6'                      -> '1.6'
      + spacing       'Üz a + i'                 -> 'üz a+i'
      trailing dot    'Gassner 3/14.'            -> 'gassner 3/14'
      lower case, single spaces
    """

    if value is None:
        return ""
    
    if isinstance(value, list):
        return " | ".join(normalize(v) for v in value)

    text = str(value).lower().strip()
    text = re.sub(r"[()]", "", text)                               # parentheses
    text = re.sub(r"(?<! )\|(?! )", "/", text)                     # 'ki 1|76' -> 'ki 1/76'
    text = re.sub(r"\bbzw\.?\s*", " | ", text)                     # 'bzw.' separates references
    text = re.sub(r"(?:^\s*\|\s*|\s*\|\s*$)", "", text)            # separator at start/end
    text = re.sub(r"(?:\s*\|\s*)+", " | ", text)                   # repeated separators
    text = re.sub(r"%\s*(\d+)", r"\1%", text)                      # '% 10' -> '10%'
    text = re.sub(r"(\d+)\s*%", r"\1%", text)                      # '10 %' -> '10%'
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"(?<=\d)\s*=\s*(?=\d)", "-", text)              # '19 = 21' -> '19-21'
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\bq-(\d)", r"q \1", text)                      # 'q-40-12' -> 'q 40-12'
    text = re.sub(r"(\d+)\s+\b(ke|str)\b", r"\1/\2", text)         # 'taf. 102 ke' -> 'taf. 102/ke'
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)                    # '1,6' -> '1.6'
    text = re.sub(r"\s*\+\s*", "+", text)                          # 'a + i' -> 'a+i'
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".")
    return text.strip()


# Coverage: ground-truth fields vs. OCR text 

# Measurement labels (after clean_text), so that "14" only counts next to "RD"
# and not when it is part of a crate number, SE code or publication.
MEASUREMENT_LABEL = {
    "rd": r"\brd?\b",          # RD or R
    "bd": r"\bbd?\b",          # BD or B
    "h": r"\bh\b",
    "r_a": r"\br_?a\b|\br\b",  # R_A or R
    "r_i": r"\br_?i\b",
    "r_m": r"\br_?m\b",
}

# Label as written in the text, for the fuzzy (ANLS) comparison "rd = 14".
MEASUREMENT_LABEL_TEXT = {"rd": "rd", 
                          "bd": "bd", 
                          "h": "h", 
                          "r_a": "r", 
                          "r_i": "r i", 
                          "r_m": "r m"}

def coverage_record(gt_fields: dict, ocr_text: str) -> dict:
    """Per field: {"emr": 0..1, "anls": 0..1, "gt": ...}."""
    results = {}
    for name, gt_value, _ in _field_pairs(gt_fields, {}):
        if name.startswith("measurement."):
            key = name.split(".", 1)[1]
            emr = emr_measurement(key, gt_value, ocr_text)
            anls = anls_measurement(key, gt_value, ocr_text)
        elif isinstance(gt_value, list):   # several values: the best one counts
            emr = max(emr_text(v, ocr_text) for v in gt_value)
            anls = max(anls_text(v, ocr_text) for v in gt_value)
        else:
            emr = emr_text(gt_value, ocr_text)
            anls = anls_text(gt_value, ocr_text)
        results[name] = {"emr": emr, "anls": anls, "gt": gt_value}
    return results

def clean_text(text) -> str:
    """Lower case and uniform spacing, so values and OCR text can be compared."""

    if text is None:
        return ""
    
    if isinstance(text, list):
        text = " ".join(str(t) for t in text if t is not None)

    text = str(text).lower()
    text = text.replace("\\/", "/")
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = text.replace("=", " = ")
    text = text.replace(":", " : ")
    text = re.sub(r"([a-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    text = re.sub(r"(\d)\s*cm\b", r"\1 cm", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def emr_text(gt_value, ocr_text) -> float:
    """1 if the value appears exactly (as substring) in the text."""
    if ocr_text is None:
        return 0.0
    return 1.0 if clean_text(gt_value) in clean_text(ocr_text) else 0.0

def anls_text(gt_value, ocr_text) -> float:
    """Best similarity between the value and any window of the text with as many words."""
    if ocr_text is None:
        return 0.0
    return _best_window_similarity(clean_text(gt_value), clean_text(ocr_text))

def emr_measurement(key: str, gt_value, ocr_text) -> float:
    """1 if 'label [=:] value' appears, e.g. 'rd = 14'. For ["8", "% 10"] also the percentage."""
    if ocr_text is None:
        return 0.0
    
    text = clean_text(ocr_text)
    label = MEASUREMENT_LABEL.get(key, rf"\b{re.escape(key)}\b")

    number = gt_value[0] if isinstance(gt_value, list) else gt_value
    pattern = rf"(?:{label})\s*[=:]?\s*{re.escape(clean_text(str(number)))}\b"
    number_found = bool(re.search(pattern, text, re.IGNORECASE))

    percent_found = True
    if isinstance(gt_value, list) and len(gt_value) > 1:
        digits = re.search(r"\d+", str(gt_value[1]))
        if digits:
            percent = digits.group()
            percent_found = bool(re.search(rf"\b{percent}\s*%|%\s*{percent}\b", text))

    return 1.0 if number_found and percent_found else 0.0

def anls_measurement(key: str, gt_value, ocr_text) -> float:
    """Fuzzy match of label = value."""
    if ocr_text is None:
        return 0.0
    number = gt_value[0] if isinstance(gt_value, list) else gt_value
    target = f"{MEASUREMENT_LABEL_TEXT.get(key, key)} = {clean_text(str(number))}"
    return _best_window_similarity(target, clean_text(ocr_text))

def _best_window_similarity(target: str, text: str) -> float:
    """Slide a window of len(target words) over the text: 1 - normalized edit distance."""
    target_words = target.split()
    text_words = [w.strip(",;") for w in text.split()]

    if not target_words or not text_words:
        return 0.0
    
    best = 0.0

    for i in range(max(1, len(text_words) - len(target_words) + 1)):
        window = " ".join(text_words[i:i + len(target_words)])
        longest = max(len(target), len(window))
        if longest > 0:
            best = max(best, 1.0 - Levenshtein.distance(target, window) / longest)
    return best if best >= ANLS_THRESHOLD else 0.0

def _field_pairs(gt_fields: dict, pred_fields: dict):
    """(name, gt value, predicted value) for every field with a ground-truth value."""
    for name in SINGLE_VALUE_FIELDS:
        if gt_fields.get(name) is not None:
            yield name, gt_fields[name], pred_fields.get(name)

    for group in ("measurement", "excavation"):
        for key, gt_value in gt_fields.get(group, {}).items():
            if gt_value is not None:
                yield f"{group}.{key}", gt_value, pred_fields.get(group, {}).get(key)

ACCURACY_SCORES = {"correct": ("accuracy (%)", "overall_accuracy (%)")}
COVERAGE_SCORES = {"emr": ("EMR (%)", "Overall EMR (%)"), "anls": ("ANLS (%)", "Overall ANLS (%)")}

def summarize(records: list[dict], scores: dict) -> dict:
    """Per field and overall in percent."""
    totals = defaultdict(lambda: defaultdict(float))
    for record in records:
        for field, result in record.items():
            for key in scores:
                totals[field][key] += result[key]
            totals[field]["n"] += 1

    per_field = {
        field: {**{name: round(t[key] / t["n"] * 100, 2) for key, (name, _) in scores.items()},
                "n": int(t["n"])}
        for field, t in totals.items()
    }

    documents = [record for record in records if record]
    overall = {}
    if documents:
        for key, (_, name) in scores.items():
            per_document = [sum(r[key] for r in doc.values()) / len(doc) for doc in documents]
            overall[name] = round(sum(per_document) / len(documents) * 100, 2)
        overall["num_documents"] = len(documents)
    
    return {"overall": overall, "per_field": per_field}