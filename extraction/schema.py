"""
The output fields and the fixes applied to every LLM answer.

All extractors return a dict with these fields:

    {
      "excavation":  {"provenance", "project", "fn", "se", "fl", "quadrant", "kiste"},
      "pottery_form": ..., "artifact_category": ..., "publication_type": ..., "surface_treatment": ...,
      "measurement": {"rd", "bd", "h", "r_a", "r_i", "r_m"},
      "others": [...]
    }

"""

import json
import re

EXCAVATION_KEYS = ("provenance", "project", "fn", "se", "fl", "quadrant", "kiste")
MEASUREMENT_KEYS = ("rd", "bd", "h", "r_a", "r_i", "r_m")
SINGLE_VALUE_KEYS = ("pottery_form", "artifact_category", "publication_type", "surface_treatment")

# Units at the end of a measurement, including frequent OCR corruptions of "cm"
UNIT = re.compile(r"\s*(cm|mm|μm|µm|am|an|au|Mcm|M_cm|locum|cu)\s*$", re.IGNORECASE)

def empty_result() -> dict:
    return {
        "excavation": dict.fromkeys(EXCAVATION_KEYS),
        "pottery_form": None,
        "artifact_category": None,
        "publication_type": None,
        "surface_treatment": None,
        "measurement": dict.fromkeys(MEASUREMENT_KEYS),
        "others": [],
    }

def parse_json(response: str) -> dict:
    """Find the JSON object in an LLM answer. Returns {} if there is none."""
    
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"```(?:json)?", "", response)

    match = re.search(r"\{.*\}", response, re.DOTALL)

    if not match:
        return {}

    text = match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # The answer was cut off: close open lists and objects and try again.
        text += "]" * max(0, text.count("[") - text.count("]"))
        text += "}" * max(0, text.count("{") - text.count("}"))
        try:
            return json.loads(text)
        except Exception:
            return {}

def fix_llm_result(result: dict) -> dict:
    """Repair the LLM's typical mistakes so the result has the expected shape."""
    if not isinstance(result.get("excavation"), dict):
        result["excavation"] = {}

    if not isinstance(result.get("measurement"), dict):
        result["measurement"] = {}

    # The model sometimes writes the name of the list ("others") as a value
    for key in SINGLE_VALUE_KEYS:
        if result.get(key) == "others":
            result[key] = None

    # Measurements: number only, without unit
    for key in MEASUREMENT_KEYS:
        result["measurement"][key] = strip_unit(result["measurement"].get(key))

    if not isinstance(result.get("others"), list):
        result["others"] = []
    return result

def strip_unit(value):
    """ "14 cm" -> "14", also for each element of a list like ["3,5 cm", "22%"]."""
    if isinstance(value, list):
        return [strip_unit(v) for v in value]
    
    if isinstance(value, str):
        return UNIT.sub("", value).strip() or None
    
    return value
