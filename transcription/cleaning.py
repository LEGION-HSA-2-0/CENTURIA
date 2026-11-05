"""
Model-specific cleaning of raw OCR output.

Some models answer with LaTeX, a JSON wrapper or image descriptions instead of plain text. 
One function per model turns that into a plain transcription.
Models without an entry in CLEANERS are returned unchanged.

"""

import json
import logging
import re

from pylatexenc.latex2text import LatexNodes2Text

logging.getLogger("pylatexenc").setLevel(logging.ERROR)
_LATEX = LatexNodes2Text()

MARKDOWN_IMAGE = re.compile(r"!\[.*?\](?:\(.*?\))?")   # ![alt] and ![alt](url)
WEB_LINK = re.compile(r"https?://\S+")

IMAGE_DESCRIPTION = re.compile(
    r"\[(Sketch|Image|Diagram|Picture|Photo|Signature|Redacted).*?\]", re.IGNORECASE
)
COMMENTARY = re.compile(
    r"Note:|^\s*(?:The|This) (?:image|document|text|handwritten|drawing)\b", re.MULTILINE
)

def clean_olmocr(text: str) -> str:
    text = extract_natural_text(text).replace("\\n", "\n") # Later fix: escaped line breaks: "\\n" -> "\n"
    text = latex_to_text(text)
    return IMAGE_DESCRIPTION.sub("", text)

def clean_olmocr2(text: str) -> str:
    text = latex_to_text(text)
    text = WEB_LINK.sub("", text)
    return MARKDOWN_IMAGE.sub("", text)

def clean_lightonocr(text: str) -> str:
    text = latex_to_text(simplify_latex(text)) # simplify_latex added for fewshot 5
    text = COMMENTARY.split(text, maxsplit=1)[0] # the model sometimes appends a comment
    #text = MARKDOWN_IMAGE.sub("", text)
    #return truncate_repetitions(text) # test quality of repetition output
    return MARKDOWN_IMAGE.sub("", text)

CLEANERS = {
    "olmocr": clean_olmocr,
    "olmocr2": clean_olmocr2,
    "lightonocr": clean_lightonocr
}

def clean_output(model_name: str, text: str) -> str:
    cleaner = CLEANERS.get(model_name)
    return cleaner(text) if cleaner else text

def latex_to_text(text: str) -> str:
    """
    LaTeX -> plain text. Raises Warning if pylatexenc cannot convert the text. 
    Workaroungs for fewshot LightOnOCR outputs.
    """
    try:
        return _LATEX.latex_to_text(text)
    except Exception as error:
        print(f"Warning: pylatexenc failed ({type(error).__name__}), stripping LaTeX by hand: {text[:80]!r}")        
        text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\$([^$\n]{0,200})\$", r"\1", text)
        return re.sub(r"[$\\]", "", text)
    
def simplify_latex(text: str) -> str:
    """Rewrite LaTeX that pylatexenc handles badly (LightOnOCR output)."""
    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", text)   # \frac{a}{b} -> a/b
    text = re.sub(r"\\frac\{([^{}]*)\}\{\s*\}", r"\1", text)           # \frac{a}{}  -> a
    text = re.sub(r"\\frac\b", "", text)
    text = re.sub(r"\$\$\s*\\begin\{[^}]+\}\s*\\end\{[^}]+\}\s*\$\$", "", text)  # empty environments
    text = re.sub(r"\\begin\{[^}]+\}\s*\\end\{[^}]+\}", "", text)
    return re.sub(r"\$\\text\{([^}]*)\}(?![^\n]*\$)", r"\1", text)

def truncate_repetitions(text: str, max_repeats: int = 3) -> str:
    """Cut the text once a non-empty line occurs more than max_repeats times (repetition loops)."""
    seen, kept = {}, []
    for line in text.split("\n"):
        key = line.strip()
        if key:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > max_repeats:
                break
        kept.append(line)
    return "\n".join(kept)

def extract_natural_text(text: str) -> str:
    """olmOCR v1 answers with JSON {"natural_text": ...}."""
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "natural_text" in data:
            return data["natural_text"] or ""
    except json.JSONDecodeError:
        pass
    match = re.search(r'"natural_text"\s*:\s*"(.*)', text, re.DOTALL)
    if not match:
        return text
    text = re.sub(r'"\s*,\s*"[a-zA-Z_]+"\s*:.*', "", match.group(1), flags=re.DOTALL)
    return re.sub(r'"\s*\}?\s*$', "", text)
