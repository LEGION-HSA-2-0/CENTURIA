"""
Rule-based field extraction with regular expressions and controlled vocabularies.

The fields are found one after another. Every match is blanked out of the text, 
so later rules cannot pick it up again. What is left at the end goes into "others".
"""

import re

# Controlled vocabularies (longest first, so "Dreifußschale" wins over "Dreifuß") 
POTTERY_FORMS = sorted([
    "Flacher Teller", "Deckel/Schüssel", "Schüssel/Sieb", "Räucherschale", "Reibschüssel",
    "Faltenbecher", "Mortarium", "Vorratsgefäß", "Dreifuß", "Dreifußschale", "Firmalampe",
    "Bildlampe", "Nachttopf", "Käseform", "Knickwandschüssel", "Kragenschale",
    "Fläschchen", "Amphore", "Schüssel", "Schale", "Deckel", "Flasche",
    "Becher", "Krug", "Teller", "Topf", "Sieb", "Kanne", "Henkel",
], key=len, reverse=True)

ARTIFACT_CATEGORIES = sorted([
    "Feinware f/ox-red", "Feinware f/ox", "Feinware f/red", "f/ox Glasierte GK",
    "f/ox PGW", "f/red PGW", "f/red GG", "f/ox GK", "f/ox GG", "g/red GG", "GG f/red",
    "f/red GK", "g/red GK", "g/ox GK", "g/red Gk", "f/ox FW", "FW f/ox", "f/red PG",
    "Pann. Glanztonware", "Glanztonware", "Terra Sigillata", "Graphittonkeramik",
    "Germanische Keramik", "Horreum Keramik", "Horreum Ker.", "Eingeglättete Keramik",
    "Eingegl. Ker.", "Eingegl. Ker", "Glasierte Keramik", "Feinkeramik", "Feinware",
    "Oberfläche geglättet", "oxid. GK", "oxid. Gk", "red. GK", "red. Gk",
    "red GK", "GK red.", "GK red", "GK ox", "GK oxid.", "GK. ox", "GG red, grau",
    "GK Mischbrand",
], key=len, reverse=True)

PROVENANCES = ["Carn.", "Carn", "Car.", "CAR", "Carnuntum"]

AUTHORS = ["adler", "grüne", "gassner", "petznek", "horvat"] # lower case, for matching

SURFACE_PATTERN = re.compile(
    r"\b(?:(?:rot|weiß|bräun|rötl|weißl|bräunl)\w*\.?\s+)?(?:Üz|Uz|Üu)"
    r"(?:\s*(?:a\s*\+\s*i|a\s*\+\s*I|a|i))?(?:\s+Grießbewurf)?\b|"
    r"\bGrießbewurf\b|"
    r"\b(?:[A-Za-zÄÖÜäöüß]+\s+)?Glasur\b|"
    r"\bGlasspritzer\b|"
    r"\bRatterdekor\b",
    re.IGNORECASE,
)

# Reference number: n/n, n / n or n.n (e.g. Petznek 7.1)
NUMBER = r"(?:\d+(?:\s*/\s*\d+|\.\d+)*)"

PUBLICATION_PATTERNS = [
    # Author (+ year) (+ Taf./Abb./Sü/Typ) + number (+ Str./KE references)
    (r"\(?(?:Adler-Wölfl|Gattringer\s*-\s*Grünewald(?:e|f)?|Grünewald(?:e)?|Gassner|"
     r"Petznek(?:-Radauer)?|Horvat|Gattringer|Stuppner|Friesinger-Kerchler|Petzn\.)"
     r"(?:\s+-)?\s*(?:\d{4})?\s*(?:(?:Taf\.|Abb\.|Sü|Su|Typ)\s*)?"
     + NUMBER + r"?"
     r"(?:\s*(?:bzw\.\s*|/\s*)?\(?\s*(?:Str\.|str\.|KE)\s*[\d/.-]+\s*\)?)*\s*\)?"),
    # Plate reference followed by KE/Str. references, without author
    (r"\(?(?:bzw\.\s*)?(?:Taf\.|Abb\.|KE|Str\.|str\.)\s*[\d/.-]+"
     r"(?:\s*/\s*(?:KE|Str\.|str\.)\s*[\d/.-]+)+\s*\)?"),
    # Dragendorff types
    r"\b(?:immit\.\s*)?Drag\.?\s*\d+\b",
    # Single plate reference without author
    r"\(?(?:bzw\.\s*)?(?:Taf\.|Abb\.|KE|Str\.|str\.)\s*" + NUMBER + r"\)?",
]

class RegexExtractor:
    """Extract fields from one label text with fixed rules."""

    def extract(self, text: str) -> dict:
        return extract_fields(text)

def extract_fields(text: str) -> dict:
    text = text.strip()
    remaining = text
    result = {
        "excavation": {},
        "pottery_form": None,
        "artifact_category": None,
        "publication_type": None,
        "surface_treatment": None,
        "measurement": {},
        "others": [],
    }

    def blank_out(match):
        """Replace a match with spaces, so it is not found again."""

        nonlocal remaining
        start, end = match.start(), match.end()
        remaining = remaining[:start] + " " * (end - start) + remaining[end:]

    # Provenance: "Carn.", "CAR", ... 
    found_provenances = []
    provenance_regex = "|".join(
        r"\b" + re.escape(p) + ("" if p.endswith(".") else r"\b") for p in PROVENANCES
    )
    for m in re.finditer(provenance_regex, remaining, re.IGNORECASE):
        value = text[m.start():m.end()].strip()
        if value not in found_provenances:
            found_provenances.append(value)
        blank_out(m)
    if found_provenances:
        result["excavation"]["provenance"] = (
            found_provenances[0] if len(found_provenances) == 1 else found_provenances
        )

    # Project: "Survey GB 2017", "GS - 2012", "Limesgasse 2016", ...
    m = re.search(
        r"(?:-[^\S\n]*)?\b(?:Survey|EVN|GS|Limesgasse)"
        r"(?:[^\S\n]+[A-Za-zÄÖÜäöüß]+)*"
        r"(?:[^\S\n]*-[^\S\n]*\d{4}|[^\S\n]+\d{4})?\b",
        remaining, re.IGNORECASE,
    )
    if m:
        result["excavation"]["project"] = m.group().strip().strip("-").strip()
        blank_out(m)
    else:
        # No keyword: a year on the first line (not later, where years belong to publications)
        first_line = remaining.split("\n")[0]
        m_year = re.search(r"(?<!\d)\b((?:19|20)\d{2})\b(?!\d)", first_line)
        if m_year:
            result["excavation"]["project"] = m_year.group(1)
            m_year_full = re.search(r"(?<!\d)\b" + m_year.group(1) + r"\b(?!\d)", remaining)
            if m_year_full:
                blank_out(m_year_full)

    # FN (find number) and FL (area): "FN 220", "FL 1"
    for key in ["fn", "fl"]:
        m = re.search(r"\b" + key.upper() + r"\s*(\d+(?:/\d+)?)(?:\b|,|\s)",
                      remaining, re.IGNORECASE)
        if m:
            result["excavation"][key] = m.group(1)
            blank_out(m)

    # SE (stratigraphic unit), also compound codes: "SE 39 - 36 - 42"
    m = re.search(r"\bSE\s*(\d+(?:\s*-\s*\d+)*)\b", remaining, re.IGNORECASE)
    if m:
        result["excavation"]["se"] = m.group(1).strip()
        blank_out(m)

    # Find number without "FN": "E 14/4", "B 15/7"
    if "fn" not in result["excavation"]:
        m = re.search(r"\b([A-Za-zЀ-ӿ])\s*(\d+/\d+)\b", remaining)
        if m:
            result["excavation"]["fn"] = f"{m.group(1)} {m.group(2)}"
            blank_out(m)

    # Quadrant: "Q 40 - 12", "Q - 40 - 12" -> "Q 40-12"
    m = re.search(r"\bQ\s*=?\s*-?\s*\d+\s*-\s*\d+\b", remaining, re.IGNORECASE)
    if m:
        quadrant = re.sub(r"^Q\s*=?\s*-?\s*", "Q ", m.group().strip())
        result["excavation"]["quadrant"] = re.sub(r"\s*-\s*", "-", quadrant).strip()
        blank_out(m)

    # Kiste (crate): "Ki 1/76"
    m = re.search(r"\bKi\s*\d+/\d+", remaining, re.IGNORECASE)
    if m:
        result["excavation"]["kiste"] = m.group().strip()
        blank_out(m)

    # Measurements: "RD = 12 cm", "R 8 cm % 10" -> ["8", "% 10"]
    measurements = list(re.finditer(
        r"\b(RD|BD|RD_D|BD_D|R_A|R_M|R_i|R|H|Н|i)\s*[=:]?\s*(\d+(?:[,\.]\d+)?)\s*(?:cm)?",
        remaining, re.IGNORECASE,
    ))
    for m in measurements:
        label = m.group(1).lower()
        if label == "r":
            label = "rd"
        elif label == "i":
            label = "r_i"
        elif label == "н": # Cyrillic "Н" read instead of "H"
            label = "h"
        value = m.group(2)

        # A percentage directly after the value belongs to it: "% 10" or "10 %"
        percent = re.match(r"^\s*(?:[=:;,-]|\bbzw\.?\b)?\s*(?:%\s*(\d+)|(\d+)\s*%)",
                           remaining[m.end():], re.IGNORECASE)
        if percent:
            result["measurement"][label] = [value, f"% {percent.group(1) or percent.group(2)}"]
            start, end = m.start(), m.end() + percent.end()
            remaining = remaining[:start] + " " * (end - start) + remaining[end:]
        else:
            result["measurement"][label] = value
            blank_out(m)

    # Publication: "Petznek Taf. 20/1788", "(Adler-Wölfl 2010 Taf. 98/ KE 2852)"
    found = [] # (value, position in text)
    for pattern in PUBLICATION_PATTERNS:
        for m in list(re.finditer(pattern, remaining, re.IGNORECASE)):
            value = m.group().strip()

            if not value:
                continue

            value = " ".join(value.split()).strip("- ")
            value = re.sub(r"\s*/\s*", "/", value)

            if value.endswith("/") and not re.search(r"/(?:KE|Str\.)", value, re.IGNORECASE):
                value = value.rstrip("/")

            if not any(m.start() == position for _, position in found):
                found.append((value, m.start()))

            blank_out(m)

    # Join parts that belong together, in reading order
    found.sort(key=lambda item: item[1])
    publications = []
    for value, _ in found:
        previous_has_author = publications and any(a in publications[-1].lower() for a in AUTHORS)

        # Closing part of a reference split over two lines: "(Adler-Wölfl ... Taf. 98/" + "KE 2852)"
        if (value.endswith(")") and publications and publications[-1].startswith("(")
                and not publications[-1].endswith(")") and "adler" in publications[-1].lower()):
            publications[-1] = f"{publications[-1]} {value}"
            continue

        is_plate_reference = re.match(r"^\(?(?:bzw\.\s*)?(?:Taf\.|Abb\.|KE|Str\.|str\.)",
                                      value, re.IGNORECASE)
        has_author = any(a in value.lower() for a in AUTHORS)
        if is_plate_reference and not has_author and previous_has_author:
            publications[-1] = f"{publications[-1]} {value}"
            continue

        publications.append(value)

    if publications:
        result["publication_type"] = publications[0] if len(publications) == 1 else publications

    # Surface treatment: "Üz a + i", "Grießbewurf", ... joined with " | "
    coatings = []
    for m in SURFACE_PATTERN.finditer(remaining):
        value = m.group().strip()
        if value:
            coatings.append(" ".join(value.split()))
            blank_out(m)
    if coatings:
        result["surface_treatment"] = " | ".join(coatings)

    # Artifact category: first (longest) match, spacing and dots flexible
    for category in ARTIFACT_CATEGORIES:
        pattern = (re.escape(category)
                   .replace("/", r"\s*/\s*")
                   .replace(r"\ ", r"\s*")
                   .replace(r"\.", r"\.?"))
        if category[0].isalnum():
            pattern = r"\b" + pattern
        if category[-1].isalnum() or category[-1] == ".":
            pattern = pattern + r"\b"
        m = re.search(pattern, remaining, re.IGNORECASE)
        if m:
            result["artifact_category"] = category
            blank_out(m)
            break

    # A firing note like "red/ox" is added to the category
    if result["artifact_category"]:
        m = re.search(r"\b(?:red|ox)\.?\s*/\s*(?:red|ox)\.?\b", remaining, re.IGNORECASE)
        if m:
            result["artifact_category"] += f" {m.group().strip()}"
            blank_out(m)

    # Pottery form
    forms = []
    for form in POTTERY_FORMS:
        m = re.search(r"\b" + re.escape(form) + r"\b", remaining, re.IGNORECASE)
        if m:
            found_form = text[m.start():m.end()].strip()
            if found_form not in forms:
                forms.append(found_form)
            blank_out(m)

    if len(forms) == 1:
        result["pottery_form"] = forms[0]
    elif len(forms) > 1:
        lower = [f.lower() for f in forms]
        if "flacher teller" in lower or ("teller" in lower and "flacher" in remaining.lower()):
            result["pottery_form"] = "Flacher Teller"
            remaining = re.sub(r"\bflacher\b", " ", remaining, flags=re.IGNORECASE)
        else:
            result["pottery_form"] = forms

    # Others: every token that is left
    left = re.sub(r"\b1\s*:\s*1\b", "1:1", remaining)
    result["others"] = [
        token.strip().strip(",;.:-/%[]?")
        for token in left.split()
        if token.strip().strip(",;.:-/%[]?") and token.strip() not in ["=", "m", "M"]
    ]
    return result
