# ingest_acgm.py
# Antikythera Pipeline — LSCO ABC NCCER Alignment
# ACGM PDF ingest: OCR extraction for IDF corpus construction.
# ACGM is IDF corpus source only — not a claims block source.
# One corpus string per content unit: title + description + outcomes.

import re
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
import pandas as pd
from code.config import (
    TESSERACT_CMD, POPPLER_PATH, ACGM_FILE, PROCESSED_DIR
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────

# Course header: "RUBRIC NNNN Course Title"
RE_COURSE_HEADER = re.compile(
    r"^([A-Z]{2,6})\s+(\d{4})\s+(.+)$"
)

# Rubric section header: "RUBRIC - (Discipline)"
RE_RUBRIC_SECTION = re.compile(
    r"^[A-Z]{2,6}\s+-\s+\(.+\)\s*$"
)

# Learning Outcomes anchor
RE_OUTCOMES_ANCHOR = re.compile(
    r"Upon successful completion of this course", re.IGNORECASE
)

# Skip these metadata lines
SKIP_PREFIXES = [
    "Prerequisite:",
    "Corequisite:",
    "Approval Number:",
    "Maximum SCH",
    "Learning Outcomes",
    "None",
]

# ─────────────────────────────────────────────
# LINE CLASSIFIER
# ─────────────────────────────────────────────

def should_skip(line: str) -> bool:
    for prefix in SKIP_PREFIXES:
        if line.startswith(prefix):
            return True
    return False


def is_rubric_section(line: str) -> bool:
    return bool(RE_RUBRIC_SECTION.match(line.strip()))


def is_course_header(line: str):
    """Return (rubric, number, title) or None."""
    m = RE_COURSE_HEADER.match(line.strip())
    if m:
        return m.group(1), m.group(2), m.group(3).strip()
    return None


def is_outcomes_anchor(line: str) -> bool:
    return bool(RE_OUTCOMES_ANCHOR.search(line))


# ─────────────────────────────────────────────
# CONTENT UNIT KEY
# ─────────────────────────────────────────────

def get_content_key(rubric: str, number: str) -> str:
    """
    Content-stable deduplication key.
    Same as WECM: Rubric_Level_ContentID
    Number format: LSCD — level digit + SCH digit + 2-digit content ID
    """
    if len(number) >= 4:
        level      = number[0]
        content_id = number[2:]
        return f"{rubric}_{level}_{content_id}"
    return f"{rubric}_{number}"


# ─────────────────────────────────────────────
# OCR — PDF TO TEXT
# ─────────────────────────────────────────────

def pdf_to_text_pages(filepath: Path) -> list[str]:
    """
    Convert PDF to images at 300dpi and run Tesseract OCR.
    Returns list of page text strings.
    """
    print(f"OCR: {filepath.name} — this will take several minutes...")
    images = convert_from_path(filepath, dpi=300, poppler_path=POPPLER_PATH)
    pages = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        pages.append(text)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(images)} pages...")
    return pages


# ─────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────

def parse_acgm_text(pages: list[str]) -> list[dict]:
    """
    Parse OCR text from ACGM PDF.
    Returns one record per SCH variant.
    Deduplication applied downstream.
    """
    records = []
    current = {}
    in_description = False
    in_outcomes = False

    for page_text in pages:
        lines = page_text.split("\n")

        for line in lines:
            line_stripped = line.strip()

            if not line_stripped:
                continue

            # Skip rubric section headers
            if is_rubric_section(line_stripped):
                continue

            # Skip metadata lines
            if should_skip(line_stripped):
                in_description = False
                in_outcomes = False
                continue

            # Detect course header
            header = is_course_header(line_stripped)
            if header:
                # Save previous record
                if current.get("rubric"):
                    records.append(current)

                rubric, number, title = header
                current = {
                    "rubric":      rubric,
                    "number":      number,
                    "title":       title,
                    "description": [],
                    "outcomes":    [],
                }
                in_description = True
                in_outcomes = False
                continue

            if not current:
                continue

            # Learning outcomes anchor
            if is_outcomes_anchor(line_stripped):
                in_description = False
                in_outcomes = True
                continue

            # Collect description
            if in_description and not in_outcomes:
                current["description"].append(line_stripped)
                continue

            # Collect outcomes
            if in_outcomes:
                current["outcomes"].append(line_stripped)
                continue

    # Append final record
    if current.get("rubric"):
        records.append(current)

    print(f"Raw records parsed: {len(records)}")
    return records


# ─────────────────────────────────────────────
# DEDUPLICATION AND CORPUS STRING
# ─────────────────────────────────────────────

def deduplicate_and_build_corpus(records: list[dict]) -> list[dict]:
    """
    Deduplicate by content unit key.
    Build corpus string: title + description + outcomes.
    """
    content_units = {}

    for r in records:
        key = get_content_key(r["rubric"], r["number"])
        if key not in content_units:
            desc     = " ".join(r["description"]).strip()
            outcomes = " ".join(r["outcomes"]).strip()
            corpus   = " ".join(filter(None, [r["title"], desc, outcomes]))

            content_units[key] = {
                "content_key":   key,
                "rubric":        r["rubric"],
                "number":        r["number"],
                "title":         r["title"],
                "description":   desc,
                "outcomes":      outcomes,
                "corpus_string": corpus,
            }

    deduped = list(content_units.values())
    print(f"Content units after deduplication: {len(deduped)}")
    return deduped


# ─────────────────────────────────────────────
# SAVE TO PROCESSED
# ─────────────────────────────────────────────

def save_processed(records: list[dict], filename: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outpath = PROCESSED_DIR / filename
    df = pd.DataFrame(records)
    df.to_csv(outpath, index=False)
    print(f"Saved: {outpath} ({len(df)} records)")
    return outpath


# ─────────────────────────────────────────────
# MAIN INGEST
# ─────────────────────────────────────────────

def run_acgm_ingest() -> list[dict]:
    """
    Full ACGM ingest pipeline:
    1. OCR all 266 pages
    2. Parse course records
    3. Deduplicate by content unit
    4. Build corpus strings
    5. Save to processed/
    Returns list of content unit dicts.
    """
    print("\n=== ACGM INGEST ===")
    pages   = pdf_to_text_pages(ACGM_FILE)
    records = parse_acgm_text(pages)
    units   = deduplicate_and_build_corpus(records)
    save_processed(units, "acgm_processed.csv")
    print("=== ACGM INGEST COMPLETE ===\n")
    return units


if __name__ == "__main__":
    run_acgm_ingest()