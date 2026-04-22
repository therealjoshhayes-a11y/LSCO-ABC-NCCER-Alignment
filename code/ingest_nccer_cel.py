# ingest_nccer_cel.py
# Antikythera Pipeline — LSCO ABC NCCER Alignment
# NCCER CEL PDF ingest: OCR extraction and rules-based parsing
# Handles CEL format only — Pipefitting L1-L4 and Welding L2.
# CEL format differs from OBJ: standalone 5-digit module delimiter,
# title case module title, same objective structure as new OBJ format.

import re
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
import pandas as pd
from code.config import (
    TESSERACT_CMD, POPPLER_PATH, NCCER_FILES, PROCESSED_DIR
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ─────────────────────────────────────────────
# SKIP PATTERNS
# ─────────────────────────────────────────────

SKIP_TRIGGERS = [
    "Course Planning Tools",
    "Competencies/Equipment",
    "Classroom Equipment",
    "Before You Begin",
    "Additional Resources",
    "Recommended Teaching Time",
    "Copyright",
    "NCCERConnect",
    "LCD projector",
    "Whiteboard",
    "Markers/chalk",
    "Pencils and paper",
    "Computer with Internet",
    "PowerPoint",
    "Module Review",
    "Module Examinations",
    "Poster board",
    "Flip chart",
]

# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────

# CEL module delimiter: standalone 5-digit number
RE_MODULE_HEADER_CEL = re.compile(
    r"^(\d{5})\s*$"
)

# Learning Objective line
RE_LEARNING_OBJ = re.compile(
    r"^Learning Objective\s+\d+", re.IGNORECASE
)

# "Successful completion" descriptor line
RE_SUCCESSFUL_COMPLETION = re.compile(
    r"^Successful completion of this module prepares trainees to", re.IGNORECASE
)

# Sub-item: letter + period or bullet OCR artifact
RE_SUB_ITEM = re.compile(r"^[a-z]\.\s+\S")
RE_SUB_ITEM_BULLET = re.compile(r"^[¢°*]\s+\S")

# Performance Tasks header
RE_PERF_TASKS = re.compile(r"^Performance Tasks?", re.IGNORECASE)

# Knowledge-based module
RE_KNOWLEDGE_BASED = re.compile(r"knowledge.based module", re.IGNORECASE)

# Page header pattern — "Some Title | Module Name | N" — skip these
RE_PAGE_HEADER = re.compile(r".+\|.+\|\s*\d+\s*$")

# ─────────────────────────────────────────────
# LINE CLASSIFIER
# ─────────────────────────────────────────────

def should_skip_line(line: str) -> bool:
    for trigger in SKIP_TRIGGERS:
        if trigger.lower() in line.lower():
            return True
    return False


def is_module_header(line: str):
    """Return module_id if line is a CEL module delimiter, else None."""
    stripped = line.strip()
    m = RE_MODULE_HEADER_CEL.match(stripped)
    if m:
        return m.group(1)
    return None


def is_learning_objective(line: str) -> bool:
    return bool(RE_LEARNING_OBJ.match(line.strip()))


def is_successful_completion(line: str) -> bool:
    return bool(RE_SUCCESSFUL_COMPLETION.match(line.strip()))


def is_sub_item(line: str) -> bool:
    stripped = line.strip()
    return bool(RE_SUB_ITEM.match(stripped)) or bool(RE_SUB_ITEM_BULLET.match(stripped))


def is_performance_task_header(line: str) -> bool:
    return bool(RE_PERF_TASKS.match(line.strip()))


def is_knowledge_based(line: str) -> bool:
    return bool(RE_KNOWLEDGE_BASED.search(line))


def is_page_header(line: str) -> bool:
    return bool(RE_PAGE_HEADER.match(line.strip()))


# ─────────────────────────────────────────────
# OCR — PDF TO TEXT
# ─────────────────────────────────────────────

def pdf_to_text(filepath: Path) -> str:
    """Convert PDF to images at 300dpi and run Tesseract OCR."""
    print(f"  OCR: {filepath.name}")
    images = convert_from_path(
        filepath, dpi=300, poppler_path=POPPLER_PATH
    )
    pages = []
    for image in images:
        text = pytesseract.image_to_string(image)
        pages.append(text)
    return "\n".join(pages)


# ─────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────

def parse_cel_text(raw_text: str, credential: str) -> list[dict]:
    """
    Parse OCR text from a CEL format PDF.
    One record per module with concatenated claims block.
    """
    lines = raw_text.split("\n")
    modules = []
    current_module = None
    in_objectives = False
    in_perf_tasks = False
    skip_mode = False
    title_captured = False

    for line in lines:
        line_stripped = line.strip()

        if not line_stripped:
            continue

        # Skip page headers
        if is_page_header(line_stripped):
            continue

        # Skip administrative content
        if should_skip_line(line_stripped):
            skip_mode = True
            in_objectives = False
            in_perf_tasks = False
            continue

        # Check for module header — resets everything
        module_id = is_module_header(line_stripped)
        if module_id:
            if current_module and current_module.get("module_id"):
                modules.append(current_module)

            current_module = {
                "credential":   credential,
                "module_id":    module_id,
                "module_title": "",
                "objectives":   [],
                "perf_tasks":   [],
            }
            skip_mode = False
            in_objectives = False
            in_perf_tasks = False
            title_captured = False
            continue

        if not current_module:
            continue

        if skip_mode:
            continue

        # Capture module title — first non-empty line after module header
        # that is not a skip trigger and not a learning objective
        if not title_captured and not is_learning_objective(line_stripped):
            if not line_stripped.lower() in ("pipefitting", "welding",
                                              "overview", "pipefitting level one",
                                              "welding level two"):
                current_module["module_title"] = line_stripped
                title_captured = True
            continue

        # Learning objective header
        if is_learning_objective(line_stripped):
            in_objectives = True
            in_perf_tasks = False
            current_module["objectives"].append(line_stripped)
            continue

        # Successful completion descriptor
        if in_objectives and is_successful_completion(line_stripped):
            current_module["objectives"].append(line_stripped)
            continue

        # Performance tasks header
        if is_performance_task_header(line_stripped):
            in_objectives = False
            in_perf_tasks = True
            continue

        # Knowledge-based note
        if is_knowledge_based(line_stripped):
            current_module["perf_tasks"].append("knowledge-based")
            in_perf_tasks = False
            continue

        # Sub-items
        if in_objectives and is_sub_item(line_stripped):
            current_module["objectives"].append(line_stripped)
            continue

        # Performance task text
        if in_perf_tasks and line_stripped:
            if not line_stripped.startswith("Recommended"):
                current_module["perf_tasks"].append(line_stripped)
            else:
                in_perf_tasks = False
            continue

    # Append final module
    if current_module and current_module.get("module_id"):
        modules.append(current_module)

    return modules


# ─────────────────────────────────────────────
# CLAIMS BLOCK BUILDER
# ─────────────────────────────────────────────

def build_module_claims_block(module: dict) -> str:
    parts = []
    if module.get("module_title"):
        parts.append(module["module_title"])
    for obj in module.get("objectives", []):
        parts.append(obj)
    for task in module.get("perf_tasks", []):
        if task != "knowledge-based":
            parts.append(task)
    return " ".join(parts)


# ─────────────────────────────────────────────
# SAVE TO PROCESSED
# ─────────────────────────────────────────────

def save_processed(records: list[dict], filename: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outpath = PROCESSED_DIR / filename
    rows = []
    for r in records:
        rows.append({
            "credential":   r["credential"],
            "module_id":    r["module_id"],
            "module_title": r["module_title"],
            "objectives":   " | ".join(r["objectives"]),
            "perf_tasks":   " | ".join(r["perf_tasks"]),
            "claims_block": r["claims_block"],
            "source_file":  r.get("source_file", ""),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    print(f"Saved: {outpath} ({len(df)} records)")
    return outpath


# ─────────────────────────────────────────────
# PROCESS SINGLE FILE
# ─────────────────────────────────────────────

def process_cel_file(filepath: Path, credential: str) -> list[dict]:
    """OCR + parse a single CEL PDF. Returns list of module records."""
    raw_text = pdf_to_text(filepath)
    modules  = parse_cel_text(raw_text, credential)
    for module in modules:
        module["claims_block"] = build_module_claims_block(module)
        module["source_file"]  = filepath.name
    print(f"  Modules extracted: {len(modules)}")
    return modules


# ─────────────────────────────────────────────
# CREDENTIAL MAP
# ─────────────────────────────────────────────

CEL_CREDENTIAL_MAP = {
    "t3_pipefitting": "NCCER Pipefitting",
    "t2_welding_l2":  "NCCER Welding",
}

CEL_FILES = {
    "t3_pipefitting": NCCER_FILES["t3_pipefitting"],
    "t2_welding_l2":  [f for f in NCCER_FILES["t2_welding"]
                       if "CEL" in f.name],
}


# ─────────────────────────────────────────────
# MAIN INGEST
# ─────────────────────────────────────────────

def run_cel_ingest() -> dict:
    """
    Process all CEL PDF files — Pipefitting L1-L4 and Welding L2.
    Saves one CSV per trace to processed/.
    """
    print("\n=== NCCER CEL FULL INGEST ===")
    all_results = {}

    for trace_id, credential in CEL_CREDENTIAL_MAP.items():
        files = CEL_FILES.get(trace_id, [])
        if not files:
            print(f"No files for {trace_id} — skipping.")
            continue

        print(f"\nTrace {trace_id}: {credential}")
        trace_modules = []

        for filepath in files:
            modules = process_cel_file(filepath, credential)
            trace_modules.extend(modules)

        filename = "t3_pipefitting_processed.csv" if "pipefitting" in trace_id \
                   else "t2_welding_cel_processed.csv"
        save_processed(trace_modules, filename)
        all_results[trace_id] = trace_modules
        print(f"  Total modules for {trace_id}: {len(trace_modules)}")

    print("\n=== CEL INGEST COMPLETE ===")
    return all_results


if __name__ == "__main__":
    run_cel_ingest()