# ingest_nccer_obj.py
# Antikythera Pipeline — LSCO ABC NCCER Alignment
# NCCER OBJ PDF ingest: OCR extraction and rules-based parsing
# Handles OBJ format only. CEL format uses separate module.
# One claims block per module — title + all learning objectives + performance tasks.

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
# Administrative content — skip these sections
# ─────────────────────────────────────────────

SKIP_TRIGGERS = [
    "Before You Begin",
    "Safety Considerations",
    "Classroom Equipment",
    "Additional Resources",
    "Materials Checklist",
    "Recommended Teaching Time",
    "Copyright",
    "NCCERConnect",
    "LCD projector",
    "Whiteboard",
    "Markers/chalk",
    "Pencils and paper",
    "Computer with Internet",
]

# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────

# Module delimiter: "Lesson Plans for Module XXXXX"
RE_MODULE_HEADER = re.compile(
    r"Lesson Plans for Module\s+(\d{5})", re.IGNORECASE
)

# Learning Objective line: "Learning Objective N"
RE_LEARNING_OBJ = re.compile(
    r"^Learning Objective\s+\d+", re.IGNORECASE
)

# Sub-item line: starts with "a." "b." "c." etc.
RE_SUB_ITEM = re.compile(
    r"^[a-z]\.\s+\S"
)

# Performance Tasks header
RE_PERF_TASKS = re.compile(
    r"^Performance Tasks?", re.IGNORECASE
)

# Knowledge-based module — no performance tasks
RE_KNOWLEDGE_BASED = re.compile(
    r"knowledge.based module", re.IGNORECASE
)

# ─────────────────────────────────────────────
# LINE CLASSIFIER
# ─────────────────────────────────────────────

def should_skip_line(line: str) -> bool:
    """Return True if this line is administrative content to skip."""
    for trigger in SKIP_TRIGGERS:
        if trigger.lower() in line.lower():
            return True
    return False


def is_module_header(line: str):
    """Return match object if line is a module header, else None."""
    return RE_MODULE_HEADER.search(line)


def is_learning_objective(line: str) -> bool:
    return bool(RE_LEARNING_OBJ.match(line.strip()))


def is_sub_item(line: str) -> bool:
    return bool(RE_SUB_ITEM.match(line.strip()))


def is_performance_task_header(line: str) -> bool:
    return bool(RE_PERF_TASKS.match(line.strip()))


def is_knowledge_based(line: str) -> bool:
    return bool(RE_KNOWLEDGE_BASED.search(line))

# ─────────────────────────────────────────────
# OCR — PDF TO TEXT
# ─────────────────────────────────────────────

def pdf_to_text(filepath: Path) -> str:
    """
    Convert PDF to images at 300dpi and run Tesseract OCR.
    Returns full text of all pages concatenated.
    """
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

def parse_obj_text(raw_text: str, credential: str) -> list[dict]:
    """
    Parse OCR text from an OBJ format PDF.
    Extracts one record per module containing:
    - module_id, module_title, credential
    - learning objectives text
    - performance tasks text
    - claims_block (concatenated)
    """
    lines = raw_text.split("\n")
    modules = []
    current_module = None
    in_objectives = False
    in_perf_tasks = False
    skip_mode = False

    for line in lines:
        line_stripped = line.strip()

        if not line_stripped:
            continue

        # Check for skip triggers
        if should_skip_line(line_stripped):
            skip_mode = True
            in_objectives = False
            in_perf_tasks = False
            continue

        # Check for new module header — resets skip mode
        module_match = is_module_header(line_stripped)
        if module_match:
            # Save previous module
            if current_module and current_module.get("module_id"):
                modules.append(current_module)

            module_id = module_match.group(1)
            current_module = {
                "credential":    credential,
                "module_id":     module_id,
                "module_title":  "",
                "objectives":    [],
                "perf_tasks":    [],
            }
            skip_mode = False
            in_objectives = False
            in_perf_tasks = False
            continue

        if not current_module:
            continue

        if skip_mode:
            continue

        # Capture module title — ALL CAPS line after header
        if not current_module["module_title"] and line_stripped.isupper():
            current_module["module_title"] = line_stripped.title()
            continue

        # Learning objective header line
        if is_learning_objective(line_stripped):
            in_objectives = True
            in_perf_tasks = False
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

        # Collect objective sub-items
        if in_objectives and is_sub_item(line_stripped):
            current_module["objectives"].append(line_stripped)
            continue

        # Collect performance task text
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
    """
    Concatenate module title + objectives + performance tasks
    into a single claims block string for SBERT embedding.
    Excludes 'knowledge-based' placeholder from embedding text.
    """
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
    """Save processed module records to processed/ as CSV."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outpath = PROCESSED_DIR / filename

    rows = []
    for r in records:
        rows.append({
            "credential":    r["credential"],
            "module_id":     r["module_id"],
            "module_title":  r["module_title"],
            "objectives":    " | ".join(r["objectives"]),
            "perf_tasks":    " | ".join(r["perf_tasks"]),
            "claims_block":  r["claims_block"],
            "source_file":   r.get("source_file", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    print(f"Saved: {outpath} ({len(df)} records)")
    return outpath


# ─────────────────────────────────────────────
# PROCESS SINGLE FILE
# ─────────────────────────────────────────────

def process_obj_file(filepath: Path, credential: str) -> list[dict]:
    """OCR + parse a single OBJ PDF. Returns list of module records."""
    raw_text = pdf_to_text(filepath)
    modules  = parse_obj_text(raw_text, credential)

    for module in modules:
        module["claims_block"] = build_module_claims_block(module)
        module["source_file"]  = filepath.name

    print(f"  Modules extracted: {len(modules)}")
    return modules


# ─────────────────────────────────────────────
# MAIN INGEST — CORE ONLY (TEST RUN)
# ─────────────────────────────────────────────

def run_core_ingest() -> list[dict]:
    """
    Process NCCER Core OBJ PDF only.
    Use this to validate parser before running all traces.
    """
    print("\n=== NCCER CORE OBJ INGEST ===")
    filepath = NCCER_FILES["t0_core"][0]
    modules  = process_obj_file(filepath, "NCCER Core")
    save_processed(modules, "nccer_core_processed.csv")
    print("=== CORE INGEST COMPLETE ===\n")
    return modules


if __name__ == "__main__":
    run_core_ingest()