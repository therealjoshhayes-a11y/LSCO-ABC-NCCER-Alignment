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

# Format 1: "Lesson Plans for Module XXXXX" (Core, Welding, Scaffold)
RE_MODULE_HEADER_CLASSIC = re.compile(
    r"Lesson Plans for Module\s+(\d{5})", re.IGNORECASE
)

# Format 2: "Lesson Plans for XXXXX-XX" (Instrumentation)
RE_MODULE_HEADER_DATED = re.compile(
    r"Lesson Plans for\s+(\d{5}-\d{2})", re.IGNORECASE
)

# Format 3a: "XXXXX-XX" standalone line (Electrical)
RE_MODULE_HEADER_NEW = re.compile(
    r"^(\d{5}-\d{2})\s*$"
)

# Format 3b: "XX/XXX" OCR artifact for module numbers (Carpentry Form/Advanced)
RE_MODULE_HEADER_SLASH = re.compile(
    r"^(\d{1,2})/(\d{3})\s*$"
)

# Learning Objective line
RE_LEARNING_OBJ = re.compile(
    r"^Learning Objective\s+\d+", re.IGNORECASE
)

# "Successful completion" descriptor line — new format objectives
RE_SUCCESSFUL_COMPLETION = re.compile(
    r"^Successful completion of this module prepares trainees to", re.IGNORECASE
)

# Sub-item line: starts with "a." "b." "c." or bullet "e " (OCR artifact for bullet)
RE_SUB_ITEM = re.compile(
    r"^[a-z]\.\s+\S"
)

RE_SUB_ITEM_BULLET = re.compile(
    r"^e\s+[A-Z]"
)

# Performance Tasks header
RE_PERF_TASKS = re.compile(
    r"^Performance Tasks?", re.IGNORECASE
)

# Knowledge-based module
RE_KNOWLEDGE_BASED = re.compile(
    r"knowledge.based module", re.IGNORECASE
)

# ─────────────────────────────────────────────
# LINE CLASSIFIER
# ─────────────────────────────────────────────

def should_skip_line(line: str) -> bool:
    for trigger in SKIP_TRIGGERS:
        if trigger.lower() in line.lower():
            return True
    return False


def is_module_header(line: str):
    """
    Return (module_id, format_name) if line is a module header, else None.
    Tries all four format patterns in priority order.
    """
    stripped = line.strip()

    m = RE_MODULE_HEADER_CLASSIC.search(stripped)
    if m:
        return m.group(1), "classic"

    m = RE_MODULE_HEADER_DATED.search(stripped)
    if m:
        return m.group(1).replace("-", ""), "dated"

    m = RE_MODULE_HEADER_NEW.match(stripped)
    if m:
        return m.group(1).replace("-", ""), "new"

    m = RE_MODULE_HEADER_SLASH.match(stripped)
    if m:
        return m.group(1) + m.group(2), "slash"

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

            module_id, fmt = module_match
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

        # Capture "Successful completion" descriptor as part of objective
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
# CREDENTIAL LABEL MAP
# OBJ traces only — excludes CEL and XLSX traces
# ─────────────────────────────────────────────

OBJ_CREDENTIAL_MAP = {
    "t0_core":            "NCCER Core",
    "t1_electrical":      "NCCER Electrical",
    "t2_welding":         "NCCER Welding",
    "t4_instrumentation": "NCCER Instrumentation",
    "t5_carpentry":       "NCCER Carpentry",
    "t7_scaffold":        "NCCER Scaffold Builder",
}


def run_obj_ingest() -> dict:
    """
    Process all OBJ PDF files across all applicable traces.
    Saves one CSV per trace to processed/.
    CEL format handled by ingest_nccer_cel.py.
    Rigger XLSX handled by ingest_rigger.py.
    """
    print("\n=== NCCER OBJ FULL INGEST ===")
    all_results = {}

    for trace_id, credential in OBJ_CREDENTIAL_MAP.items():
        files = NCCER_FILES.get(trace_id, [])
        if not files:
            print(f"No files for {trace_id} — skipping.")
            continue

        print(f"\nTrace {trace_id}: {credential}")
        trace_modules = []

        for filepath in files:
            if "CEL" in filepath.name:
                print(f"  Skipping CEL file: {filepath.name}")
                continue
            modules = process_obj_file(filepath, credential)
            trace_modules.extend(modules)

        save_processed(trace_modules, f"{trace_id}_processed.csv")
        all_results[trace_id] = trace_modules
        print(f"  Total modules for {trace_id}: {len(trace_modules)}")

    print("\n=== OBJ FULL INGEST COMPLETE ===")
    return all_results


if __name__ == "__main__":
    run_obj_ingest()