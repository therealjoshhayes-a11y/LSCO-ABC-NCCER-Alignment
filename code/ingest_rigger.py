# ingest_rigger.py
# Antikythera Pipeline — LSCO ABC NCCER Alignment
# NCCER Rigger XLSX ingest: direct structured read for Basic, Intermediate,
# and Advanced Rigger objective files. No OCR required.
# One claims block per module — all objectives concatenated.

import openpyxl
import pandas as pd
from pathlib import Path
from code.config import NCCER_FILES, PROCESSED_DIR

# ─────────────────────────────────────────────
# RIGGER FILE REGISTRY
# ─────────────────────────────────────────────

RIGGER_FILES = {
    "basic":        NCCER_FILES["t6_rigging"][0],
    "intermediate": NCCER_FILES["t6_rigging"][1],
    "advanced":     NCCER_FILES["t6_rigging"][2],
}

SKIP_VALUES = {
    "Learning Objectives / Competencies",
    "Craft: ",
    "Basic Rigger",
    "Intermediate Rigger",
    "Advanced Rigger",
    "Module Number",
}

# ─────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────

def parse_rigger_file(filepath: Path, level: str) -> list[dict]:
    """
    Parse a single Rigger XLSX file.
    Groups all objectives under their module number.
    Returns one record per module with concatenated claims block.
    """
    print(f"Parsing: {filepath.name}")
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active

    modules = {}
    module_order = []

    for row in ws.iter_rows(values_only=True):
        if not any(v is not None for v in row):
            continue

        module_raw = row[0]
        obj_num    = row[1]
        obj_text   = row[2]

        # Skip header rows
        if str(obj_text).strip() in SKIP_VALUES:
            continue
        if str(module_raw).strip() in SKIP_VALUES or module_raw is None:
            continue
        if str(obj_text).strip() == "Objectives":
            continue

        module_id = str(module_raw).strip()
        obj_text  = str(obj_text).strip() if obj_text else ""

        if not obj_text:
            continue

        if module_id not in modules:
            modules[module_id] = []
            module_order.append(module_id)

        modules[module_id].append(obj_text)

    wb.close()

    records = []
    for module_id in module_order:
        objectives = modules[module_id]
        claims_block = " ".join(objectives)
        records.append({
            "credential":    f"NCCER Rigger {level.capitalize()}",
            "level":         level,
            "module_id":     module_id,
            "objectives":    objectives,
            "claims_block":  claims_block,
            "source_file":   filepath.name,
        })

    print(f"  Modules parsed: {len(records)}")
    return records


# ─────────────────────────────────────────────
# SAVE TO PROCESSED
# ─────────────────────────────────────────────

def save_processed(records: list[dict], filename: str) -> Path:
    """Save processed rigger records to processed/ as CSV."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outpath = PROCESSED_DIR / filename
    # objectives list won't serialize cleanly — convert to string
    df = pd.DataFrame(records)
    df["objectives"] = df["objectives"].apply(lambda x: " | ".join(x))
    df.to_csv(outpath, index=False)
    print(f"Saved: {outpath} ({len(df)} records)")
    return outpath


# ─────────────────────────────────────────────
# MAIN INGEST PIPELINE
# ─────────────────────────────────────────────

def run_rigger_ingest() -> list[dict]:
    """
    Full Rigger ingest pipeline:
    1. Parse all three Rigger XLSX files
    2. Build claims blocks per module
    3. Save combined processed output
    Returns combined list of module records.
    """
    print("\n=== RIGGER INGEST ===")
    all_records = []

    for level, filepath in RIGGER_FILES.items():
        records = parse_rigger_file(filepath, level)
        all_records.extend(records)

    save_processed(all_records, "rigger_processed.csv")
    print(f"Total modules across all Rigger levels: {len(all_records)}")
    print("=== RIGGER INGEST COMPLETE ===\n")
    return all_records


if __name__ == "__main__":
    run_rigger_ingest()