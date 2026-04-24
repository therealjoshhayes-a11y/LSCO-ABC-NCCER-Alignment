# config.py
# Antikythera Pipeline — LSCO ABC NCCER Alignment
# Project configuration: paths, trace definitions, run parameters
# All other modules import from this file. Do not hardcode paths elsewhere.

from pathlib import Path
import datetime

# ─────────────────────────────────────────────
# ROOT PATHS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR        = PROJECT_ROOT / "input"
SOURCE_DATA_DIR  = INPUT_DIR / "source_data"
PROVENANCE_DIR   = INPUT_DIR / "provenance"

PROCESSED_DIR    = PROJECT_ROOT / "processed"
INTERIM_DIR      = PROJECT_ROOT / "interim"
EMBEDDINGS_DIR   = INTERIM_DIR / "embeddings"
SCORES_DIR       = INTERIM_DIR / "scores"
OUTPUT_DIR       = PROJECT_ROOT / "output"
LOGS_DIR         = PROJECT_ROOT / "logs"
DOCS_DIR         = PROJECT_ROOT / "documentation"

# ─────────────────────────────────────────────
# SOURCE DATA PATHS
# ─────────────────────────────────────────────

WECM_FILE        = SOURCE_DATA_DIR / "wecm" / "Search_Course_WECM_3.xlsx"
ACGM_FILE        = SOURCE_DATA_DIR / "acgm" / "Academic_Course_Guide_Manual.pdf"
NCCER_CATALOG    = SOURCE_DATA_DIR / "nccer_catalog" / "NCCER-FULL-Catalog.pdf"

DEGREE_PLANS_DIR = SOURCE_DATA_DIR / "degree_plans"
NCCER_OBJ_DIR    = SOURCE_DATA_DIR / "nccer" / "obj"
NCCER_CEL_DIR    = SOURCE_DATA_DIR / "nccer" / "cel"

# ─────────────────────────────────────────────
# DEGREE PLAN FILES
# ─────────────────────────────────────────────

DEGREE_PLANS = {
    "electrical_aas":      DEGREE_PLANS_DIR / "Electrical_AAS.html",
    "welding_aas":         DEGREE_PLANS_DIR / "Welding_AAS.html",
    "instrumentation_aas": DEGREE_PLANS_DIR / "Instrumentation_AAS.html",
    "pipefitting_cc":      DEGREE_PLANS_DIR / "Pipefitting_CC.html",
    "bct_aas":             DEGREE_PLANS_DIR / "BCT_AAS.html",
    "bcm_aas":             DEGREE_PLANS_DIR / "BCM_AAS.html",
}

# ─────────────────────────────────────────────
# NCCER SOURCE FILES BY TRACE
# ─────────────────────────────────────────────

NCCER_FILES = {
    "t0_core": [
        NCCER_OBJ_DIR / "core" / "OBJ_Core_6E.pdf",
    ],
    "t1_electrical": [
        NCCER_OBJ_DIR / "electrical" / "OBJ_Electrical_11E_L1.pdf",
        NCCER_OBJ_DIR / "electrical" / "OBJ_Electrical_11E_L2.pdf",
        NCCER_OBJ_DIR / "electrical" / "OBJ_Electrical_11E_L3.pdf",
        NCCER_OBJ_DIR / "electrical" / "OBJ_Electrical_11E_L4.pdf",
    ],
    "t2_welding": [
        NCCER_OBJ_DIR / "welding" / "OBJ_Welding_6E_L1.pdf",
        NCCER_CEL_DIR / "welding" / "CEL_6E_Welding_L2.pdf",
    ],
    "t3_pipefitting": [
        NCCER_CEL_DIR / "pipefitting" / "CEL_Pipefitting_4E_L1.pdf",
        NCCER_CEL_DIR / "pipefitting" / "CEL_Pipefitting_4E_L2.pdf",
        NCCER_CEL_DIR / "pipefitting" / "CEL_Pipefitting_4E_L3.pdf",
        NCCER_CEL_DIR / "pipefitting" / "CEL_Pipefitting_4E_L4.pdf",
    ],
    "t4_instrumentation": [
        NCCER_OBJ_DIR / "instrumentation" / "OBJ_Instrumentation_3E_L1.pdf",
        NCCER_OBJ_DIR / "instrumentation" / "OBJ_Instrumentation_3E_L2.pdf",
        NCCER_OBJ_DIR / "instrumentation" / "OBJ_Instrumentation_3E_L3.pdf",
        NCCER_OBJ_DIR / "instrumentation" / "OBJ_Instrumentation_3E_L4.pdf",
    ],
    "t5_carpentry": [
        NCCER_OBJ_DIR / "carpentry" / "OBJ_Carpentry_6E_GeneralCarpentry.pdf",
        NCCER_OBJ_DIR / "carpentry" / "OBJ_Carpentry_6E_FormCarpentry.pdf",
        NCCER_OBJ_DIR / "carpentry" / "OBJ_Carpentry_6E_AdvancedCarpentry.pdf",
    ],
    "t6_rigging": [
        NCCER_OBJ_DIR / "rigging" / "OBJ_BasicRigger_3E.xlsx",
        NCCER_OBJ_DIR / "rigging" / "OBJ_IntermediateRigger_2E.xlsx",
        NCCER_OBJ_DIR / "rigging" / "OBJ_AdvancedRigger_2E.xlsx",
    ],
    "t7_scaffold": [
        NCCER_OBJ_DIR / "scaffolding" / "OBJ_Scaffolding_2E_L1.pdf",
    ],
}

# ─────────────────────────────────────────────
# TRACE DEFINITIONS
# ─────────────────────────────────────────────

TRACES = {
    "t0_core": {
        "name": "NCCER Core",
        "cip_scope": ["46", "48", "15"],
        "target_programs": ["electrical_aas", "welding_aas", "instrumentation_aas",
                            "pipefitting_cc", "bct_aas", "bcm_aas"],
        "rubric_scope": None,
        "notes": "Shared prerequisite layer. Run before all craft traces. Lock allocations before proceeding.",
    },
    "t1_electrical": {
        "name": "NCCER Electrical L1-L4",
        "cip_scope": ["46.0300"],
        "target_programs": ["electrical_aas"],
        "rubric_scope": ["ELPT", "ELTN", "INTC", "OSHT"],
        "notes": "Post-Core courses only. Exclude Core ledger allocations.",
    },
    "t2_welding": {
        "name": "NCCER Welding L1-L2",
        "cip_scope": ["48.0500"],
        "target_programs": ["welding_aas"],
        "rubric_scope": ["WLDG"],
        "notes": "Post-divergence welding track only. Exclude Core ledger allocations.",
    },
    "t3_pipefitting": {
        "name": "NCCER Pipefitting L1-L4",
        "cip_scope": ["46.0500"],
        "target_programs": ["welding_aas", "pipefitting_cc"],
        "rubric_scope": ["PFPB"],
        "notes": "Post-divergence pipefitting track. Pipefitting CC resolves as K=4 subset. Exclude Core ledger.",
    },
    "t4_instrumentation": {
    "name": "NCCER Instrumentation L1-L4",
    "cip_scope": ["15.0400"],
    "target_programs": ["instrumentation_aas"],
    "rubric_scope": ["INTC", "PTAC", "INCR", "ELPT"],
    "notes": "Post-Core. Note scheduling constraints. Exclude Core ledger allocations.",
    },
    "t5_carpentry": {
        "name": "NCCER Carpentry (General, Form, Advanced)",
        "cip_scope": ["46.0400"],
        "target_programs": ["bct_aas", "bcm_aas"],
        "rubric_scope": ["CNBT", "WDWK"],
        "notes": "Full scope for BCT. BCM inherits from BCT ledger at coverage map stage — no independent SME queue.",
    },
    "t6_rigging": {
        "name": "NCCER Rigging (Basic, Intermediate, Advanced)",
        "cip_scope": ["46"],
        "target_programs": ["bct_aas"],
        "rubric_scope": ["CNBT", "WDWK"],
        "notes": "BCT candidate pool only. BCM inherits from BCT ledger. No CNSE in BCT degree plan.",
    },
    "t7_scaffold": {
        "name": "NCCER Scaffold Builder L1",
        "cip_scope": ["46"],
        "target_programs": ["bct_aas"],
        "rubric_scope": ["CNBT", "WDWK"],
        "notes": "BCT candidate pool only. BCM inherits from BCT ledger. No CNSE in BCT degree plan.",
    },
}

# ─────────────────────────────────────────────
# PIPELINE PARAMETERS
# ─────────────────────────────────────────────

MODEL_NAME       = "all-mpnet-base-v2"
RANDOM_SEED      = 42

TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF       = 2
TFIDF_NGRAM_RANGE  = (1, 3)

# ─────────────────────────────────────────────
# RUN ID
# ─────────────────────────────────────────────

def get_run_id():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ─────────────────────────────────────────────
# BCM SCOPE RESTRICTION
# ─────────────────────────────────────────────

BCM_ALLOWED_RUBRICS = ["CNBT"]
# BCM AAS included at limited scope — T-0 only, CNBT rubric courses only.
# All other rubrics excluded from BCM candidate pool.

# ─────────────────────────────────────────────
# TESSERACT AND POPPLER
# ─────────────────────────────────────────────

TESSERACT_CMD = r"C:\Users\jhayes4\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
POPPLER_PATH  = r"C:\Users\jhayes4\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin"