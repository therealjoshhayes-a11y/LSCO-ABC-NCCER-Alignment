"""
ledger.py — Queue Builder and Allocation Ledger
Antikythera Pipeline | LSCO–ABC Alignment Project

Builds the reordered SME presentation queue per trace × program × module:
  1. Full SBERT ranking across candidate pool
  2. TF-IDF promotion — +1 position if TF-IDF rank is higher
  3. Jaccard promotion — +1 position if Jaccard rank is higher (after TF-IDF)
  4. Truncate to K = ceil(N/2)
  5. Write queue state file (persistent — supports save and return)
  6. Record Accept / Not Accept decisions
  7. Write allocation ledger from accepted decisions

Module review terminates at 3 Accept decisions.
Queue exhausted before 3 accepts — orphan flagged for analyst override.
Traces are fully independent — no cross-ledger exclusions.

Track-aware candidate pool logic:
  T-2 Welding  — WLDG rubric only (excludes PFPB pipefitting track courses)
  T-3 Pipefitting — PFPB all courses + WLDG trunk courses (choose_one=False only)
"""

import math
import pandas as pd
from pathlib import Path
from datetime import datetime
from code.config import PROCESSED_DIR, SCORES_DIR, INTERIM_DIR, TRACES

LEDGER_DIR   = INTERIM_DIR / "ledger"
QUEUE_DIR    = LEDGER_DIR / "queues"
LEDGER_FILE  = LEDGER_DIR / "allocation_ledger.csv"

# ── Gen-ed rubrics excluded from all candidate pools ─────────────────────────

GENED_RUBRICS = {"EDUC", "ENGL", "MATH", "COSC", "BCIS", "ARCE"}

# Specific course exclusions — internships and practicums
EXCLUDE_COURSES = {"WLDG 2489", "WLDG 2488", "CNBT 1266", "ELPT 2264"}

# ── Trace × program scope definitions ────────────────────────────────────────
#
# Keys per program entry:
#   semester_max          — upper semester boundary (99 = all semesters)
#   rubric_include        — if set, only these rubrics are included
#   wldg_trunk_only       — if True, include WLDG only where choose_one=False
#                           (T-3 pipefitting: shared trunk only, not welding track)

TRACE_PROGRAMS = {
    "t0_core": {
        "electrical_aas":      {"semester_max": 2},
        "welding_aas":         {"semester_max": 2},
        "instrumentation_aas": {"semester_max": 2},
        "bct_aas":             {"semester_max": 2},
    },
    "t1_electrical": {
        "electrical_aas": {"semester_max": 99},
    },
    "t2_welding": {
        # Welding track: WLDG courses only — excludes PFPB pipefitting track
        "welding_aas": {
            "semester_max":   99,
            "rubric_include": ["WLDG"],
        },
    },
    "t3_pipefitting": {
        # Pipefitting track: all PFPB + WLDG trunk (choose_one=False) only
        "welding_aas": {
            "semester_max":    99,
            "rubric_include":  ["PFPB", "WLDG"],
            "wldg_trunk_only": True,
        },
    },
    "t4_instrumentation": {
        "instrumentation_aas": {"semester_max": 99},
    },
    "t5_carpentry": {
        "bct_aas": {"semester_max": 99},
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def course_id_to_content_key(course_id: str) -> str:
    """
    Convert degree plan course_id to WECM content_key format.
    'ELPT 1411' → 'ELPT_1_11'
    """
    parts   = course_id.strip().split()
    rubric  = parts[0]
    number  = parts[1]
    level   = number[0]
    content = number[2:]
    return f"{rubric}_{level}_{content}"


def get_candidate_pool(
    program:     str,
    trace_id:    str,
    semester_max: int,
    rubric_include:   list | None = None,
    wldg_trunk_only:  bool = False,
) -> list[str]:
    """
    Return list of course_ids eligible for this trace × program.

    Filters applied in order:
      1. Program match
      2. Semester boundary
      3. Gen-ed rubric exclusion
      4. Specific course exclusion (internships, practicums)
      5. rubric_include — if provided, keep only these rubrics
      6. wldg_trunk_only — if True, drop WLDG courses where choose_one=True
      7. Trace rubric_scope from config (craft traces)
    """
    dp = pd.read_csv(PROCESSED_DIR / "degree_plans_processed.csv")

    pool = dp[
        (dp["program"]      == program) &
        (dp["semester_num"] <= semester_max) &
        (~dp["rubric"].isin(GENED_RUBRICS)) &
        (~dp["course_id"].isin(EXCLUDE_COURSES))
    ].copy()

    # rubric_include filter (track-specific)
    if rubric_include:
        pool = pool[pool["rubric"].isin(rubric_include)]

    # wldg_trunk_only — T-3 pipefitting: keep WLDG only where choose_one=False
    if wldg_trunk_only:
        wldg_mask   = pool["rubric"] == "WLDG"
        trunk_mask  = pool["choose_one"] == False
        non_wldg    = pool[~wldg_mask]
        wldg_trunk  = pool[wldg_mask & trunk_mask]
        pool        = pd.concat([non_wldg, wldg_trunk], ignore_index=True)

    # Trace rubric_scope from config (craft traces — T-1, T-4, T-5)
    rubric_scope = TRACES.get(trace_id, {}).get("rubric_scope")
    if rubric_scope and trace_id not in ("t2_welding", "t3_pipefitting"):
        pool = pool[pool["rubric"].isin(rubric_scope)]

    return pool["course_id"].tolist()


# ── Queue builder ─────────────────────────────────────────────────────────────

def build_module_queue(
    trace_id:             str,
    program:              str,
    module_id:            int | str,
    candidate_course_ids: list[str],
) -> pd.DataFrame:
    """
    For one module × program:
      1. Load SBERT, TF-IDF, Jaccard scores
      2. Filter to candidate pool
      3. Rank all N pairs by SBERT descending
      4. TF-IDF promotion (+1 if TF-IDF rank < current position)
      5. Jaccard promotion (+1 if Jaccard rank < current position, after TF-IDF)
      6. Truncate to K = ceil(N/2)
    """
    content_keys = {course_id_to_content_key(c): c for c in candidate_course_ids}
    key_set      = set(content_keys.keys())

    sbert   = pd.read_csv(SCORES_DIR / f"{trace_id}_sbert.csv")
    tfidf   = pd.read_csv(SCORES_DIR / f"{trace_id}_tfidf.csv")
    jaccard = pd.read_csv(SCORES_DIR / f"{trace_id}_jaccard.csv")

    sbert   = sbert[sbert["module_id"]   == module_id].copy()
    tfidf   = tfidf[tfidf["module_id"]   == module_id].copy()
    jaccard = jaccard[jaccard["module_id"] == module_id].copy()

    sbert   = sbert[sbert["wecm_course_id"].isin(key_set)].copy()
    tfidf   = tfidf[tfidf["wecm_course_id"].isin(key_set)].copy()
    jaccard = jaccard[jaccard["wecm_course_id"].isin(key_set)].copy()

    if sbert.empty:
        return pd.DataFrame()

    N = len(sbert)
    K = math.ceil(N / 2)

    # Re-rank within candidate pool
    sbert   = sbert.sort_values("sbert_score",    ascending=False).reset_index(drop=True)
    tfidf   = tfidf.sort_values("tfidf_score",    ascending=False).reset_index(drop=True)
    jaccard = jaccard.sort_values("jaccard_score", ascending=False).reset_index(drop=True)

    sbert["pool_sbert_rank"]     = range(1, N + 1)
    tfidf["pool_tfidf_rank"]     = range(1, N + 1)
    jaccard["pool_jaccard_rank"] = range(1, N + 1)

    # Merge all three measures
    df = sbert[["module_id", "module_title", "wecm_course_id", "wecm_title",
                "sbert_score", "pool_sbert_rank"]].copy()
    df = df.merge(
        tfidf[["wecm_course_id", "tfidf_score", "pool_tfidf_rank"]],
        on="wecm_course_id", how="left"
    )
    df = df.merge(
        jaccard[["wecm_course_id", "jaccard_score", "pool_jaccard_rank"]],
        on="wecm_course_id", how="left"
    )

    df["pool_tfidf_rank"]   = df["pool_tfidf_rank"].fillna(N + 1).astype(int)
    df["pool_jaccard_rank"] = df["pool_jaccard_rank"].fillna(N + 1).astype(int)

    # ── Promotion logic ───────────────────────────────────────────────────────
    df = df.sort_values("pool_sbert_rank").reset_index(drop=True)
    df["queue_pos"]       = range(1, N + 1)
    df["promotion_delta"] = 0

    # Step 1 — TF-IDF promotion
    for idx in df.index:
        current_pos = df.at[idx, "queue_pos"]
        tfidf_rank  = df.at[idx, "pool_tfidf_rank"]
        if tfidf_rank < current_pos and current_pos > 1:
            swap_idx = df[df["queue_pos"] == current_pos - 1].index
            if len(swap_idx) > 0:
                df.at[swap_idx[0], "queue_pos"] = current_pos
                df.at[idx, "queue_pos"]         = current_pos - 1
                df.at[idx, "promotion_delta"]  += 1

    # Step 2 — Jaccard promotion (after TF-IDF)
    df = df.sort_values("queue_pos").reset_index(drop=True)
    for idx in df.index:
        current_pos  = df.at[idx, "queue_pos"]
        jaccard_rank = df.at[idx, "pool_jaccard_rank"]
        if jaccard_rank < current_pos and current_pos > 1:
            swap_idx = df[df["queue_pos"] == current_pos - 1].index
            if len(swap_idx) > 0:
                df.at[swap_idx[0], "queue_pos"] = current_pos
                df.at[idx, "queue_pos"]         = current_pos - 1
                df.at[idx, "promotion_delta"]  += 1

    # ── Truncate to K ─────────────────────────────────────────────────────────
    df = df[df["queue_pos"] <= K].sort_values("queue_pos").reset_index(drop=True)

    df.insert(0, "trace_id", trace_id)
    df.insert(1, "program",  program)
    df["N"]             = N
    df["K"]             = K
    df["decision"]      = "PENDING"
    df["analyst"]       = ""
    df["decision_date"] = ""
    df["rationale"]     = ""

    return df


def build_trace_queue(trace_id: str) -> pd.DataFrame:
    """Build complete queue for all modules × programs in a trace."""
    programs = TRACE_PROGRAMS.get(trace_id, {})
    if not programs:
        print(f"  No programs defined for {trace_id} — skipping.")
        return pd.DataFrame()

    sbert   = pd.read_csv(SCORES_DIR / f"{trace_id}_sbert.csv")
    modules = sbert[["module_id", "module_title"]].drop_duplicates()

    all_queues = []

    for program, scope in programs.items():
        print(f"  Program: {program}")

        candidate_ids = get_candidate_pool(
            program          = program,
            trace_id         = trace_id,
            semester_max     = scope.get("semester_max", 99),
            rubric_include   = scope.get("rubric_include"),
            wldg_trunk_only  = scope.get("wldg_trunk_only", False),
        )
        print(f"    Candidate pool: {len(candidate_ids)} courses")
        for c in candidate_ids:
            print(f"      {c}")

        modules_queued = 0
        for _, mod_row in modules.iterrows():
            queue = build_module_queue(
                trace_id, program, mod_row["module_id"], candidate_ids
            )
            if not queue.empty:
                all_queues.append(queue)
                modules_queued += 1

        print(f"    Modules queued: {modules_queued}")

    if not all_queues:
        return pd.DataFrame()

    return pd.concat(all_queues, ignore_index=True)


def save_queue(trace_id: str, df: pd.DataFrame):
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    path = QUEUE_DIR / f"{trace_id}_queue.csv"
    df.to_csv(path, index=False)
    print(f"  Queue saved: {path.name} — {len(df):,} pairs")


def load_queue(trace_id: str) -> pd.DataFrame:
    path = QUEUE_DIR / f"{trace_id}_queue.csv"
    if not path.exists():
        raise FileNotFoundError(f"Queue not found: {path}")
    return pd.read_csv(path)


# ── Decision recording ────────────────────────────────────────────────────────

def record_decision(
    trace_id:       str,
    program:        str,
    module_id:      int | str,
    wecm_course_id: str,
    decision:       str,
    analyst:        str,
    rationale:      str = "",
):
    """
    Write a single decision to the queue state file.
    Enforces max 3 accepts per module × program.
    """
    assert decision in ("ACCEPT", "NOT_ACCEPT"), \
        "Decision must be ACCEPT or NOT_ACCEPT"

    df   = load_queue(trace_id)
    mask = (
        (df["program"]        == program) &
        (df["module_id"]      == module_id) &
        (df["wecm_course_id"] == wecm_course_id)
    )

    if mask.sum() == 0:
        raise ValueError(
            f"Pair not found in queue: {program} / {module_id} / {wecm_course_id}"
        )

    if decision == "ACCEPT":
        existing = df[
            (df["program"]   == program) &
            (df["module_id"] == module_id) &
            (df["decision"]  == "ACCEPT")
        ]
        if len(existing) >= 3:
            raise ValueError(
                f"Module {module_id} in {program} already has 3 Accept decisions. "
                f"Module review is closed."
            )

    df.loc[mask, "decision"]      = decision
    df.loc[mask, "analyst"]       = analyst
    df.loc[mask, "decision_date"] = datetime.now().strftime("%Y-%m-%d")
    df.loc[mask, "rationale"]     = rationale

    df.to_csv(QUEUE_DIR / f"{trace_id}_queue.csv", index=False)
    print(f"  Recorded: {decision} — {program} / module {module_id} / {wecm_course_id}")


# ── Ledger compiler ───────────────────────────────────────────────────────────

def build_ledger():
    """
    Compile all ACCEPT decisions across all trace queues into the
    master allocation ledger. Flag orphans where K queue exhausted
    before 3 accepts.
    """
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    all_accepts = []
    orphans     = []

    for trace_id in TRACE_PROGRAMS:
        queue_path = QUEUE_DIR / f"{trace_id}_queue.csv"
        if not queue_path.exists():
            continue

        df      = pd.read_csv(queue_path)
        accepts = df[df["decision"] == "ACCEPT"].copy()
        if not accepts.empty:
            all_accepts.append(accepts)

        for (program, module_id), grp in df.groupby(["program", "module_id"]):
            n_accepts = (grp["decision"] == "ACCEPT").sum()
            n_pending = (grp["decision"] == "PENDING").sum()
            if n_accepts < 3 and n_pending == 0:
                orphans.append({
                    "trace_id":         trace_id,
                    "program":          program,
                    "module_id":        module_id,
                    "module_title":     grp["module_title"].iloc[0],
                    "accepts_recorded": n_accepts,
                    "status":           "ORPHAN — analyst override required",
                })

    if all_accepts:
        ledger_df  = pd.concat(all_accepts, ignore_index=True)
        ledger_cols = [
            "trace_id", "program", "module_id", "module_title",
            "wecm_course_id", "wecm_title",
            "sbert_score", "pool_sbert_rank",
            "tfidf_score", "pool_tfidf_rank",
            "jaccard_score", "pool_jaccard_rank",
            "promotion_delta", "queue_pos", "N", "K",
            "analyst", "decision_date", "rationale",
        ]
        ledger_df = ledger_df[[c for c in ledger_cols if c in ledger_df.columns]]
        ledger_df.to_csv(LEDGER_FILE, index=False)
        print(f"Ledger written: {LEDGER_FILE.name} — {len(ledger_df)} allocations")
    else:
        print("No Accept decisions recorded yet — ledger is empty.")

    if orphans:
        orphan_df   = pd.DataFrame(orphans)
        orphan_path = LEDGER_DIR / "orphans.csv"
        orphan_df.to_csv(orphan_path, index=False)
        print(f"Orphans: {orphan_path.name} — {len(orphans)} modules require override")
    else:
        print("No orphans detected.")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_all_queues():
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    for trace_id in TRACE_PROGRAMS:
        print(f"\n=== {trace_id.upper()} ===")
        df = build_trace_queue(trace_id)
        if not df.empty:
            save_queue(trace_id, df)
    print("\n=== QUEUE BUILD COMPLETE ===")


if __name__ == "__main__":
    build_all_queues()