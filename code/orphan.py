"""
orphan.py — Orphan Recycle Protocol
Antikythera Pipeline | LSCO–ABC Alignment Project

An orphan is any module with zero Accept decisions after primary K queue
exhaustion. This module:

  1. Identifies orphaned modules from primary queue state files
  2. Builds a secondary queue — top 3 SBERT pairs from full N per orphan
  3. Records single-reviewer decisions (one pass only, no recycle)
  4. Finalizes unresolved orphans as unallocated in the gap analysis
  5. Feeds accepted decisions into the master allocation ledger via
     build_ledger() — flagged as ORPHAN_RECYCLE in allocation_source

Architecture:
  score files → orphan queue → single reviewer decisions → ledger compiler
  Orphan queue files: interim/ledger/queues/{trace_id}_orphan_queue.csv
  Same column structure as primary queue + allocation_source column.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from code.config import PROCESSED_DIR, SCORES_DIR, INTERIM_DIR
from code.ledger import (
    QUEUE_DIR,
    LEDGER_DIR,
    LEDGER_FILE,
    TRACE_PROGRAMS,
    get_candidate_pool,
    course_id_to_content_key,
)

ORPHAN_QUEUE_SUFFIX = "_orphan_queue.csv"
GAP_FILE            = LEDGER_DIR / "gap_analysis.csv"

ORPHAN_K = 3  # Fixed — top 3 SBERT from full N, one pass only


# ── Identify orphans from primary queue ───────────────────────────────────────

def get_orphaned_modules(trace_id: str) -> pd.DataFrame:
    """
    Return DataFrame of modules with zero Accept decisions after primary
    K queue exhaustion — i.e. no PENDING rows remain and accepts == 0.
    """
    queue_path = QUEUE_DIR / f"{trace_id}_queue.csv"
    if not queue_path.exists():
        raise FileNotFoundError(f"Primary queue not found: {queue_path}")

    df = pd.read_csv(queue_path)

    orphans = []
    for (program, module_id), grp in df.groupby(["program", "module_id"]):
        n_accepts = (grp["decision"] == "ACCEPT").sum()
        n_pending = (grp["decision"] == "PENDING").sum()
        if n_accepts == 0 and n_pending == 0:
            orphans.append({
                "trace_id":    trace_id,
                "program":     program,
                "module_id":   module_id,
                "module_title": grp["module_title"].iloc[0],
            })

    return pd.DataFrame(orphans)


# ── Build orphan recycle queue ────────────────────────────────────────────────

def build_orphan_queue(trace_id: str) -> pd.DataFrame:
    """
    For each orphaned module × program:
      - Pull full N SBERT-ranked pairs from score file
      - Filter to candidate pool
      - Take top 3 by SBERT score (no reordering — clean signal only)
      - Add all three signal scores for committee transparency
    Returns combined orphan queue DataFrame.
    """
    orphans = get_orphaned_modules(trace_id)

    if orphans.empty:
        print(f"  No orphans found for {trace_id}.")
        return pd.DataFrame()

    print(f"  Orphaned modules: {len(orphans)}")

    sbert   = pd.read_csv(SCORES_DIR / f"{trace_id}_sbert.csv")
    tfidf   = pd.read_csv(SCORES_DIR / f"{trace_id}_tfidf.csv")
    jaccard = pd.read_csv(SCORES_DIR / f"{trace_id}_jaccard.csv")

    all_rows = []

    for _, orphan in orphans.iterrows():
        program   = orphan["program"]
        module_id = orphan["module_id"]

        # Get candidate pool for this program × trace
        scope         = TRACE_PROGRAMS.get(trace_id, {}).get(program, {})
        candidate_ids = get_candidate_pool(
            program         = program,
            trace_id        = trace_id,
            semester_max    = scope.get("semester_max", 99),
            rubric_include  = scope.get("rubric_include"),
            wldg_trunk_only = scope.get("wldg_trunk_only", False),
        )

        content_keys = {course_id_to_content_key(c): c for c in candidate_ids}
        key_set      = set(content_keys.keys())

        # Filter score files to this module × candidate pool
        mod_sbert   = sbert[(sbert["module_id"] == module_id) &
                             (sbert["wecm_course_id"].isin(key_set))].copy()
        mod_tfidf   = tfidf[(tfidf["module_id"] == module_id) &
                             (tfidf["wecm_course_id"].isin(key_set))].copy()
        mod_jaccard = jaccard[(jaccard["module_id"] == module_id) &
                               (jaccard["wecm_course_id"].isin(key_set))].copy()

        if mod_sbert.empty:
            print(f"    WARNING: No score pairs found for module {module_id} "
                  f"in {program} — skipping.")
            continue

        N = len(mod_sbert)

        # Re-rank within pool
        mod_sbert   = mod_sbert.sort_values("sbert_score",
                                             ascending=False).reset_index(drop=True)
        mod_tfidf   = mod_tfidf.sort_values("tfidf_score",
                                             ascending=False).reset_index(drop=True)
        mod_jaccard = mod_jaccard.sort_values("jaccard_score",
                                               ascending=False).reset_index(drop=True)

        mod_sbert["pool_sbert_rank"]     = range(1, N + 1)
        mod_tfidf["pool_tfidf_rank"]     = range(1, N + 1)
        mod_jaccard["pool_jaccard_rank"] = range(1, N + 1)

        # Take top ORPHAN_K by SBERT — no promotion reordering
        top_k = mod_sbert.head(ORPHAN_K).copy()

        top_k = top_k.merge(
            mod_tfidf[["wecm_course_id", "tfidf_score", "pool_tfidf_rank"]],
            on="wecm_course_id", how="left"
        )
        top_k = top_k.merge(
            mod_jaccard[["wecm_course_id", "jaccard_score", "pool_jaccard_rank"]],
            on="wecm_course_id", how="left"
        )

        top_k["pool_tfidf_rank"]   = top_k["pool_tfidf_rank"].fillna(N + 1).astype(int)
        top_k["pool_jaccard_rank"] = top_k["pool_jaccard_rank"].fillna(N + 1).astype(int)
        top_k["queue_pos"]         = range(1, len(top_k) + 1)
        top_k["promotion_delta"]   = 0  # No promotion on orphan pass
        top_k["N"]                 = N
        top_k["K"]                 = ORPHAN_K
        top_k["trace_id"]          = trace_id
        top_k["program"]           = program
        top_k["decision"]          = "PENDING"
        top_k["analyst"]           = ""
        top_k["decision_date"]     = ""
        top_k["rationale"]         = ""
        top_k["allocation_source"] = "ORPHAN_RECYCLE"

        all_rows.append(top_k)
        print(f"    Module {module_id} ({orphan['module_title'][:40]}) "
              f"— {len(top_k)} recycle pairs")

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)

    # Reorder columns to match primary queue + allocation_source
    col_order = [
        "trace_id", "program", "module_id", "module_title",
        "wecm_course_id", "wecm_title",
        "sbert_score", "pool_sbert_rank",
        "tfidf_score", "pool_tfidf_rank",
        "jaccard_score", "pool_jaccard_rank",
        "promotion_delta", "queue_pos", "N", "K",
        "decision", "analyst", "decision_date", "rationale",
        "allocation_source",
    ]
    result = result[[c for c in col_order if c in result.columns]]
    return result


def save_orphan_queue(trace_id: str, df: pd.DataFrame):
    path = QUEUE_DIR / f"{trace_id}{ORPHAN_QUEUE_SUFFIX}"
    df.to_csv(path, index=False)
    print(f"  Orphan queue saved: {path.name} — {len(df)} pairs")


def load_orphan_queue(trace_id: str) -> pd.DataFrame:
    path = QUEUE_DIR / f"{trace_id}{ORPHAN_QUEUE_SUFFIX}"
    if not path.exists():
        raise FileNotFoundError(f"Orphan queue not found: {path}")
    return pd.read_csv(path)


# ── Decision recording ────────────────────────────────────────────────────────

def record_orphan_decision(
    trace_id:       str,
    program:        str,
    module_id:      int | str,
    wecm_course_id: str,
    decision:       str,
    analyst:        str,
    rationale:      str = "",
):
    """
    Record a single Accept / Not Accept decision on the orphan queue.
    One pass only — no further recycle after this.
    """
    assert decision in ("ACCEPT", "NOT_ACCEPT"), \
        "Decision must be ACCEPT or NOT_ACCEPT"

    df   = load_orphan_queue(trace_id)
    mask = (
        (df["program"]        == program) &
        (df["module_id"]      == module_id) &
        (df["wecm_course_id"] == wecm_course_id)
    )

    if mask.sum() == 0:
        raise ValueError(
            f"Pair not found in orphan queue: "
            f"{program} / {module_id} / {wecm_course_id}"
        )

    # One accept per module on orphan pass is sufficient — no 3-accept ceiling
    df.loc[mask, "decision"]      = decision
    df.loc[mask, "analyst"]       = analyst
    df.loc[mask, "decision_date"] = datetime.now().strftime("%Y-%m-%d")
    df.loc[mask, "rationale"]     = rationale

    df.to_csv(QUEUE_DIR / f"{trace_id}{ORPHAN_QUEUE_SUFFIX}", index=False)
    print(f"  Orphan decision recorded: {decision} — "
          f"{program} / module {module_id} / {wecm_course_id}")


# ── Gap analysis finalizer ────────────────────────────────────────────────────

def finalize_gaps():
    """
    After orphan review is complete, identify modules still at zero accepts
    across both primary and orphan queues. Write to gap_analysis.csv.
    These are finalized as unallocated — no further recycle.
    """
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    gaps = []

    for trace_id in TRACE_PROGRAMS:
        # Check primary queue
        primary_path = QUEUE_DIR / f"{trace_id}_queue.csv"
        orphan_path  = QUEUE_DIR / f"{trace_id}{ORPHAN_QUEUE_SUFFIX}"

        if not primary_path.exists():
            continue

        primary = pd.read_csv(primary_path)

        # Build set of modules with at least one Accept in primary queue
        primary_accepts = set(
            primary[primary["decision"] == "ACCEPT"]
            .apply(lambda r: (r["program"], r["module_id"]), axis=1)
        )

        # Add orphan queue accepts if it exists
        orphan_accepts = set()
        if orphan_path.exists():
            orphan = pd.read_csv(orphan_path)
            orphan_accepts = set(
                orphan[orphan["decision"] == "ACCEPT"]
                .apply(lambda r: (r["program"], r["module_id"]), axis=1)
            )

        all_accepts = primary_accepts | orphan_accepts

        # Any module × program with no pending rows in primary and not in accepts
        for (program, module_id), grp in primary.groupby(["program", "module_id"]):
            n_pending = (grp["decision"] == "PENDING").sum()
            if n_pending == 0 and (program, module_id) not in all_accepts:
                gaps.append({
                    "trace_id":    trace_id,
                    "program":     program,
                    "module_id":   module_id,
                    "module_title": grp["module_title"].iloc[0],
                    "status":      "UNALLOCATED — no placement after recycle",
                })

    if gaps:
        gap_df = pd.DataFrame(gaps)
        gap_df.to_csv(GAP_FILE, index=False)
        print(f"Gap analysis written: {GAP_FILE.name} — {len(gaps)} unallocated modules")
    else:
        print("No gaps — all modules allocated.")

    return pd.DataFrame(gaps) if gaps else pd.DataFrame()


# ── Ledger contribution ───────────────────────────────────────────────────────

def build_orphan_ledger_contribution() -> pd.DataFrame:
    """
    Collect all ACCEPT decisions from orphan queues across all traces.
    Returns DataFrame ready to concatenate into the master allocation ledger.
    allocation_source = ORPHAN_RECYCLE on all rows.
    """
    contributions = []

    for trace_id in TRACE_PROGRAMS:
        orphan_path = QUEUE_DIR / f"{trace_id}{ORPHAN_QUEUE_SUFFIX}"
        if not orphan_path.exists():
            continue
        df      = pd.read_csv(orphan_path)
        accepts = df[df["decision"] == "ACCEPT"].copy()
        if not accepts.empty:
            contributions.append(accepts)

    if not contributions:
        return pd.DataFrame()

    return pd.concat(contributions, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_all_orphan_queues():
    """Build and save orphan recycle queues for all traces."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    for trace_id in TRACE_PROGRAMS:
        print(f"\n=== {trace_id.upper()} ORPHAN CHECK ===")
        queue_path = QUEUE_DIR / f"{trace_id}_queue.csv"
        if not queue_path.exists():
            print(f"  Primary queue not found — skipping.")
            continue

        df = build_orphan_queue(trace_id)
        if not df.empty:
            save_orphan_queue(trace_id, df)
        else:
            print(f"  No orphan queue generated.")

    print("\n=== ORPHAN QUEUE BUILD COMPLETE ===")


if __name__ == "__main__":
    build_all_orphan_queues()