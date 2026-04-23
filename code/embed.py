"""
embed.py — SBERT Embedding Layer
Antikythera Pipeline | LSCO–ABC Alignment Project

Loads all-mpnet-base-v2, embeds WECM content units and all NCCER trace
claims blocks, L2-normalizes all vectors, serializes to interim/embeddings/.
WECM matrix is computed once and reused across all eight traces.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from code.config import PROCESSED_DIR, EMBEDDINGS_DIR

# ── Trace file map ──────────────────────────────────────────────────────────
TRACE_FILES = {
    "t0_core":            "t0_core_processed.csv",
    "t1_electrical":      "t1_electrical_processed.csv",
    "t2_welding":         ["t2_welding_processed.csv", "t2_welding_cel_processed.csv"],
    "t3_pipefitting":     "t3_pipefitting_processed.csv",
    "t4_instrumentation": "t4_instrumentation_processed.csv",
    "t5_carpentry":       "t5_carpentry_processed.csv",
    "t6_rigging":         "rigger_processed.csv",
    "t7_scaffold":        "t7_scaffold_processed.csv",
}


def load_trace(trace_id: str) -> pd.DataFrame:
    """Load one or more processed CSVs for a trace and return combined DataFrame."""
    entry = TRACE_FILES[trace_id]
    if isinstance(entry, list):
        frames = [pd.read_csv(PROCESSED_DIR / f) for f in entry]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(PROCESSED_DIR / entry)
    return df


def embed_texts(model: SentenceTransformer, texts: list[str], label: str) -> np.ndarray:
    """Encode a list of strings, L2-normalize, return float32 matrix."""
    print(f"  Embedding {len(texts)} records — {label}...")
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    return vectors.astype(np.float32)


def run_embeddings():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all-mpnet-base-v2...")
    model = SentenceTransformer("all-mpnet-base-v2")
    print("Model loaded.\n")

    # ── WECM (internal catalog) ─────────────────────────────────────────────
    print("=== WECM ===")
    wecm = pd.read_csv(PROCESSED_DIR / "wecm_processed.csv")
    wecm_texts = wecm["claims_block"].fillna("").tolist()
    wecm_vectors = embed_texts(model, wecm_texts, "WECM")
    np.save(EMBEDDINGS_DIR / "wecm_embeddings.npy", wecm_vectors)
    wecm["embed_index"] = range(len(wecm))
    wecm.to_csv(EMBEDDINGS_DIR / "wecm_index.csv", index=False)
    print(f"  Saved wecm_embeddings.npy — shape: {wecm_vectors.shape}\n")

    # ── NCCER traces ────────────────────────────────────────────────────────
    for trace_id in TRACE_FILES:
        print(f"=== {trace_id.upper()} ===")
        df = load_trace(trace_id)
        texts = df["claims_block"].fillna("").tolist()
        vectors = embed_texts(model, texts, trace_id)
        np.save(EMBEDDINGS_DIR / f"{trace_id}_embeddings.npy", vectors)
        df["embed_index"] = range(len(df))
        df.to_csv(EMBEDDINGS_DIR / f"{trace_id}_index.csv", index=False)
        print(f"  Saved {trace_id}_embeddings.npy — shape: {vectors.shape}\n")

    print("=== EMBEDDING COMPLETE ===")
    print(f"All files written to {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    run_embeddings()