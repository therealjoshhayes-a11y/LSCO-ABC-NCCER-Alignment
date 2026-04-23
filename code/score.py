"""
score.py — Similarity Scoring Layer
Antikythera Pipeline | LSCO–ABC Alignment Project

For each trace, computes three independent similarity measures:
  1. SBERT cosine similarity (from pre-computed L2-normalized embeddings)
  2. TF-IDF cosine similarity (ACGM + WECM IDF corpus, post-hoc audit signal)
  3. Jaccard coefficient (stopword removal, Porter stemming, no lemmatization)

Outputs one ranked CSV per measure per trace to interim/scores/.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from code.config import PROCESSED_DIR, EMBEDDINGS_DIR, SCORES_DIR

TRACES = [
    "t0_core",
    "t1_electrical",
    "t2_welding",
    "t3_pipefitting",
    "t4_instrumentation",
    "t5_carpentry",
    "t6_rigging",
    "t7_scaffold",
]

# ── Text preprocessing ───────────────────────────────────────────────────────

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_jaccard(text: str) -> set:
    """Lowercase, remove stopwords, Porter stem. No lemmatization."""
    tokens = str(text).lower().split()
    return {stemmer.stem(t) for t in tokens if t not in STOP_WORDS}


# ── SBERT scoring ────────────────────────────────────────────────────────────

def score_sbert(trace_id: str) -> pd.DataFrame:
    wecm_vecs = np.load(EMBEDDINGS_DIR / "wecm_embeddings.npy")
    trace_vecs = np.load(EMBEDDINGS_DIR / f"{trace_id}_embeddings.npy")
    wecm_idx = pd.read_csv(EMBEDDINGS_DIR / "wecm_index.csv")
    trace_idx = pd.read_csv(EMBEDDINGS_DIR / f"{trace_id}_index.csv")

    # Cosine similarity — vectors already L2-normalized, so dot product = cosine
    sim_matrix = np.dot(trace_vecs, wecm_vecs.T)  # (n_modules, n_wecm)

    records = []
    for i, mod_row in trace_idx.iterrows():
        for j, wecm_row in wecm_idx.iterrows():
            records.append({
                "trace_id":        trace_id,
                "module_id":       mod_row.get("module_id", mod_row.get("module_number", i)),
                "module_title":    mod_row.get("module_title", ""),
                "wecm_course_id":  wecm_row.get("course_id", ""),
                "wecm_title":      wecm_row.get("course_title", ""),
                "sbert_score":     float(sim_matrix[i, j]),
            })

    df = pd.DataFrame(records)
    df = df.sort_values("sbert_score", ascending=False).reset_index(drop=True)
    df["sbert_rank"] = df.index + 1
    return df


# ── TF-IDF scoring ───────────────────────────────────────────────────────────

def build_idf_corpus() -> list[str]:
    """Combine WECM and ACGM claims blocks for IDF corpus construction."""
    wecm = pd.read_csv(PROCESSED_DIR / "wecm_processed.csv")
    acgm = pd.read_csv(PROCESSED_DIR / "acgm_processed.csv")
    corpus = (
        wecm["claims_block"].fillna("").tolist() +
        acgm["corpus_string"].fillna("").tolist()
    )
    return corpus


def score_tfidf(trace_id: str, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    wecm = pd.read_csv(PROCESSED_DIR / "wecm_processed.csv")
    trace = pd.read_csv(EMBEDDINGS_DIR / f"{trace_id}_index.csv")

    wecm_texts  = wecm["claims_block"].fillna("").tolist()
    trace_texts = trace["claims_block"].fillna("").tolist()

    wecm_vecs  = vectorizer.transform(wecm_texts)
    trace_vecs = vectorizer.transform(trace_texts)

    sim_matrix = cosine_similarity(trace_vecs, wecm_vecs)  # (n_modules, n_wecm)

    records = []
    for i, mod_row in trace.iterrows():
        for j, wecm_row in wecm.iterrows():
            records.append({
                "trace_id":       trace_id,
                "module_id":      mod_row.get("module_id", mod_row.get("module_number", i)),
                "module_title":   mod_row.get("module_title", ""),
                "wecm_course_id": wecm_row.get("course_id", ""),
                "wecm_title":     wecm_row.get("course_title", ""),
                "tfidf_score":    float(sim_matrix[i, j]),
            })

    df = pd.DataFrame(records)
    df = df.sort_values("tfidf_score", ascending=False).reset_index(drop=True)
    df["tfidf_rank"] = df.index + 1
    return df


# ── Jaccard scoring ──────────────────────────────────────────────────────────

def score_jaccard(trace_id: str) -> pd.DataFrame:
    wecm  = pd.read_csv(PROCESSED_DIR / "wecm_processed.csv")
    trace = pd.read_csv(EMBEDDINGS_DIR / f"{trace_id}_index.csv")

    wecm_sets  = [preprocess_jaccard(t) for t in wecm["claims_block"].fillna("")]
    trace_sets = [preprocess_jaccard(t) for t in trace["claims_block"].fillna("")]

    records = []
    for i, mod_row in trace.iterrows():
        for j, wecm_row in wecm.iterrows():
            a, b = trace_sets[i], wecm_sets[j]
            intersection = len(a & b)
            union = len(a | b)
            jaccard = intersection / union if union > 0 else 0.0
            records.append({
                "trace_id":       trace_id,
                "module_id":      mod_row.get("module_id", mod_row.get("module_number", i)),
                "module_title":   mod_row.get("module_title", ""),
                "wecm_course_id": wecm_row.get("course_id", ""),
                "wecm_title":     wecm_row.get("course_title", ""),
                "jaccard_score":  jaccard,
            })

    df = pd.DataFrame(records)
    df = df.sort_values("jaccard_score", ascending=False).reset_index(drop=True)
    df["jaccard_rank"] = df.index + 1
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def run_scoring():
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    # Build TF-IDF vectorizer once from full IDF corpus
    print("Building TF-IDF vectorizer from WECM + ACGM corpus...")
    corpus = build_idf_corpus()
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(corpus)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}\n")

    for trace_id in TRACES:
        print(f"=== {trace_id.upper()} ===")

        print("  SBERT...")
        sbert_df = score_sbert(trace_id)
        sbert_df.to_csv(SCORES_DIR / f"{trace_id}_sbert.csv", index=False)
        print(f"  Saved {trace_id}_sbert.csv — {len(sbert_df):,} pairs")

        print("  TF-IDF...")
        tfidf_df = score_tfidf(trace_id, vectorizer)
        tfidf_df.to_csv(SCORES_DIR / f"{trace_id}_tfidf.csv", index=False)
        print(f"  Saved {trace_id}_tfidf.csv — {len(tfidf_df):,} pairs")

        print("  Jaccard...")
        jaccard_df = score_jaccard(trace_id)
        jaccard_df.to_csv(SCORES_DIR / f"{trace_id}_jaccard.csv", index=False)
        print(f"  Saved {trace_id}_jaccard.csv — {len(jaccard_df):,} pairs\n")

    print("=== SCORING COMPLETE ===")
    print(f"All files written to {SCORES_DIR}")


if __name__ == "__main__":
    run_scoring()