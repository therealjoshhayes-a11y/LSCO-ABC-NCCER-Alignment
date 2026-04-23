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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from code.config import (
    PROCESSED_DIR, EMBEDDINGS_DIR, SCORES_DIR,
    TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_NGRAM_RANGE,
)

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

# ── Shared preprocessing resources ──────────────────────────────────────────

STOP_WORDS = set(stopwords.words("english"))
stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# ── Penn Treebank → WordNet POS mapping ─────────────────────────────────────

def penn_to_wn(tag: str) -> str | None:
    """Map Penn Treebank POS tag to WordNet POS constant. Returns None if not verb."""
    if tag.startswith("VB"):
        return wordnet.VERB
    return None


# ── TF-IDF preprocessing ─────────────────────────────────────────────────────

def preprocess_tfidf(text: str) -> str:
    """
    MinK v3.2 TF-IDF preprocessing pipeline:
      lowercase → stop word removal → lemmatize with POS tagging →
      verb-only WordNet synonym expansion → return token string for vectorizer.
    Vectorizer receives pre-cleaned token stream; stop_words=None on vectorizer.
    """
    tokens = word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]

    tagged = pos_tag(tokens)

    expanded = []
    for token, tag in tagged:
        wn_pos = penn_to_wn(tag)
        if wn_pos == wordnet.VERB:
            lemma = lemmatizer.lemmatize(token, pos=wordnet.VERB)
            expanded.append(lemma)
            # Verb-only WordNet synonym expansion
            for synset in wordnet.synsets(lemma, pos=wordnet.VERB):
                for syn_lemma in synset.lemmas():
                    synonym = syn_lemma.name().replace("_", " ")
                    if synonym != lemma:
                        expanded.append(synonym)
        else:
            # Non-verbs: lemmatize as noun, no synonym expansion
            expanded.append(lemmatizer.lemmatize(token))

    return " ".join(expanded)


# ── Jaccard preprocessing ────────────────────────────────────────────────────

def preprocess_jaccard(text: str) -> set:
    """Lowercase, remove stopwords, Porter stem. No lemmatization per MinK v3.2."""
    tokens = str(text).lower().split()
    return {stemmer.stem(t) for t in tokens if t not in STOP_WORDS}


# ── SBERT scoring ────────────────────────────────────────────────────────────

def score_sbert(trace_id: str) -> pd.DataFrame:
    wecm_vecs  = np.load(EMBEDDINGS_DIR / "wecm_embeddings.npy")
    trace_vecs = np.load(EMBEDDINGS_DIR / f"{trace_id}_embeddings.npy")
    wecm_idx   = pd.read_csv(EMBEDDINGS_DIR / "wecm_index.csv")
    trace_idx  = pd.read_csv(EMBEDDINGS_DIR / f"{trace_id}_index.csv")

    # Vectors are L2-normalized — dot product equals cosine similarity
    sim_matrix = np.dot(trace_vecs, wecm_vecs.T)  # (n_modules, n_wecm)

    records = []
    for i, mod_row in trace_idx.iterrows():
        for j, wecm_row in wecm_idx.iterrows():
            records.append({
                "trace_id":       trace_id,
                "module_id":      mod_row.get("module_id", mod_row.get("module_number", i)),
                "module_title":   mod_row.get("module_title", ""),
                "wecm_course_id": wecm_row.get("content_key", ""),
                "wecm_title":     wecm_row.get("title", ""),
                "sbert_score":    float(sim_matrix[i, j]),
            })

    df = pd.DataFrame(records)
    df = df.sort_values("sbert_score", ascending=False).reset_index(drop=True)
    df["sbert_rank"] = df.index + 1
    return df


# ── TF-IDF scoring ───────────────────────────────────────────────────────────

def build_idf_corpus() -> list[str]:
    """
    Combine WECM and ACGM text for IDF corpus. Apply TF-IDF preprocessing
    before fitting so IDF weights reflect the cleaned token stream.
    """
    wecm = pd.read_csv(PROCESSED_DIR / "wecm_processed.csv")
    acgm = pd.read_csv(PROCESSED_DIR / "acgm_processed.csv")
    raw  = (
        wecm["claims_block"].fillna("").tolist() +
        acgm["corpus_string"].fillna("").tolist()
    )
    print(f"  Preprocessing {len(raw):,} corpus documents for TF-IDF...")
    return [preprocess_tfidf(t) for t in raw]


def score_tfidf(trace_id: str, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    wecm  = pd.read_csv(EMBEDDINGS_DIR / "wecm_index.csv")
    trace = pd.read_csv(EMBEDDINGS_DIR / f"{trace_id}_index.csv")

    # Preprocess before transforming — same pipeline as corpus fit
    wecm_texts  = [preprocess_tfidf(t) for t in wecm["claims_block"].fillna("")]
    trace_texts = [preprocess_tfidf(t) for t in trace["claims_block"].fillna("")]

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
                "wecm_course_id": wecm_row.get("content_key", ""),
                "wecm_title":     wecm_row.get("title", ""),
                "tfidf_score":    float(sim_matrix[i, j]),
            })

    df = pd.DataFrame(records)
    df = df.sort_values("tfidf_score", ascending=False).reset_index(drop=True)
    df["tfidf_rank"] = df.index + 1
    return df


# ── Jaccard scoring ──────────────────────────────────────────────────────────

def score_jaccard(trace_id: str) -> pd.DataFrame:
    wecm  = pd.read_csv(EMBEDDINGS_DIR / "wecm_index.csv")
    trace = pd.read_csv(EMBEDDINGS_DIR / f"{trace_id}_index.csv")

    wecm_sets  = [preprocess_jaccard(t) for t in wecm["claims_block"].fillna("")]
    trace_sets = [preprocess_jaccard(t) for t in trace["claims_block"].fillna("")]

    records = []
    for i, mod_row in trace.iterrows():
        for j, wecm_row in wecm.iterrows():
            a, b = trace_sets[i], wecm_sets[j]
            intersection = len(a & b)
            union        = len(a | b)
            jaccard      = intersection / union if union > 0 else 0.0
            records.append({
                "trace_id":       trace_id,
                "module_id":      mod_row.get("module_id", mod_row.get("module_number", i)),
                "module_title":   mod_row.get("module_title", ""),
                "wecm_course_id": wecm_row.get("content_key", ""),
                "wecm_title":     wecm_row.get("title", ""),
                "jaccard_score":  jaccard,
            })

    df = pd.DataFrame(records)
    df = df.sort_values("jaccard_score", ascending=False).reset_index(drop=True)
    df["jaccard_rank"] = df.index + 1
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def run_scoring():
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    # Build TF-IDF vectorizer once from preprocessed full IDF corpus
    print("Building TF-IDF vectorizer from WECM + ACGM corpus...")
    corpus = build_idf_corpus()
    vectorizer = TfidfVectorizer(
        stop_words=None,                  # Stop words removed in preprocess_tfidf()
        max_features=TFIDF_MAX_FEATURES,  # 5000
        min_df=TFIDF_MIN_DF,              # 2
        ngram_range=TFIDF_NGRAM_RANGE,    # (1, 3)
    )
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