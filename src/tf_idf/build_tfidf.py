"""
Build TF-IDF sparse matrix + vectorizer; save to disk; also export doc index -> (id, title) mapping.

dependency:
  pip install scikit-learn scipy joblib tqdm

Input:
  A JSONL file. Each line is one paper with field "processed_content" (title+abstract, already cleaned).

Output:
  - tfidf_matrix.npz: scipy csr_matrix
  - tfidf_vectorizer.joblib: Trained TfidfVectorizer, use for future transform/query
  - doc_ids.npy / doc_titles.npy: Arrays of IDs / titles corresponding to each row of the matrix
  - Terminal output: Scale information, non-zero feature statistics for the first few documents
"""

import json
import os
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib
from pathlib import Path
#Config

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_JSONL = ROOT_DIR / "data" / "preprocess" / "arxiv-cs-data-with-citations-final-dataset_preprocessed.json"
OUT_DIR = ROOT_DIR / "data" / "tf_idf"
TFIDF_NPZ_PATH = OUT_DIR / "tfidf_matrix.npz"
VECTORIZER_PKL_PATH = OUT_DIR / "tfidf_vectorizer.joblib"
DOC_IDS_NPY = OUT_DIR / "doc_ids.npy"
DOC_TITLES_NPY = OUT_DIR / "doc_titles.npy"
CUSTOM_STOPWORDS_PATH = ROOT_DIR / "src" / "custom_stopwords.txt"

# Recommended params for ~900k CS papers (tune if needed)
def load_custom_stopwords(path: Path) -> list[str]:
    if not path.exists():
        print(f"[warn] Custom stopword file not found: {path}")
        return []
    words = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            words.add(token)
    print(f"[i] Loaded {len(words)} custom stopwords from {path}")
    return sorted(words)


VECTORIZER_KW = dict(
    max_df=0.8,        #drop terms in >=80% docs
    min_df=5,         #drop terms in <5 docs
    max_features=100_000,  #cap vocab size to control memory
    ngram_range=(1, 2),    # # Only unigrams; change to (1,2) for phrases (significantly increases scale)
    sublinear_tf=True,     # log/sublinear TF scaling
    norm="l2",             # 
    dtype=np.float32,      # halve memory vs float64
    stop_words=None,       # Data has been cleaned and stemmed; no additional universal stop words are applied here
    lowercase=False,       # processed_content already lower
)

# Sample inspection
PRINT_TOP_N_DOCS = 5       # inspect first N docs
TOP_TERMS_PER_DOC = 5     # show top-K terms per doc


def read_corpus_and_meta(jsonl_path):
    """
    Read corpus & metadata: return (texts, ids, titles)
    """
    texts, ids, titles = [], [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # print("[warn] malformed JSON line skipped")
                continue
            # --------------------------------------------------------------
            text = rec.get("processed_content") or ""
            paper_id = rec.get("id") or ""
            title = rec.get("title") or ""
            texts.append(text)
            ids.append(paper_id)
            titles.append(title)
    return texts, np.array(ids), np.array(titles)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    custom_stopwords = load_custom_stopwords(CUSTOM_STOPWORDS_PATH)
    if custom_stopwords:
        VECTORIZER_KW["stop_words"] = custom_stopwords
    # 1) Read texts & metadata
    texts, ids, titles = read_corpus_and_meta(INPUT_JSONL)
    print(f"[i] Loaded documents: {len(texts):,}")
    # 2)TF-IDF:Fit + transform
    vectorizer = TfidfVectorizer(**VECTORIZER_KW)
    X = vectorizer.fit_transform(texts)
    print(f"[i] TF-IDF shape: {X.shape}, nnz={X.nnz:,}, dtype={X.dtype}, type={type(X)}")
    # 3)Persist to disk
    sparse.save_npz(TFIDF_NPZ_PATH, X)
    joblib.dump(vectorizer, VECTORIZER_PKL_PATH)
    np.save(DOC_IDS_NPY, ids)
    np.save(DOC_TITLES_NPY, titles)
    print(f"[i] Saved: {TFIDF_NPZ_PATH}")
    print(f"[i] Saved: {VECTORIZER_PKL_PATH}")
    print(f"[i] Saved: {DOC_IDS_NPY}, {DOC_TITLES_NPY}")
    #4 Quick sanity check: show top terms for first few docs
    feats = vectorizer.get_feature_names_out()
    for di in range(min(PRINT_TOP_N_DOCS, X.shape[0])):
        row = X[di]
        idx = row.indices
        val = row.data
        if val.size == 0:
            print(f"Doc#{di} ({ids[di]}): <empty vector>")
            continue
        order = np.argsort(-val)[:TOP_TERMS_PER_DOC]
        top_terms = [(feats[idx[j]], float(val[j])) for j in order]
        print(f"Doc#{di} ({ids[di]}) '{titles[di]}' top terms:", top_terms)

if __name__ == "__main__":
    main()