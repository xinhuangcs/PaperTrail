
"""
Load TF-IDF & vectorizer; take a query string; return top-K similar papers (dot ≈ cosine).

Requirements:
  pip install scikit-learn scipy joblib numpy nltk
"""

import re
import numpy as np
from scipy import sparse
import joblib

#Lightweight stemming
try:
    from nltk.stem import PorterStemmer
    STEMMER = PorterStemmer()
except Exception:
    STEMMER = None

#config
OUT_DIR = "/Users/jasonh/Desktop/02807/PaperTrail/data/tf_idf"
TFIDF_NPZ_PATH = OUT_DIR + "/tfidf_matrix.npz"
VECTORIZER_PKL_PATH = OUT_DIR + "/tfidf_vectorizer.joblib"
DOC_IDS_NPY = OUT_DIR + "/doc_ids.npy"
DOC_TITLES_NPY = OUT_DIR + "/doc_titles.npy"

TOP_K = 10  #top-K results

def preprocess_query(s: str) -> str:
#Approximate your corpus processing: lowercase, strip non-words, Porter stemming
    s = s.lower()
    #keep alnum & space
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if STEMMER:
        tokens = s.split()
        tokens = [STEMMER.stem(t) for t in tokens]
        s = " ".join(tokens)
    return s

def main():
    # 1) Load
    X = sparse.load_npz(TFIDF_NPZ_PATH)             # (N_docs, V) CSR
    vec = joblib.load(VECTORIZER_PKL_PATH)          # fitted vectorizer
    ids = np.load(DOC_IDS_NPY, allow_pickle=True)   # (N_docs,)
    titles = np.load(DOC_TITLES_NPY, allow_pickle=True)

    print("[i] Loaded TF-IDF & vectorizer. Type your query. Empty to exit.")
    while True:
        try:
            q = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break

        q_clean = preprocess_query(q)
        # Note: vectorizer wasn't trained to stem, but corpus was stemmed already;
        # so stem the query to match the corpus morphology.

        qv = vec.transform([q_clean])  # (1, V) sparse
        # L2-normalized → dot equals cosine similarity
        scores = (qv @ X.T).toarray().ravel()  # (N_docs,)

        if scores.max() == 0.0:
            print("No similar documents found (all-zero). Try another query.")
            continue

        top_idx = np.argsort(scores)[-TOP_K:][::-1]
        print(f"Top-{TOP_K} similar papers:")
        for rank, di in enumerate(top_idx, start=1):
            print(f"{rank:2d}. [{ids[di]}] {titles[di]}  (score={scores[di]:.4f})")

if __name__ == "__main__":
    main()