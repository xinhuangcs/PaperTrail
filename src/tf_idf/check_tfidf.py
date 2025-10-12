
"""
Load saved TF-IDF matrix & vectorizer; print top-k terms for first N docs for check.

Requirements:
  pip install scikit-learn scipy joblib numpy
"""

import numpy as np
from scipy import sparse
import joblib

#config
OUT_DIR = "/Users/jasonh/Desktop/02807/PaperTrail/data/tf_idf"
TFIDF_NPZ_PATH = OUT_DIR + "/tfidf_matrix.npz"
VECTORIZER_PKL_PATH = OUT_DIR + "/tfidf_vectorizer.joblib"
DOC_IDS_NPY = OUT_DIR + "/doc_ids.npy"
DOC_TITLES_NPY = OUT_DIR + "/doc_titles.npy"

PRINT_TOP_N_DOCS = 10
TOP_TERMS_PER_DOC = 10

def main():
    # 1)Load
    X = sparse.load_npz(TFIDF_NPZ_PATH)             # (N_docs, V)
    vectorizer = joblib.load(VECTORIZER_PKL_PATH)   # fitted vectorizer
    ids = np.load(DOC_IDS_NPY, allow_pickle=True)
    titles = np.load(DOC_TITLES_NPY, allow_pickle=True)

    feats = vectorizer.get_feature_names_out()
    print(f"[i] Matrix: {X.shape}, nnz={X.nnz:,}")
    print(f"[i] Vocabulary size: {len(feats):,}")

    # 2)Show top-K terms for first N docs
    N = min(PRINT_TOP_N_DOCS, X.shape[0])
    for di in range(N):
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