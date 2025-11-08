"""
Interpret K-Means clusters via LSA back-projection
"""

import json
import numpy as np
from pathlib import Path
import joblib
from collections import defaultdict

#config
TFIDF_DIR = Path("/Users/jasonh/Desktop/02807/PaperTrail/data/tf_idf")
VECTORIZER_PKL_PATH = TFIDF_DIR / "tfidf_vectorizer.joblib"
DOC_IDS_NPY = TFIDF_DIR / "doc_ids.npy"
DOC_TITLES_NPY = TFIDF_DIR / "doc_titles.npy"

LSA_DIR = Path("/Users/jasonh/Desktop/02807/PaperTrail/data/lsa")
LSA_REDUCED_NPZ = LSA_DIR / "lsa_reduced.npz"
CLUSTER_LABELS_NPY = LSA_DIR / "cluster_labels.npy"
SVD_MODEL_PKL = LSA_DIR / "lsa_model.joblib"

OUT_DIR = LSA_DIR / "cluster_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


TOP_K_TERMS = 20


def load_artifacts():
    # LSA doc-topic matrix
    if not LSA_REDUCED_NPZ.exists():
        raise FileNotFoundError(f"Missing {LSA_REDUCED_NPZ}")
    X_lsa = np.load(LSA_REDUCED_NPZ)["X_reduced"]

    # cluster labels
    if not CLUSTER_LABELS_NPY.exists():
        raise FileNotFoundError(f"Missing {CLUSTER_LABELS_NPY}")
    labels = np.load(CLUSTER_LABELS_NPY)

    if X_lsa.shape[0] != labels.shape[0]:
        raise ValueError(f"Row mismatch: LSA rows {X_lsa.shape[0]} vs labels {labels.shape[0]}")

    # vectorizer to get terms
    if not VECTORIZER_PKL_PATH.exists():
        raise FileNotFoundError(f"Missing {VECTORIZER_PKL_PATH}")
    vectorizer = joblib.load(VECTORIZER_PKL_PATH)
    terms = np.array(vectorizer.get_feature_names_out())

    # doc ids / titles
    if not DOC_IDS_NPY.exists() or not DOC_TITLES_NPY.exists():
        raise FileNotFoundError("Missing doc_ids.npy or doc_titles.npy")
    doc_ids = np.load(DOC_IDS_NPY, allow_pickle=True)
    doc_titles = np.load(DOC_TITLES_NPY, allow_pickle=True)

    if doc_ids.shape[0] != X_lsa.shape[0] or doc_titles.shape[0] != X_lsa.shape[0]:
        raise ValueError("doc_ids/doc_titles length must match LSA rows")

    # SVD model for components_
    if not SVD_MODEL_PKL.exists():
        raise FileNotFoundError(
            f"Missing {SVD_MODEL_PKL}. Please joblib.dump(svd, SVD_MODEL_PKL) in build_lsa.py"
        )
    svd = joblib.load(SVD_MODEL_PKL)
    components = svd.components_  # (n_components, n_terms)

    if X_lsa.shape[1] != components.shape[0]:
        raise ValueError(
            f"LSA dim {X_lsa.shape[1]} must match SVD components rows {components.shape[0]}"
        )

    return X_lsa, labels, terms, components, doc_ids, doc_titles


def backproject_centroid(centroid_lsa: np.ndarray, components: np.ndarray) -> np.ndarray:
    # term_scores ≈ centroid * components  (V matrix)
    return centroid_lsa @ components  # (n_terms,)


def top_k_terms(term_scores: np.ndarray, terms: np.ndarray, k: int):
    # get top-k indices and sort
    idx = np.argpartition(term_scores, -k)[-k:]
    idx = idx[np.argsort(term_scores[idx])[::-1]]
    return terms[idx].tolist(), term_scores[idx].astype(float).tolist()


def main():
    print("[i] Loading artifacts ...")
    X_lsa, labels, terms, components, doc_ids, doc_titles = load_artifacts()
    n_docs, n_comp = X_lsa.shape
    print(f"[i] LSA: {X_lsa.shape}, components: {components.shape}, docs: {n_docs:,}")

    # group doc indices by cluster
    clusters = defaultdict(list)
    for i, lab in enumerate(labels):
        clusters[int(lab)].append(i)

    # open JSONL outputs
    kw_f = (OUT_DIR / "cluster_keywords.json").open("w", encoding="utf-8")
    cp_f = (OUT_DIR / "cluster_papers.json").open("w", encoding="utf-8")

    try:
        for c_label, idxs in clusters.items():
            idxs = np.array(idxs, dtype=int)
            if idxs.size == 0:
                continue

            # compute centroid in LSA space
            centroid = X_lsa[idxs].mean(axis=0)  # (n_components,)
            # back-project to term space
            term_scores = backproject_centroid(centroid, components)
            # get top terms
            keywords, scores = top_k_terms(term_scores, terms, TOP_K_TERMS)

            # write one line per cluster (keywords)
            kw_record = {
                "cluster": int(c_label),
                "keywords": keywords,
                "scores": scores
            }
            kw_f.write(json.dumps(kw_record, ensure_ascii=False) + "\n")

            # write one line per cluster (papers)
            papers = [{"id": str(doc_ids[i]), "title": str(doc_titles[i])} for i in idxs]
            cp_record = {
                "cluster": int(c_label),
                "papers": papers
            }
            cp_f.write(json.dumps(cp_record, ensure_ascii=False) + "\n")

            print(f"[i] Cluster {c_label}: size={len(idxs)}, top_keywords={keywords[:8]} ...")
    finally:
        kw_f.close()
        cp_f.close()

    print("Saved JSON to:", OUT_DIR)


if __name__ == "__main__":
    main()
