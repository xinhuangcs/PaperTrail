
"""
Similar Paper Retrieval

"""

import os
import re
import csv
import time
import json
import math
import joblib
import numpy as np
from typing import Tuple
from datetime import datetime
from scipy import sparse

# --------------------- CONFIG 可修改配置 / Config ---------------------
CONFIG = {
    # Input
    "TFIDF_MATRIX": "/Users/jasonh/Desktop/02807/PaperTrail/data/tf_idf/tfidf_matrix.npz",
    "TFIDF_VECTORIZER": "/Users/jasonh/Desktop/02807/PaperTrail/data/tf_idf/tfidf_vectorizer.joblib",
    "DOC_IDS": "/Users/jasonh/Desktop/02807/PaperTrail/data/tf_idf/doc_ids.npy",
    "DOC_TITLES": "/Users/jasonh/Desktop/02807/PaperTrail/data/tf_idf/doc_titles.npy",
    # LSA artifacts
    "LSA_MATRIX_NPZ": "/Users/jasonh/Desktop/02807/PaperTrail/data/lsa/lsa_reduced.npz",  # contains 'X_reduced'
    "SVD_MODEL": "/Users/jasonh/Desktop/02807/PaperTrail/data/lsa/svd_model.joblib",  # optional for query transform

    # Runtime switches
    "USE_TFIDF": False,  # use TF-IDF space
    "USE_LSA": True,  #  use LSA space
    "RETRIEVE_K": 1000,  # initial recall size
    "TOP_N": 50,  # exported top N

    #Output dir
    "OUT_DIR": "/Users/jasonh/Desktop/02807/PaperTrail/data/retrieval/query_results",

    # Query preprocessing
    "APPLY_PORTER_STEM": True,  #apply Porter stemming on query
    "LOWERCASE": True,  # lowercase
    "REMOVE_NON_ALNUM": True,  # remove non-alphanumeric
}

# --------------------- preprocessing ---------------------
try:
    from nltk.stem import PorterStemmer

    _STEMMER = PorterStemmer()
except Exception:
    _STEMMER = None


def preprocess_query(q: str) -> str:
#cleaning consistent with training.
    s = q or ""
    if CONFIG["LOWERCASE"]:
        s = s.lower()
    if CONFIG["REMOVE_NON_ALNUM"]:
        s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if CONFIG["APPLY_PORTER_STEM"] and _STEMMER is not None:
        tokens = [_STEMMER.stem(t) for t in s.split()]
        s = " ".join(tokens)
    return s


#Similarity
def topk_by_dot(query_vec, doc_matrix, k: int) -> Tuple[np.ndarray, np.ndarray]:
#Compute query·docs^T, return Top-K indices and scores.
    scores = (query_vec @ doc_matrix.T)
    if sparse.issparse(scores):
        scores = scores.toarray()
    scores = np.asarray(scores).ravel()

    k = min(k, scores.shape[0])
    if k <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    # argpartition O(N)
    idx_part = np.argpartition(scores, -k)[-k:]
    idx_sorted_local = idx_part[np.argsort(scores[idx_part])[::-1]]
    return idx_sorted_local, scores[idx_sorted_local]


# --------------------- 主流程 / Main ---------------------
def main():
    os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)

    # Load artifacts
    print("[i] Loading artifacts ...")
    doc_ids = np.load(CONFIG["DOC_IDS"], allow_pickle=True)
    doc_titles = np.load(CONFIG["DOC_TITLES"], allow_pickle=True)

    # TF-IDF
    X_tfidf = None
    vectorizer = None
    if CONFIG["USE_TFIDF"]:
        X_tfidf = sparse.load_npz(CONFIG["TFIDF_MATRIX"])  # CSR (N, V), 通常已L2归一化
        vectorizer = joblib.load(CONFIG["TFIDF_VECTORIZER"])
        print(f"[i] TF-IDF loaded: shape={X_tfidf.shape}, nnz={X_tfidf.nnz}")

    # LSA
    X_lsa = None
    svd_model = None
    if CONFIG["USE_LSA"]:
        lsa_npz = np.load(CONFIG["LSA_MATRIX_NPZ"], mmap_mode="r")
        X_lsa = lsa_npz["X_reduced"]
        print(
            f"[i] LSA loaded: shape={X_lsa.shape} (~{X_lsa.shape[0] * X_lsa.shape[1] * 4 / 1e6:.1f} MB if fully loaded)")
        if os.path.exists(CONFIG["SVD_MODEL"]):
            svd_model = joblib.load(CONFIG["SVD_MODEL"])
            print("[i] SVD model loaded for query projection.")

    # Interactive loop
    while True:
        try:
            q = input("\n[Query] 请输入查询文本（回车退出 / press Enter to quit）：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[bye]")
            break
        if not q:
            break

        top_n = input(f"[TopN] 请输入Top N（默认 {CONFIG['TOP_N']}）：").strip()
        if top_n.isdigit():
            top_n = int(top_n)
        else:
            top_n = CONFIG["TOP_N"]

        # 预处理 & 向量化 / preprocess & vectorize
        q_clean = preprocess_query(q)
        print(f"[i] preprocessed query: \"{q_clean}\"")

        # 收集来自不同空间的候选 / gather candidates from selected spaces
        candidates = {}  # doc_index -> best score among spaces
        records = []  # for CSV rows

        # (A) TF-IDF space
        if CONFIG["USE_TFIDF"]:
            if vectorizer is None or X_tfidf is None:
                print("[warn] TF-IDF artifacts missing, skip TF-IDF retrieval.")
            else:
                t0 = time.time()
                qv = vectorizer.transform([q_clean])  # (1, V) sparse
                idx, scores = topk_by_dot(qv, X_tfidf, k=max(CONFIG["RETRIEVE_K"], top_n))
                dt = time.time() - t0
                print(f"[i] TF-IDF retrieval took {dt:.3f}s; got {len(idx)} candidates.")

                for i, s in zip(idx, scores):
                    prev = candidates.get(i, -math.inf)
                    if s > prev:
                        candidates[i] = s
                    records.append(("tfidf", int(i), float(s)))

        # (B) LSA space
        if CONFIG["USE_LSA"]:
            if X_lsa is None:
                print("[warn] LSA artifacts missing, skip LSA retrieval.")
            else:
                # 若有SVD模型，用TF-IDF向量 -> SVD投影；否则只能用LSA矩阵做“被动检索”（无法投影query）
                if svd_model is not None and vectorizer is not None:
                    t0 = time.time()
                    qv = vectorizer.transform([q_clean])  # (1, V)
                    qv_lsa = svd_model.transform(qv)  # (1, d)
                    # 余弦相似度（未归一化时，用点积近似；Phase 6会再标准化）
                    idx, scores = topk_by_dot(qv_lsa, X_lsa, k=max(CONFIG["RETRIEVE_K"], top_n))
                    dt = time.time() - t0
                    print(f"[i] LSA retrieval took {dt:.3f}s; got {len(idx)} candidates.")
                    for i, s in zip(idx, scores):
                        prev = candidates.get(i, -math.inf)
                        if s > prev:
                            candidates[i] = s
                        records.append(("lsa", int(i), float(s)))
                else:
                    print("[warn] No SVD model for query projection; skip LSA query projection.")

        if not candidates:
            print("[x] 无候选返回 / No candidates.")
            continue

        # 融合两个空间的候选：取两空间中更高的相似度作为该文档的候选分（简单策略）
        # Merge candidates by taking the higher score between spaces
        cand_idx = np.array(list(candidates.keys()), dtype=int)
        cand_scores = np.array([candidates[i] for i in cand_idx], dtype=float)

        # 最终 Top-N（Phase 5 输出）/ Final Top-N for Phase 5 output
        k = min(top_n, cand_idx.shape[0])
        top_part = np.argpartition(cand_scores, -k)[-k:]
        order = top_part[np.argsort(cand_scores[top_part])[::-1]]
        final_idx = cand_idx[order]
        final_scores = cand_scores[order]

        # 导出CSV / export CSV
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = ("tfidf" if CONFIG["USE_TFIDF"] else "") + (
            "+" if (CONFIG["USE_TFIDF"] and CONFIG["USE_LSA"]) else "") + ("lsa" if CONFIG["USE_LSA"] else "")
        out_csv = os.path.join(CONFIG["OUT_DIR"], f"retrieve_{stamp}__mode-{mode}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "doc_index", "id", "title", "sim_score", "vector_space", "query"])
            # 标注这条结果来源于哪个空间：取该doc在records中最高分对应的空间（近似）
            space_by_doc = {}
            for sp, i, s in records:
                if (i not in space_by_doc) or (s > space_by_doc[i][1]):
                    space_by_doc[i] = (sp, s)
            for r, (i, s) in enumerate(zip(final_idx, final_scores), start=1):
                vid = str(doc_ids[i])
                vtitle = str(doc_titles[i])
                sp = space_by_doc.get(int(i), ("mixed", float(s)))[0]
                w.writerow([r, int(i), vid, vtitle, float(s), sp, q])

        print(f"[i] Saved Phase5 result CSV: {out_csv}")


if __name__ == "__main__":
    main()
