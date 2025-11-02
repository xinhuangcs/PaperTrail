from pathlib import Path
import re, time, json
import numpy as np
from typing import List, Tuple
from scipy import sparse

# 1) paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TFIDF_DIR = DATA_DIR / "tf_idf"
LSA_DIR = DATA_DIR / "lsa"
OUT_DIR = DATA_DIR / "similarity_results" / "similarity_results_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2) inputs (big matrices)
DOC_IDS_PATH = TFIDF_DIR / "doc_ids.npy"
DOC_TITLES_PATH = TFIDF_DIR / "doc_titles.npy"
TFIDF_MATRIX_PATH = TFIDF_DIR / "tfidf_matrix.npz"
LSA_MATRIX_PATH = LSA_DIR / "lsa_reduced.npz"  # expects 'X_reduced'

# 3) artifacts (read them from similarity_results_v2)
VOCAB_JSON = OUT_DIR / "vocab.json"
IDF_NPY = OUT_DIR / "idf.npy"
USE_L2_TXT = OUT_DIR / "use_l2_norm.txt"
SVD_COMPONENTS_NPY = OUT_DIR / "svd_components.npy"
NCOMP_TXT = OUT_DIR / "n_components.txt"
TFIDF_ROW_NORMS_NPY = OUT_DIR / "row_l2_norms.npy"
LSA_PRENORM_NPZ = OUT_DIR / "lsa_reduced_l2norm.npz"

RAW_JSONL_PATH = DATA_DIR / "preprocess" / "arxiv-cs-data-with-citations-final-dataset.json"

def preprocess_query(q: str) -> str:
    # lowercase + keep [a-z0-9] + collapse spaces
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def load_minimal_artifacts():
    # load small pieces
    vocab = json.loads(VOCAB_JSON.read_text(encoding="utf-8"))
    idf = np.load(IDF_NPY)
    use_l2 = USE_L2_TXT.read_text().strip() == "1"
    comps = np.load(SVD_COMPONENTS_NPY)   # (k, V)
    ncomp = int(NCOMP_TXT.read_text().strip() or comps.shape[0])
    doc_ids = np.load(DOC_IDS_PATH, allow_pickle=True)
    doc_titles = np.load(DOC_TITLES_PATH, allow_pickle=True)
    return vocab, idf, use_l2, comps, ncomp, doc_ids, doc_titles

def query_to_tfidf_vec(query: str, vocab: dict, idf: np.ndarray) -> sparse.csr_matrix:
    # tf (relative) * idf, then l2 norm
    tokens = query.split()
    counts = {}
    for t in tokens:
        j = vocab.get(t)
        if j is not None:
            counts[j] = counts.get(j, 0) + 1
    if not counts:
        return sparse.csr_matrix((1, idf.shape[0]))
    cols = np.fromiter(counts.keys(), dtype=np.int32)
    tf = np.fromiter(counts.values(), dtype=np.float64)
    tf = tf / tf.sum()
    data = tf * idf[cols]
    vec = sparse.csr_matrix((data, (np.zeros_like(cols), cols)), shape=(1, idf.shape[0]))
    norm = np.sqrt((vec.multiply(vec)).sum())
    if norm > 0:
        vec /= norm
    return vec

def cosine_topk_sparse(q: sparse.csr_matrix, D: sparse.csr_matrix, k: int, D_row_norms=None):
    # cosine for sparse: (q @ D.T) then divide by ||D_i||
    sims = (q @ D.T).toarray().ravel()
    if D_row_norms is not None:
        safe = D_row_norms.copy()
        safe[safe == 0] = 1.0
        sims = sims / safe
    if k >= len(sims):
        top = np.argsort(-sims)
    else:
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
    return top, sims

def l2_normalize_dense(A: np.ndarray) -> np.ndarray:
    # row-wise l2 norm
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n

def search_tfidf(query: str, top_k: int) -> List[Tuple[str, str, float]]:
    # tfidf retrieval
    vocab, idf, use_l2, comps, ncomp, doc_ids, doc_titles = load_minimal_artifacts()
    X = sparse.load_npz(TFIDF_MATRIX_PATH)
    q = query_to_tfidf_vec(preprocess_query(query), vocab, idf)
    D_norms = np.load(TFIDF_ROW_NORMS_NPY) if TFIDF_ROW_NORMS_NPY.exists() else None
    top_idx, sims = cosine_topk_sparse(q, X, top_k, D_norms)
    results: List[Tuple[str, str, float]] = []
    for i in top_idx:
        if sims[i] > 0:
            results.append((str(doc_ids[i]), str(doc_titles[i]), float(sims[i])))
    return results

def search_lsa(query: str, top_k: int) -> List[Tuple[str, str, float]]:
    # lsa retrieval
    vocab, idf, use_l2, comps, ncomp, doc_ids, doc_titles = load_minimal_artifacts()
    d = np.load(LSA_MATRIX_PATH)
    Xr = d["X_reduced"]
    if LSA_PRENORM_NPZ.exists():
        Xr = np.load(LSA_PRENORM_NPZ)["Xr_norm"]
    else:
        Xr = l2_normalize_dense(Xr)
    q_tfidf = query_to_tfidf_vec(preprocess_query(query), vocab, idf).toarray()
    q_lsa = q_tfidf @ comps.T
    q_lsa = l2_normalize_dense(q_lsa)
    sims = (q_lsa @ Xr.T).ravel()
    if top_k >= len(sims):
        top = np.argsort(-sims)
    else:
        top = np.argpartition(-sims, top_k)[:top_k]
        top = top[np.argsort(-sims[top])]
    results: List[Tuple[str, str, float]] = []
    for i in top:
        if sims[i] > 0:
            results.append((str(doc_ids[i]), str(doc_titles[i]), float(sims[i])))
    return results

def save_results_jsonl(query: str, method: str, results: List[Tuple[str, str, float]]) -> Path:
    # 收集需要的 paper id
    need_ids = {pid for (pid, _title, _s) in results}
    raw_meta = load_raw_meta(need_ids)

    # 输出文件路径（仍然是 jsonl 格式内容）
    ts = int(time.time())
    out_path = OUT_DIR / f"similarity_for_recommend_{method}_{ts}.json"

    with out_path.open("w", encoding="utf-8") as f:
        for rank, (pid, title, sc) in enumerate(results, 1):
            base = raw_meta.get(pid, {})
            base.update({
                "sim_score": float(sc),
                "score": float(sc),
                "similarity": float(sc),
                "rank": rank,
                "query": query,
                "method": method,
            })
            f.write(json.dumps(base, ensure_ascii=False) + "\n")

    print(f"saved to: {out_path}")
    return out_path



def print_results(query: str, method: str, results: List[Tuple[str, str, float]]):
    # simple pretty print
    print("\n" + "=" * 70)
    print(f"query: {query}")
    print(f"model: {method}")
    print(f"top {len(results)} here ↓")
    print("=" * 70)
    for i, (pid, title, sc) in enumerate(results, 1):
        print(f"{i:2d}. [{pid}] {title}")
        print(f"    score: {sc:.4f}")
    if not results:
        print("hmm no hit, maybe try other words")


def load_raw_meta(need_ids: set) -> dict:
    hit = {}
    if not RAW_JSONL_PATH.exists():
        print("raw file not found, skip extra fields")
        return hit
    found = 0
    target = len(need_ids)
    with RAW_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = obj.get("id")
            if pid in need_ids:
                hit[pid] = obj
                found += 1
                if found >= target:
                    break
    return hit


def main():
    # interactive cli
    print("similarity search")
    method = input("model? (tfidf/lsa): ").strip().lower()
    if method not in {"tfidf", "lsa"}:
        print("oops, pick tfidf or lsa")
        return
    query = input("type your query: ").strip()
    top_k_s = input("top-k? (default 10): ").strip()
    top_k = int(top_k_s) if top_k_s.isdigit() and int(top_k_s) > 0 else 10

    t0 = time.time()
    if method == "tfidf":
        results = search_tfidf(query, top_k)
    else:
        results = search_lsa(query, top_k)
    dt = time.time() - t0

    print_results(query, method, results)
    out_file = save_results_jsonl(query, method, results)
    print(f"saved to: {out_file}")
    print(f"time: {dt:.2f}s")

if __name__ == "__main__":
    main()
