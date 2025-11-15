from pathlib import Path
import re, time, json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from scipy import sparse

# 1) paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TFIDF_DIR = DATA_DIR / "tf_idf"
LSA_DIR = DATA_DIR / "lsa"
OUT_DIR = DATA_DIR / "similarity_results" / "similarity_results_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LSH_PARAMS = {
    "n_planes": 48,
    "band_size": 4,
    "seed": 13,
    "max_candidates": 800,
}

_LSH_CACHE: Optional[Dict[str, object]] = None

# 2) inputs (big matrices)
DOC_IDS_PATH = TFIDF_DIR / "doc_ids.npy"
DOC_TITLES_PATH = TFIDF_DIR / "doc_titles.npy"
TFIDF_MATRIX_PATH = TFIDF_DIR / "tfidf_matrix.npz"
LSA_MATRIX_PATH = LSA_DIR / "lsa_reduced.npz"  # expects 'X_reduced'
LSA_CLUSTER_LABELS_PATH = LSA_DIR / "cluster_labels.npy"
LSA_CLUSTER_TOPICS_JSON = ROOT_DIR /"src"/"cluster_topics.json"

# 3) artifacts (read them from similarity_results_v2)
VOCAB_JSON = OUT_DIR / "vocab.json"
IDF_NPY = OUT_DIR / "idf.npy"
USE_L2_TXT = OUT_DIR / "use_l2_norm.txt"
SVD_COMPONENTS_NPY = OUT_DIR / "svd_components.npy"
NCOMP_TXT = OUT_DIR / "n_components.txt"
TFIDF_ROW_NORMS_NPY = OUT_DIR / "row_l2_norms.npy"
LSA_PRENORM_NPZ = OUT_DIR / "lsa_reduced_l2norm.npz"

RAW_JSONL_PATH = DATA_DIR / "preprocess" / "arxiv-metadata-oai-snapshot_preprocessed.json"
CUSTOM_STOPWORDS_PATH = ROOT_DIR / "src" / "custom_stopwords.txt"


def load_custom_stopwords() -> set:
    """Load custom stopwords from file."""
    if not CUSTOM_STOPWORDS_PATH.exists():
        print(f"[warn] Custom stopwords file not found: {CUSTOM_STOPWORDS_PATH}")
        return set()
    
    try:
        stopwords_set = {
            line.strip().lower()
            for line in CUSTOM_STOPWORDS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        print(f"[i] Loaded {len(stopwords_set)} custom stopwords.")
        return stopwords_set
    except Exception as exc:
        print(f"[warn] Failed to load custom stopwords: {exc}")
        return set()


_CUSTOM_STOPWORDS: Optional[set] = None

def get_custom_stopwords() -> set:
    """Get custom stopwords (cached)."""
    global _CUSTOM_STOPWORDS
    if _CUSTOM_STOPWORDS is None:
        _CUSTOM_STOPWORDS = load_custom_stopwords()
    return _CUSTOM_STOPWORDS


def preprocess_query(q: str) -> str:
    # lowercase + keep [a-z0-9] + collapse spaces
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    
    # Filter custom stopwords
    custom_stopwords = get_custom_stopwords()
    if custom_stopwords:
        tokens = [token for token in q.split() if token not in custom_stopwords]
        q = " ".join(tokens)
    
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

    cluster_labels: Optional[np.ndarray] = None
    if LSA_CLUSTER_LABELS_PATH.exists():
        cluster_labels = np.load(LSA_CLUSTER_LABELS_PATH)
        if cluster_labels.shape[0] != doc_ids.shape[0]:
            print(
                "[warn] LSA cluster labels length mismatch; ignoring cluster mapping "
                f"({cluster_labels.shape[0]} vs {doc_ids.shape[0]})"
            )
            cluster_labels = None
    else:
        print("[info] LSA cluster labels not found; cluster ids will be omitted.")

    cluster_topics: Optional[Dict[int, List[str]]] = None
    if LSA_CLUSTER_TOPICS_JSON.exists():
        try:
            raw_topics = json.loads(LSA_CLUSTER_TOPICS_JSON.read_text(encoding="utf-8"))
            if isinstance(raw_topics, dict):
                cluster_topics = {}
                for key, value in raw_topics.items():
                    try:
                        cid = int(key)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(value, (list, tuple)):
                        cluster_topics[cid] = [str(t) for t in value]
                    else:
                        cluster_topics[cid] = [str(value)]
        except Exception as exc:
            print(f"[warn] failed to load cluster topics: {exc}")

    return (
        vocab,
        idf,
        use_l2,
        comps,
        ncomp,
        doc_ids,
        doc_titles,
        cluster_labels,
        cluster_topics,
    )

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


def _load_lsa_matrix() -> np.ndarray:
    if LSA_PRENORM_NPZ.exists():
        d = np.load(LSA_PRENORM_NPZ, allow_pickle=False)
        try:
            return d["Xr_norm"]
        finally:
            if hasattr(d, "close"):
                d.close()
    d = np.load(LSA_MATRIX_PATH, allow_pickle=False)
    try:
        Xr = d["X_reduced"]
    finally:
        if hasattr(d, "close"):
            d.close()
    return l2_normalize_dense(Xr)


def _build_lsh_cache() -> Dict[str, object]:
    Xr = _load_lsa_matrix()
    params = dict(LSH_PARAMS)
    n_planes = params["n_planes"]
    band_size = max(1, min(params["band_size"], n_planes))
    seed = params["seed"]

    rng = np.random.default_rng(seed)
    hyperplanes = rng.standard_normal((n_planes, Xr.shape[1]))
    hyperplanes = l2_normalize_dense(hyperplanes)

    projections = Xr @ hyperplanes.T
    signatures = (projections >= 0).astype(np.uint8)

    buckets: Dict[Tuple[int, int, int], Tuple[int, ...]] = {}
    tmp = defaultdict(list)
    for idx in range(signatures.shape[0]):
        sig = signatures[idx]
        for start in range(0, n_planes, band_size):
            end = min(start + band_size, n_planes)
            code = 0
            for bit in sig[start:end]:
                code = (code << 1) | int(bit)
            key = (start // band_size, end - start, code)
            tmp[key].append(idx)
    for key, vals in tmp.items():
        buckets[key] = tuple(vals)

    return {
        "Xr": Xr,
        "hyperplanes": hyperplanes,
        "band_size": band_size,
        "n_planes": n_planes,
        "buckets": buckets,
    }


def _get_lsh_cache() -> Dict[str, object]:
    global _LSH_CACHE
    if _LSH_CACHE is None:
        _LSH_CACHE = _build_lsh_cache()
    return _LSH_CACHE


def _lsh_candidates(q_lsa: np.ndarray, cache: Dict[str, object]) -> List[int]:
    hyperplanes = cache["hyperplanes"]
    band_size = cache["band_size"]
    n_planes = cache["n_planes"]
    buckets = cache["buckets"]

    projections = q_lsa @ hyperplanes.T
    signature = (projections >= 0).astype(np.uint8).ravel()

    cand = set()
    for start in range(0, n_planes, band_size):
        end = min(start + band_size, n_planes)
        code = 0
        for bit in signature[start:end]:
            code = (code << 1) | int(bit)
        key = (start // band_size, end - start, code)
        bucket = buckets.get(key)
        if bucket:
            cand.update(bucket)
    return list(cand)

def search_tfidf(query: str, top_k: int) -> List[Tuple[str, str, float, Optional[int], Optional[List[str]]]]:
    # tfidf retrieval
    (
        vocab,
        idf,
        use_l2,
        comps,
        ncomp,
        doc_ids,
        doc_titles,
        cluster_labels,
        cluster_topics,
    ) = load_minimal_artifacts()
    X = sparse.load_npz(TFIDF_MATRIX_PATH)
    q = query_to_tfidf_vec(preprocess_query(query), vocab, idf)
    D_norms = np.load(TFIDF_ROW_NORMS_NPY) if TFIDF_ROW_NORMS_NPY.exists() else None
    top_idx, sims = cosine_topk_sparse(q, X, top_k, D_norms)
    results: List[Tuple[str, str, float, Optional[int], Optional[List[str]]]] = []
    for i in top_idx:
        if sims[i] > 0:
            cluster_id: Optional[int] = None
            cluster_topics_list: Optional[List[str]] = None
            if cluster_labels is not None and i < cluster_labels.shape[0]:
                cluster_id = int(cluster_labels[i])
                if cluster_topics is not None:
                    topics = cluster_topics.get(cluster_id)
                    if topics is not None:
                        cluster_topics_list = list(topics)
            results.append(
                (
                    str(doc_ids[i]),
                    str(doc_titles[i]),
                    float(sims[i]),
                    cluster_id,
                    cluster_topics_list,
                )
            )
    return results

def search_lsa(query: str, top_k: int) -> List[Tuple[str, str, float, Optional[int], Optional[List[str]]]]:
    # lsa retrieval
    (
        vocab,
        idf,
        use_l2,
        comps,
        ncomp,
        doc_ids,
        doc_titles,
        cluster_labels,
        cluster_topics,
    ) = load_minimal_artifacts()
    d = np.load(LSA_MATRIX_PATH)
    Xr = d["X_reduced"]
    if LSA_PRENORM_NPZ.exists():
        Xr = np.load(LSA_PRENORM_NPZ)["Xr_norm"]
    else:
        Xr = l2_normalize_dense(Xr)
    q_tfidf = query_to_tfidf_vec(preprocess_query(query), vocab, idf).toarray()
    
    # Check and adjust dimensions to match comps
    vocab_size = q_tfidf.shape[1]
    comps_vocab_size = comps.shape[1]
    
    if vocab_size != comps_vocab_size:
        # Pad or truncate q_tfidf to match comps dimensions
        if vocab_size < comps_vocab_size:
            # Pad with zeros if vocab is smaller
            padding = np.zeros((1, comps_vocab_size - vocab_size))
            q_tfidf = np.hstack([q_tfidf, padding])
        else:
            # Truncate if vocab is larger
            q_tfidf = q_tfidf[:, :comps_vocab_size]
        print(f"[warn] vocab size mismatch: {vocab_size} vs {comps_vocab_size}, adjusted")
    
    q_lsa = q_tfidf @ comps.T
    q_lsa = l2_normalize_dense(q_lsa)
    sims = (q_lsa @ Xr.T).ravel()
    if top_k >= len(sims):
        top = np.argsort(-sims)
    else:
        top = np.argpartition(-sims, top_k)[:top_k]
        top = top[np.argsort(-sims[top])]
    results: List[Tuple[str, str, float, Optional[int], Optional[List[str]]]] = []
    for i in top:
        if sims[i] > 0:
            cluster_id: Optional[int] = None
            cluster_topics_list: Optional[List[str]] = None
            if cluster_labels is not None and i < cluster_labels.shape[0]:
                cluster_id = int(cluster_labels[i])
                if cluster_topics is not None:
                    topics = cluster_topics.get(cluster_id)
                    if topics is not None:
                        cluster_topics_list = list(topics)
            results.append(
                (
                    str(doc_ids[i]),
                    str(doc_titles[i]),
                    float(sims[i]),
                    cluster_id,
                    cluster_topics_list,
                )
            )
    return results


def search_lsa_lsh(
    query: str,
    top_k: int,
    max_candidates: Optional[int] = None,
) -> List[Tuple[str, str, float, Optional[int], Optional[List[str]]]]:
    (
        vocab,
        idf,
        _use_l2,
        comps,
        _ncomp,
        doc_ids,
        doc_titles,
        cluster_labels,
        cluster_topics,
    ) = load_minimal_artifacts()

    cache = _get_lsh_cache()
    Xr = cache["Xr"]
    if max_candidates is None:
        max_candidates = LSH_PARAMS.get("max_candidates", 800)

    q_tfidf = query_to_tfidf_vec(preprocess_query(query), vocab, idf).toarray()
    if q_tfidf.sum() == 0:
        return []
    
    # Check and adjust dimensions to match comps
    vocab_size = q_tfidf.shape[1]
    comps_vocab_size = comps.shape[1]
    
    if vocab_size != comps_vocab_size:
        # Pad or truncate q_tfidf to match comps dimensions
        if vocab_size < comps_vocab_size:
            # Pad with zeros if vocab is smaller
            padding = np.zeros((1, comps_vocab_size - vocab_size))
            q_tfidf = np.hstack([q_tfidf, padding])
        else:
            # Truncate if vocab is larger
            q_tfidf = q_tfidf[:, :comps_vocab_size]
        print(f"[warn] vocab size mismatch: {vocab_size} vs {comps_vocab_size}, adjusted")
    
    q_lsa = q_tfidf @ comps.T
    q_lsa = l2_normalize_dense(q_lsa)

    cand_idx = _lsh_candidates(q_lsa, cache)
    if not cand_idx:
        cand_idx = np.arange(Xr.shape[0], dtype=np.int32)
    else:
        cand_idx = np.array(cand_idx, dtype=np.int32)
        if max_candidates and len(cand_idx) > max_candidates:
            preview = (q_lsa @ Xr[cand_idx].T).ravel()
            order = np.argsort(-preview)
            cand_idx = cand_idx[order[:max_candidates]]

    sims = (q_lsa @ Xr[cand_idx].T).ravel()
    if sims.size == 0:
        return []

    if top_k >= sims.size:
        order = np.argsort(-sims)
    else:
        order = np.argpartition(-sims, top_k)[:top_k]
        order = order[np.argsort(-sims[order])]

    results: List[Tuple[str, str, float, Optional[int], Optional[List[str]]]] = []
    for pos in order:
        doc_idx = int(cand_idx[pos])
        sim_val = float(sims[pos])
        if sim_val <= 0:
            continue
        cluster_id: Optional[int] = None
        cluster_topics_list: Optional[List[str]] = None
        if cluster_labels is not None and doc_idx < cluster_labels.shape[0]:
            cluster_id = int(cluster_labels[doc_idx])
            if cluster_topics is not None:
                topics = cluster_topics.get(cluster_id)
                if topics is not None:
                    cluster_topics_list = list(topics)
        results.append(
            (
                str(doc_ids[doc_idx]),
                str(doc_titles[doc_idx]),
                sim_val,
                cluster_id,
                cluster_topics_list,
            )
        )
    return results

def save_results_jsonl(query: str, method: str, results: List[Tuple[str, str, float, Optional[int], Optional[List[str]]]]) -> Path:
    # 收集需要的 paper id
    need_ids = {pid for (pid, _title, _s, _cluster, _topics) in results}
    raw_meta = load_raw_meta(need_ids)

    # 输出文件路径（仍然是 jsonl 格式内容）
    ts = int(time.time())
    out_path = OUT_DIR / f"similarity_for_recommend_{method}_{ts}.json"

    with out_path.open("w", encoding="utf-8") as f:
        for rank, (pid, title, sc, cluster_id, cluster_topics_list) in enumerate(results, 1):
            base = raw_meta.get(pid, {})
            base.update({
                "sim_score": float(sc),
                "score": float(sc),
                "similarity": float(sc),
                "rank": rank,
                "query": query,
                "method": method,
                "lsa_cluster_id": int(cluster_id) if cluster_id is not None else None,
                "topics": cluster_topics_list if cluster_topics_list is not None else None,
            })
            f.write(json.dumps(base, ensure_ascii=False) + "\n")

    print(f"saved to: {out_path}")
    return out_path



def print_results(query: str, method: str, results: List[Tuple[str, str, float, Optional[int], Optional[List[str]]]]):
    # simple pretty print
    print("\n" + "=" * 70)
    print(f"query: {query}")
    print(f"model: {method}")
    print(f"top {len(results)} here ↓")
    print("=" * 70)
    for i, (pid, title, sc, cluster_id, cluster_topics_list) in enumerate(results, 1):
        print(f"{i:2d}. [{pid}] {title}")
        cluster_str = "N/A" if cluster_id is None else str(cluster_id)
        print(f"    score: {sc:.4f} | cluster: {cluster_str}")
        if cluster_topics_list:
            pretty_topics = ", ".join(cluster_topics_list)
            print(f"    topics: {pretty_topics}")
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
    method = input("model? (tfidf/lsa/lsa_lsh): ").strip().lower()
    if method not in {"tfidf", "lsa", "lsa_lsh"}:
        print("oops, pick tfidf, lsa, or lsa_lsh")
        return
    query = input("type your query: ").strip()
    top_k_s = input("top-k? (default 10): ").strip()
    top_k = int(top_k_s) if top_k_s.isdigit() and int(top_k_s) > 0 else 10

    t0 = time.time()
    if method == "tfidf":
        results = search_tfidf(query, top_k)
    elif method == "lsa":
        results = search_lsa(query, top_k)
    else:
        results = search_lsa_lsh(query, top_k)
    dt = time.time() - t0

    print_results(query, method, results)
    out_file = save_results_jsonl(query, method, results)
    print(f"saved to: {out_file}")
    print(f"time: {dt:.2f}s")

if __name__ == "__main__":
    main()
