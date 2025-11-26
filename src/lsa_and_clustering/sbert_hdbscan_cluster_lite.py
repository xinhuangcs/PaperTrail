import json
import joblib
import re
import math
from itertools import combinations
from collections import Counter, defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import csv
import datetime
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'"
)

class CsvLogger:
    def __init__(self, filepath: Path, fieldnames: list):
        self.filepath = filepath
        self.fieldnames = fieldnames
        with self.filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, data: Dict[str, Any]):
        # Ensure all keys in data are in fieldnames to avoid errors
        filtered_data = {k: data.get(k) for k in self.fieldnames}
        with self.filepath.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(filtered_data)




# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = REPO_ROOT / "data" / "preprocess" / "arxiv-cs-data-with-citations-final-dataset_preprocessed.json"

OUT_DIR = REPO_ROOT / "data" / "sbert_hdbscan_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = OUT_DIR / "sbert_embeddings.npy"
EMBEDDINGS_NORM_PATH = OUT_DIR / "sbert_embeddings_norm.npy"
CLUSTER_LABELS_PATH = OUT_DIR / "hdbscan_labels.npy"
DOC_IDS_PATH = OUT_DIR / "doc_ids.npy"
DOC_TITLES_PATH = OUT_DIR / "doc_titles.npy"
CLUSTER_TOP_TERMS_PATH = OUT_DIR / "cluster_top_terms.json"

# Model and filters
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
MIN_TEXT_CHARS = 30
CUSTOM_STOPWORDS_PATH = REPO_ROOT / "src" / "custom_stopwords.txt"
APRIORI_MIN_SUPPORT_RATIO = 0.2
APRIORI_MAX_SIZE = 3
APRIORI_TOP_K = 5
APRIORI_FALLBACK_TOP_K = 5

# UMAP pre-reduction (fixed small config)
UMAP_ENABLED = True
UMAP_DIM = 50
UMAP_METRIC = "cosine"  

# Sampled parameter search (broader and smarter)
PARAM_SEARCH = True
PARAM_SAMPLE_SIZE = 10000
# Baseline ranges (dynamic ranges are built inside search as well)
MIN_CLUSTER_SIZE_RANGE = [5, 10, 20, 30, 50, 100]
MIN_SAMPLES_RANGE = [1, 2, 5, 10, 15]
METHODS = ["eom", "leaf"]
EPS_RANGE = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
TARGET_SILHOUETTE = 1


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_CUSTOM_STOP_WORDS: Optional[set[str]] = None


def load_custom_stopwords() -> set[str]:
    global _CUSTOM_STOP_WORDS
    if _CUSTOM_STOP_WORDS is not None:
        return _CUSTOM_STOP_WORDS
    stops: set[str] = set()
    try:
        lines = CUSTOM_STOPWORDS_PATH.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        _CUSTOM_STOP_WORDS = stops
        return stops
    for line in lines:
        word = line.strip().lower()
        if not word or word.startswith("#"):
            continue
        stops.add(word)
    _CUSTOM_STOP_WORDS = stops
    return stops


def tokenize_without_stopwords(text: str) -> Tuple[str, List[str]]:
    text = (text or "").lower()
    tokens = [m.group(0) for m in TOKEN_PATTERN.finditer(text)]
    stops = load_custom_stopwords()
    filtered = [tok for tok in tokens if tok not in stops and len(tok) > 1]
    return " ".join(filtered), filtered


def apriori_frequent_itemsets(
    transactions: List[List[str]],
    min_support_ratio: float,
    max_size: int,
    top_k: int,
) -> List[Tuple[Tuple[str, ...], int]]:
    if not transactions:
        return []
    transactions_sets = [set(t) for t in transactions if t]
    transactions_sets = [t for t in transactions_sets if t]
    if not transactions_sets:
        return []
    n = len(transactions_sets)
    min_support = max(2, int(math.ceil(min_support_ratio * n)))

    # L1
    item_counts = Counter()
    for txn in transactions_sets:
        for item in txn:
            item_counts[(item,)] += 1
    current_freq = {items: cnt for items, cnt in item_counts.items() if cnt >= min_support}
    if not current_freq:
        return []

    freq_by_size: Dict[int, Dict[Tuple[str, ...], int]] = {1: current_freq}
    all_freq: List[Tuple[Tuple[str, ...], int]] = list(current_freq.items())
    k = 1

    while k < max_size:
        prev_freq = freq_by_size.get(k)
        if not prev_freq:
            break
        prev_keys = list(prev_freq.keys())
        candidates: set[Tuple[str, ...]] = set()
        prev_key_sets = [set(key) for key in prev_keys]
        for i in range(len(prev_keys)):
            for j in range(i + 1, len(prev_keys)):
                union_set = prev_key_sets[i] | prev_key_sets[j]
                if len(union_set) != k + 1:
                    continue
                candidate = tuple(sorted(union_set))
                # prune using Apriori property
                if all(tuple(sorted(sub)) in prev_freq for sub in combinations(candidate, k)):
                    candidates.add(candidate)
        if not candidates:
            break
        counts = Counter()
        for txn in transactions_sets:
            for cand in candidates:
                if set(cand).issubset(txn):
                    counts[cand] += 1
        next_freq = {cand: cnt for cand, cnt in counts.items() if cnt >= min_support}
        if not next_freq:
            break
        k += 1
        freq_by_size[k] = next_freq
        all_freq.extend(next_freq.items())

    # sort by (support desc, length desc, lexicographic)
    all_freq.sort(key=lambda item: (-item[1], -len(item[0]), item[0]))
    return all_freq[:top_k]


def read_texts_ids_titles(jsonl_path: Path):
    texts, ids, titles, token_lists = [], [], [], []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            raw_text = rec.get("processed_content") or ""
            _, tokens_text = tokenize_without_stopwords(raw_text)
            raw_title = rec.get("title") or ""
            _, tokens_title = tokenize_without_stopwords(raw_title)
            merged_tokens = tokens_title + tokens_text
            merged_text = " ".join(merged_tokens)
            if len(raw_text) < MIN_TEXT_CHARS or not merged_tokens:
                continue
            texts.append(merged_text)
            ids.append(rec.get("id") or "")
            titles.append(rec.get("title") or "")
            token_lists.append(merged_tokens)
    return texts, np.array(ids), np.array(titles), token_lists


def compute_cluster_top_terms(labels: np.ndarray, token_lists: List[List[str]], top_n: int = 10) -> Dict[int, List[str]]:
    buckets: Dict[int, Counter] = defaultdict(Counter)
    transactions_by_cluster: Dict[int, List[List[str]]] = defaultdict(list)
    for lbl, tokens in zip(labels, token_lists):
        if lbl is None:
            continue
        lbl_int = int(lbl)
        if lbl_int < 0:
            continue
        if tokens:
            buckets[lbl_int].update(tokens)
            transactions_by_cluster[lbl_int].append(tokens)
    topics: Dict[int, List[str]] = {}
    for lbl, counter in buckets.items():
        transactions = transactions_by_cluster.get(lbl, [])
        itemsets = apriori_frequent_itemsets(
            transactions,
            min_support_ratio=APRIORI_MIN_SUPPORT_RATIO,
            max_size=APRIORI_MAX_SIZE,
            top_k=APRIORI_TOP_K,
        )
        if itemsets:
            topics[lbl] = [" + ".join(items) for items, _cnt in itemsets]
        else:
            # fallback to top individual terms if Apriori found nothing
            topics[lbl] = [term for term, _ in counter.most_common(APRIORI_FALLBACK_TOP_K)]
    return topics


def build_or_load_embeddings(texts):
    if EMBEDDINGS_PATH.exists():
        print(f"[i] Loading cached embeddings: {EMBEDDINGS_PATH}")
        X = np.load(EMBEDDINGS_PATH)
    else:
        print(f"[i] Loading SBERT model: {SBERT_MODEL_NAME}")
        model = SentenceTransformer(SBERT_MODEL_NAME)
        print(f"[i] Encoding {len(texts)} documents...")
        X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
        np.save(EMBEDDINGS_PATH, X)
        print(f"[i] Saved embeddings: {EMBEDDINGS_PATH}")

    X = normalize(X, norm="l2", axis=1)
    try:
        np.save(EMBEDDINGS_NORM_PATH, X)
    except Exception:
        pass
    return X


def maybe_umap(X: np.ndarray) -> np.ndarray:
    if not UMAP_ENABLED:
        return X
    try:
        import umap
    except Exception:
        print("[warn] UMAP not installed, skipping pre-reduction")
        return X
    print("[i] UMAP pre-reduction...")
    reducer = umap.UMAP(n_components=UMAP_DIM, metric=UMAP_METRIC, random_state=42)
    Z = reducer.fit_transform(X)
    joblib.dump(reducer, OUT_DIR / "umap_reducer.joblib")
    print(f"[i] Saved UMAP reducer model to: {OUT_DIR / 'umap_reducer.joblib'}")


    Z = normalize(Z, norm="l2", axis=1)
    print(f"[i] UMAP shape: {Z.shape}")
    return Z


def run_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int | None, method: str, eps: float):

    kwargs = dict(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method=method,
                  prediction_data=True)
    if min_samples is not None:
        kwargs["min_samples"] = min_samples
    if eps and eps > 0:
        kwargs["cluster_selection_epsilon"] = eps
    clusterer = hdbscan.HDBSCAN(**kwargs)
    labels = clusterer.fit_predict(X)
    return clusterer, labels


def quick_metrics(X: np.ndarray, labels: np.ndarray):
    n = len(labels)
    noise = int((labels == -1).sum())
    clustered = n - noise
    k = len(np.unique(labels[labels != -1]))
    sil = None
    if clustered > 0 and k > 1:
       
        sil = float(silhouette_score(X[labels != -1], labels[labels != -1], metric="euclidean"))
    noise_rate = noise / n if n else 1.0
    coverage = clustered / n if n else 0.0
    return dict(n=n, noise=noise, clustered=clustered, k=k, silhouette=sil, noise_rate=noise_rate, coverage=coverage)


def sampled_param_search(X: np.ndarray, csv_logger: Optional[CsvLogger] = None):
    if not PARAM_SEARCH:
        return None
    Xs = X
    if PARAM_SAMPLE_SIZE and X.shape[0] > PARAM_SAMPLE_SIZE:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=PARAM_SAMPLE_SIZE, replace=False)
        Xs = X[idx]
        print(f"[i] Param search on sample: {Xs.shape[0]} / {X.shape[0]}")

    # Build dynamic ranges based on dataset size
    n = Xs.shape[0]
    dyn_mcs = sorted({
        *MIN_CLUSTER_SIZE_RANGE,
        max(5, n // 2000),
        max(10, n // 1000),
        max(20, n // 500),
    })

    def score(m):
       
        sil = m["silhouette"] if m["silhouette"] is not None else -1.0
        coverage = m["coverage"]
        k = m["k"]
        k_pref = min(k / 50.0, 1.0)
        return 0.7 * sil + 0.3 * coverage + 0.1 * k_pref

    best = None
    best_score = -1e9

    
    for mcs in dyn_mcs:
        for ms in MIN_SAMPLES_RANGE:
            for method in METHODS:
                for eps in EPS_RANGE:
                    try:
                       
                        metric = "euclidean"
                        _, labels = run_hdbscan(Xs, mcs, ms, method, eps)
                        metr = quick_metrics(Xs, labels) 
                        sc = score(metr)

                        if csv_logger:
                            log_data = {
                                "score": sc,
                                "silhouette": metr["silhouette"],
                                "num_clusters": metr["k"],
                                "clustered_percentage": metr["coverage"],
                                "noise_rate": metr["noise_rate"],
                                "min_cluster_size": mcs,
                                "min_samples": ms,
                                "method": method,
                                "eps": eps,
                                "metric": metric,
                            }
                            csv_logger.log(log_data)

                        if sc > best_score:
                            best_score = sc
                            best = dict(min_cluster_size=mcs, min_samples=ms, method=method, eps=eps, metric=metric,
                                        metrics=metr)
                            sil_txt = "NA" if metr["silhouette"] is None else f"{metr['silhouette']:.4f}"
                            print(
                                f"  best so far: score={best_score:.4f} sil={sil_txt} k={metr['k']} coverage={metr['coverage']:.2f} noise_rate={metr['noise_rate']:.2f} params={{'min_cluster_size':{mcs},'min_samples':{ms},'method':'{method}','eps':{eps},'metric':'{metric}'}}")

                        if (metr["silhouette"] is not None and metr["silhouette"] >= 0.8 and
                                metr["coverage"] >= 0.9 and metr["k"] <= 50):
                            return best
                    except Exception as e:
                        print(f"  skip error: {e}")
                        continue
    return best


def main():
    print("=" * 60)
    print("SBERT + HDBSCAN (lite)")
    print("=" * 60)

    print("\n[1] Loading texts...")
    texts, ids, titles, token_lists = read_texts_ids_titles(INPUT_JSONL)
    print(f"[i] {len(texts)} documents after filtering")

    print("\n[2] Building/loading embeddings...")
    X = build_or_load_embeddings(texts)

    print("\n[3] UMAP (optional)...")
    Z = maybe_umap(X)

    print("\n[4] Parameter search (sampled)...")
    csv_logger = None
    if PARAM_SEARCH:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        LOG_CSV_PATH = OUT_DIR / f"hdbscan_search_log_{ts}.csv"
        print(f"[i] Logging parameter search to: {LOG_CSV_PATH}")
        log_fieldnames = [
            "score", "silhouette", "num_clusters", "clustered_percentage", "noise_rate",
            "min_cluster_size", "min_samples", "method", "eps", "metric"
        ]
        csv_logger = CsvLogger(LOG_CSV_PATH, log_fieldnames)

    params = sampled_param_search(Z, csv_logger)
    if params is None:
       
        params = dict(min_cluster_size=5, min_samples=20, method="eom", eps=0.0, metric="euclidean")

   
    try:
        to_dump = dict(params={k: v for k, v in params.items() if k != "metrics"}, metrics=params.get("metrics"))
        with (OUT_DIR / "hdbscan_params.json").open("w", encoding="utf-8") as f:
            json.dump(to_dump, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[warn] Failed to save params json: {e}")

    
    final_metric = params.get("metric", "euclidean")
    print(
        f"[i] Params used: {{'min_cluster_size': {params['min_cluster_size']}, 'min_samples': {params['min_samples']}, 'method': '{params['method']}', 'eps': {params['eps']}, 'metric': '{final_metric}'}}")

    print("\n[5] Clustering on full data...")
    clusterer, labels = run_hdbscan(Z, params["min_cluster_size"], params["min_samples"], params["method"],
                                    params["eps"])
    joblib.dump(clusterer, OUT_DIR / "hdbscan_clusterer.joblib")
    print(f"[i] Saved HDBSCAN clusterer model to: {OUT_DIR / 'hdbscan_clusterer.joblib'}")
    metr = quick_metrics(Z, labels)  # Will use default 'euclidean'

    sil_txt = "NA" if metr["silhouette"] is None else f"{metr['silhouette']:.4f}"
    print(f"silhouette={sil_txt}  k={metr['k']}  coverage={metr['coverage']:.2f}  noise_rate={metr['noise_rate']:.2f}")

    # Save metrics summary
    try:
        with (OUT_DIR / "hdbscan_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metr, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[warn] Failed to save metrics json: {e}")

    print("\n[6] Saving artifacts...")
    np.save(CLUSTER_LABELS_PATH, labels)
    np.save(DOC_IDS_PATH, ids)
    np.save(DOC_TITLES_PATH, titles)

    cluster_topics = compute_cluster_top_terms(labels, token_lists)
    try:
        with CLUSTER_TOP_TERMS_PATH.open("w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in cluster_topics.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[i] saved: {CLUSTER_TOP_TERMS_PATH}")
    except Exception as exc:
        print(f"[warn] failed to save cluster top terms: {exc}")

    print(f"[i] saved: {CLUSTER_LABELS_PATH}\n[i] saved: {DOC_IDS_PATH}\n[i] saved: {DOC_TITLES_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()