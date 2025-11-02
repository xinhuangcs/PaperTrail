import json
import joblib
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

# Minimal, readable SBERT + HDBSCAN pipeline (lite)

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = REPO_ROOT / "data" / "merge" / "arxiv-cs-data-with-citations_merged_first_50000.json"

OUT_DIR = REPO_ROOT / "data" / "sbert_hdbscan_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = OUT_DIR / "sbert_embeddings.npy"
EMBEDDINGS_NORM_PATH = OUT_DIR / "sbert_embeddings_norm.npy"
CLUSTER_LABELS_PATH = OUT_DIR / "hdbscan_labels.npy"
DOC_IDS_PATH = OUT_DIR / "doc_ids.npy"
DOC_TITLES_PATH = OUT_DIR / "doc_titles.npy"

# Model and filters
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
MIN_TEXT_CHARS = 100

# UMAP pre-reduction (fixed small config)
UMAP_ENABLED = True
UMAP_DIM = 50
UMAP_METRIC = "cosine"  # UMAP *does* support cosine, this is fine

# Sampled parameter search (broader and smarter)
PARAM_SEARCH = False
PARAM_SAMPLE_SIZE = 10000
# Baseline ranges (dynamic ranges are built inside search as well)
MIN_CLUSTER_SIZE_RANGE = [5, 10, 20, 30, 50, 100]
MIN_SAMPLES_RANGE = [1, 2, 5, 10, 15]
METHODS = ["eom", "leaf"]
EPS_RANGE = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
TARGET_SILHOUETTE = 0.5


def read_texts_ids_titles(jsonl_path: Path):
    texts, ids, titles = [], [], []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = rec.get("processed_content") or ""
            if len(text) < MIN_TEXT_CHARS:
                continue
            texts.append(text)
            ids.append(rec.get("id") or "")
            titles.append(rec.get("title") or "")
    return texts, np.array(ids), np.array(titles)


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
    # Key step: Normalization
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
    # Re-normalize after UMAP

    Z = normalize(Z, norm="l2", axis=1)
    print(f"[i] UMAP shape: {Z.shape}")
    return Z


def run_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int | None, method: str, eps: float):
    # Metric is hardcoded to 'euclidean' because data is normalized
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
        # Metric is hardcoded to 'euclidean' because data is normalized
        sil = float(silhouette_score(X[labels != -1], labels[labels != -1], metric="euclidean"))
    noise_rate = noise / n if n else 1.0
    coverage = clustered / n if n else 0.0
    return dict(n=n, noise=noise, clustered=clustered, k=k, silhouette=sil, noise_rate=noise_rate, coverage=coverage)


def sampled_param_search(X: np.ndarray):
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
        # Higher silhouette and coverage are better; moderate number of clusters preferred
        sil = m["silhouette"] if m["silhouette"] is not None else -1.0
        coverage = m["coverage"]
        k = m["k"]
        k_pref = min(k / 50.0, 1.0)
        return 0.7 * sil + 0.3 * coverage + 0.1 * k_pref

    best = None
    best_score = -1e9

    # Removed the 'metric' loop, as 'euclidean' is all we need
    for mcs in dyn_mcs:
        for ms in MIN_SAMPLES_RANGE:
            for method in METHODS:
                for eps in EPS_RANGE:
                    try:
                        # Hardcode metric to 'euclidean'
                        metric = "euclidean"
                        _, labels = run_hdbscan(Xs, mcs, ms, method, eps)
                        metr = quick_metrics(Xs, labels)  # Will also use 'euclidean'
                        sc = score(metr)
                        if sc > best_score:
                            best_score = sc
                            best = dict(min_cluster_size=mcs, min_samples=ms, method=method, eps=eps, metric=metric,
                                        metrics=metr)
                            sil_txt = "NA" if metr["silhouette"] is None else f"{metr['silhouette']:.4f}"
                            print(
                                f"  best so far: score={best_score:.4f} sil={sil_txt} k={metr['k']} coverage={metr['coverage']:.2f} noise_rate={metr['noise_rate']:.2f} params={{'min_cluster_size':{mcs},'min_samples':{ms},'method':'{method}','eps':{eps},'metric':'{metric}'}}")
                        # Early stop if good enough
                        if (metr["silhouette"] is not None and metr["silhouette"] >= TARGET_SILHOUETTE and
                                metr["coverage"] >= 0.6 and metr["k"] >= 10):
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
    texts, ids, titles = read_texts_ids_titles(INPUT_JSONL)
    print(f"[i] {len(texts)} documents after filtering")

    print("\n[2] Building/loading embeddings...")
    X = build_or_load_embeddings(texts)

    print("\n[3] UMAP (optional)...")
    Z = maybe_umap(X)

    print("\n[4] Parameter search (sampled)...")
    params = sampled_param_search(Z)
    if params is None:
        # Default to euclidean
        params = dict(min_cluster_size=5, min_samples=20, method="eom", eps=0.0, metric="euclidean")

    # Persist chosen params and (if present) metrics
    try:
        to_dump = dict(params={k: v for k, v in params.items() if k != "metrics"}, metrics=params.get("metrics"))
        with (OUT_DIR / "hdbscan_params.json").open("w", encoding="utf-8") as f:
            json.dump(to_dump, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[warn] Failed to save params json: {e}")

    # Ensure metric is set for the final run
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
    print(f"[i] saved: {CLUSTER_LABELS_PATH}\n[i] saved: {DOC_IDS_PATH}\n[i] saved: {DOC_TITLES_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()