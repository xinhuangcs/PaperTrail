
"""
evaluate LSA+KMeans 与 SBERT+HDBSCAN
Davies–Bouldin / Silhouette[cosine] / Calinski–Harabasz
"""

from pathlib import Path
import numpy as np
import json, time, warnings
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import normalize

# Config
ROOT = Path(__file__).resolve().parents[2]  # /Users/jasonh/Desktop/02807/PaperTrail
DATA = ROOT / "data"
METRICS_DIR = DATA / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# LSA + KMeans
LSA_REDUCED_NPZ  = DATA / "lsa" / "lsa_reduced.npz"
KMEANS_LABELS_NPY = DATA / "lsa" / "cluster_labels.npy"

# SBERT + HDBSCAN
HDBSCAN_OUT_DIR = DATA / "sbert_hdbscan_test"
SBERT_EMB_NORM  = HDBSCAN_OUT_DIR / "sbert_embeddings_norm.npy"
SBERT_EMB       = HDBSCAN_OUT_DIR / "sbert_embeddings.npy"
HDBSCAN_LABELS  = HDBSCAN_OUT_DIR / "hdbscan_labels.npy"


def _sample_Xy(X, y, max_n=50000, seed=42):
    if X.shape[0] <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_n, replace=False)
    return X[idx], y[idx]

def _safe_scores(X, y, name, max_n=50000):
    uniq = np.unique(y)
    if uniq.size < 2 or X.shape[0] < 10:
        warnings.warn(f"[{name}] too few clusters/samples -> metrics=None")
        return {"DBI": None, "Silhouette": None, "CH": None}

    X_eval, y_eval = _sample_Xy(X, y, max_n=max_n)
    out = {}
    try:
        out["DBI"] = float(davies_bouldin_score(X_eval, y_eval))
    except Exception as e:
        warnings.warn(f"[{name}] DBI failed: {e}")
        out["DBI"] = None
    try:
        out["Silhouette"] = float(silhouette_score(X_eval, y_eval, metric="cosine"))
    except Exception as e:
        warnings.warn(f"[{name}] Silhouette failed: {e}")
        out["Silhouette"] = None
    try:
        out["CH"] = float(calinski_harabasz_score(X_eval, y_eval))
    except Exception as e:
        warnings.warn(f"[{name}] CH failed: {e}")
        out["CH"] = None
    return out

def _load_lsa_array(npz_path: Path) -> np.ndarray:
    """兼容性读取 lsa_reduced.npz（取常见键或第一个数组）"""
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path) as z:
        for k in ["X_reduced", "X", "lsa", "data", "arr_0"]:
            if k in z.files:
                return z[k]
        # 回退：取第一个
        return z[z.files[0]]

# ---------------- 评估 LSA + KMeans ----------------
def eval_lsa_kmeans():
    print("\n[1] Evaluating LSA + KMeans ...")
    if not LSA_REDUCED_NPZ.exists() or not KMEANS_LABELS_NPY.exists():
        print(f"[skip] Missing: {LSA_REDUCED_NPZ} or {KMEANS_LABELS_NPY}")
        return None

    X = _load_lsa_array(LSA_REDUCED_NPZ)
    y = np.load(KMEANS_LABELS_NPY)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Size mismatch: X={X.shape[0]} vs y={y.shape[0]}")

    metrics = _safe_scores(X, y, "LSA+KMeans")
    result = {
        "method": "LSA+KMeans",
        "n_samples": int(X.shape[0]),
        "n_clusters": int(np.unique(y).size),
        "timestamp": int(time.time()),
        **metrics
    }
    out = METRICS_DIR / "kmeans_lsa_metrics.json"
    json.dump(result, open(out, "w"), indent=2)
    print(f"[ok] Saved {out}")
    return result

# SBERT + HDBSCAN
def eval_hdbscan_sbert():
    print("\n[2] Evaluating SBERT + HDBSCAN ...")
    if not HDBSCAN_LABELS.exists() or not (SBERT_EMB_NORM.exists() or SBERT_EMB.exists()):
        print(f"[skip] Missing: {HDBSCAN_LABELS} or SBERT embeddings")
        return None

    if SBERT_EMB_NORM.exists():
        Z = np.load(SBERT_EMB_NORM)
        Z = np.asarray(Z)
    else:
        Z = np.load(SBERT_EMB)
        Z = normalize(Z, norm="l2")

    y = np.load(HDBSCAN_LABELS)
    if Z.shape[0] != y.shape[0]:
        raise ValueError(f"Size mismatch: Z={Z.shape[0]} vs y={y.shape[0]}")

    mask = (y != -1)
    Z_in, y_in = Z[mask], y[mask]
    if Z_in.size == 0:
        warnings.warn("[SBERT+HDBSCAN] all points are noise; metrics=None")
        metrics = {"DBI": None, "Silhouette": None, "CH": None}
        n_clusters_ex_noise = 0
    else:
        metrics = _safe_scores(Z_in, y_in, "SBERT+HDBSCAN")
        n_clusters_ex_noise = int(np.unique(y_in).size)

    result = {
        "method": "SBERT(+UMAP)+HDBSCAN",
        "n_samples_total": int(Z.shape[0]),
        "n_noise": int(np.sum(y == -1)),
        "noise_ratio": float(np.mean(y == -1)),
        "n_clusters_excluding_noise": n_clusters_ex_noise,
        "timestamp": int(time.time()),
        **metrics
    }
    out = METRICS_DIR / "hdbscan_sbert_metrics.json"
    json.dump(result, open(out, "w"), indent=2)
    print(f"[ok] Saved {out}")
    return result

def main():
    print("=== Clustering Evaluation Script ===")
    r1 = eval_lsa_kmeans()
    r2 = eval_hdbscan_sbert()

    print("\n=== Summary ===")
    for r in (r1, r2):
        if not r:
            continue
        print(f"\n[{r['method']}]")
        for k, v in r.items():
            if k in ("method", "timestamp"):
                continue
            print(f"  {k}: {v}")
    print("\nAll results saved in:", METRICS_DIR)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
