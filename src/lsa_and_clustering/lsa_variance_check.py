#it is only for check.


from pathlib import Path
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

# 1) Config
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG = {
    # input TF-IDF matrix (.npz, CSR)
    "tfidf_matrix_path": ROOT_DIR / "data" / "tf_idf" / "tfidf_matrix.npz",

    # where to save check results
    "plot_path":  ROOT_DIR / "data" / "lsa" / "v2-lsa_explained_variance.png",
    "csv_path":   ROOT_DIR / "data" / "lsa" / "v2-lsa_explained_variance.csv",

    # check settings
    "probe_max_components": 800,   # try up to this many components
    "target_cum_var": 0.85,        # fallback target cumulative variance
    "pref_range": (400, 600),      # preferred range for CS abstracts
    "random_state": 42,
    "svd_iter": 8,                 # 7~10 is stable
    "show_progress": True,
}
CONFIG["plot_path"].parent.mkdir(parents=True, exist_ok=True)
CONFIG["csv_path"].parent.mkdir(parents=True, exist_ok=True)


def variance_probe_and_suggest_dim(X):
    #Fit a check SVD and suggest a good LSA dimension
    n_features = X.shape[1]
    n_probe = int(min(CONFIG["probe_max_components"], max(2, n_features - 1)))
    print(f"[check] Fitting TruncatedSVD with n_components={n_probe} (features={n_features})")

    svd = TruncatedSVD(
        n_components=n_probe,
        random_state=CONFIG["random_state"],
        algorithm="randomized",
        n_iter=CONFIG["svd_iter"],
    )
    svd.fit(X)

    # explained variance per component and cumulative
    explained = svd.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    xs = np.arange(1, len(cumulative) + 1)

    # save CSV
    with open(CONFIG["csv_path"], "w", encoding="utf-8") as f:
        f.write("component,explained_variance_ratio,cumulative\n")
        for i, (r, c) in enumerate(zip(explained, cumulative), start=1):
            f.write(f"{i},{float(r)},{float(c)}\n")
    print(f"[check] Saved CSV -> {CONFIG['csv_path']}")

    # plot curve
    plt.figure(figsize=(8, 5))
    plt.plot(xs, cumulative, marker="o", linewidth=1)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("LSA Variance check")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CONFIG["plot_path"], dpi=160)
    print(f"[check] Saved plot -> {CONFIG['plot_path']}")

    # print checkpoints
    for d in [100, 200, 300, 400, 500, 600, 800]:
        if d <= len(cumulative):
            print(f"[check] {d} comps -> cumulative = {cumulative[d-1]:.4f}")

    # try knee (optional)
    best_knee = None
    try:
        from kneed import KneeLocator
        knee = KneeLocator(xs, cumulative, curve="concave", direction="increasing")
        if knee.knee is not None:
            best_knee = int(knee.knee)
    except Exception as e:
        print(f"[check] Knee detection skipped: {e}")

    # fallback: smallest d reaching target cumulative variance
    by_target = int(np.searchsorted(cumulative, CONFIG["target_cum_var"])) + 1
    by_target = min(by_target, n_probe)

    # choose candidate, then clamp to preferred range
    candidate = best_knee if best_knee else by_target
    lo, hi = CONFIG["pref_range"]
    suggested = max(lo, min(hi, candidate))
    suggested = min(suggested, n_probe)

    print(f"[check] knee={best_knee}, by_target={by_target}, "
          f"suggested_dim(after clamp)={suggested}")
    return suggested


def main():
    # load TF-IDF sparse matrix
    X = sparse.load_npz(CONFIG["tfidf_matrix_path"])
    print(f"TF-IDF loaded: shape={X.shape}, nnz={X.nnz}")

    # run check and print suggestion
    suggested_dim = variance_probe_and_suggest_dim(X)
    print("\n[RESULT]")
    print(f"- Suggested LSA dimension: {suggested_dim} "
          f"(preferred range {CONFIG['pref_range'][0]}–{CONFIG['pref_range'][1]})")
    print(f"- See curve: {CONFIG['plot_path']}")
    print(f"- See table: {CONFIG['csv_path']}")


if __name__ == "__main__":
    main()
