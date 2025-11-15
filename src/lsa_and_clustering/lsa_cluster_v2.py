import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
#Config
ROOT_DIR = Path(__file__).resolve().parents[2]
LSA_INPUT_PATH = ROOT_DIR / "data" / "lsa" / "lsa_reduced.npz"
CLUSTER_LABELS_PATH = ROOT_DIR / "data" / "lsa" / "cluster_labels.npy"

K_FIXED = 40

def main():
    # 1) Load LSA reduced matrix
    data = np.load(LSA_INPUT_PATH)
    X = data['X_reduced']
    print(f"[i] Loaded LSA matrix: shape={X.shape}")

    print(f"[i] Running MiniBatchKMeans with fixed K={K_FIXED}")
    kmeans = MiniBatchKMeans(n_clusters=K_FIXED, init='k-means++', n_init=10, random_state=42)
    final_labels = kmeans.fit_predict(X)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, final_labels)
    print(f"[i] Inertia={inertia:.2f}, Silhouette={silhouette:.4f}")

    np.save(CLUSTER_LABELS_PATH, final_labels)
    print(f"[i] Cluster labels saved to: {CLUSTER_LABELS_PATH}")

    # 4)Output Final clustering results:
    unique, counts = np.unique(final_labels, return_counts=True)
    print("Final clustering results:")
    for cid, count in zip(unique, counts):
        print(f" - Cluster {cid}: {count} papers")

if __name__ == "__main__":
    main()
