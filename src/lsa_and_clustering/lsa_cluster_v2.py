import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

#Config
LSA_INPUT_PATH = '/Users/jasonh/Desktop/02807/PaperTrail/data/lsa/lsa_reduced.npz'
CLUSTER_LABELS_PATH = '/Users/jasonh/Desktop/02807/PaperTrail/data/lsa/cluster_labels_auto.npy'

#Try K from 5 to 30
K_RANGE = range(5, 31, 5)

def main():
    # 1) Load LSA reduced matrix
    data = np.load(LSA_INPUT_PATH)
    X = data['X_reduced']
    print(f"[i] Loaded LSA matrix: shape={X.shape}")

    inertias = []
    silhouettes = []

    print("Evaluating clustering performance under different K values...")

    for k in K_RANGE:
        print(f"Evaluating k={k}")
        kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, labels)
        inertias.append(inertia)
        silhouettes.append(silhouette)
        print(f" - K={k}: Inertia={inertia:.2f}, Silhouette={silhouette:.4f}")

    # 2) Find K with highest silhouette
    best_k = K_RANGE[np.argmax(silhouettes)]
    print(f"Recommended optimal K value: {best_k} (highest silhouette score)")

    # 3)Cluster again using best K
    final_kmeans = MiniBatchKMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
    final_labels = final_kmeans.fit_predict(X)
    np.save(CLUSTER_LABELS_PATH, final_labels)
    print(f"Final cluster labels have been saved to: {CLUSTER_LABELS_PATH}")

    # 4)Output Final clustering results:
    unique, counts = np.unique(final_labels, return_counts=True)
    print("Final clustering results:")
    for cid, count in zip(unique, counts):
        print(f" - Cluster {cid}: {count} papers")

    # 5)Plot Inertia and Silhouette curves (Elbow curve)
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(list(K_RANGE), inertias, marker='o')
    plt.title('Elbow Curve (Inertia vs K)')
    plt.xlabel('K')
    plt.ylabel('Inertia')

    plt.subplot(1, 2, 2)
    plt.plot(list(K_RANGE), silhouettes, marker='o', color='green')
    plt.title('Silhouette Score vs K')
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.savefig('kmeans_eval.png')
    print("Evaluation plot saved as kmeans_eval.png")

if __name__ == "__main__":
    main()
