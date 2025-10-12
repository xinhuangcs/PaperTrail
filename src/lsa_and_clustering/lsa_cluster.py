# Perform K-Means clustering on the LSA dimensionality reduction results

import numpy as np
from sklearn.cluster import KMeans

#Config
LSA_INPUT_PATH      = '/Users/jasonh/Desktop/02807/PaperTrail/data/lsa/lsa_reduced.npz'
CLUSTER_LABELS_PATH = '/Users/jasonh/Desktop/02807/PaperTrail/data/lsa/cluster_labels.npy'
#Number of clusters for KMeans
N_CLUSTERS = 15

def main():
    # 1) Load the LSA reduced data matrix
    data = np.load(LSA_INPUT_PATH)
    X_reduced = data['X_reduced']
    print(f"[i] LSA matrix loaded for clustering: shape={X_reduced.shape}")

    # 2) Perform K-Means clustering on the data
    kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', n_init=10, random_state=42)
    # use MiniBatchKMeans for large datasets
    kmeans.fit(X_reduced)
    labels = kmeans.labels_
    print(f"[i] KMeans clustering done. Inertia: {kmeans.inertia_:.2f}")

    # 3) Save cluster labels to file
    np.save(CLUSTER_LABELS_PATH, labels)
    print(f"[i] Cluster labels saved to {CLUSTER_LABELS_PATH}")

    # Print the number of documents in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    print("[i] Cluster distribution:")
    for cid, count in zip(unique, counts):
        print(f" - Cluster {cid}: {count} documents")

if __name__ == "__main__":
    main()