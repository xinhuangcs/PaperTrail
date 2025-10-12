#Reduce LSA results to two dimensions and plot a scatter plot.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#Config paths
LSA_INPUT_PATH       = '/Users/jasonh/Desktop/02807/PaperTrail/data/lsa/lsa_reduced.npz'       # LSA降维矩阵文件路径
CLUSTER_LABELS_PATH  = '/Users/jasonh/Desktop/02807/PaperTrail/data/lsa/cluster_labels.npy'    # 聚类标签文件路径 (可选)
OUTPUT_FIG_PATH      = '/Users/jasonh/Desktop/02807/PaperTrail/data/lsa/lsa_scatter.png'       # 输出图像文件路径

USE_TSNE = True  # True: t-SNE, False: PCA for 2D projection

def main():
    # 1) Load the LSA matrix
    data = np.load(LSA_INPUT_PATH)
    X_reduced = data['X_reduced']
    print(f"[i] Loaded LSA matrix: shape={X_reduced.shape}")

    # 2) Load cluster labels if available
    labels = None
    try:
        labels = np.load(CLUSTER_LABELS_PATH)
        print(f"[i] Loaded cluster labels: total {len(labels)} labels")
    except FileNotFoundError:
        print("[i] No cluster labels file found, proceeding without labels.")

    # 3) If dataset is large, sample a subset for visualization
    N = X_reduced.shape[0]
    if N > 10000:
        idx = np.random.choice(N, size=10000, replace=False)
        X_plot = X_reduced[idx]
        labels_plot = labels[idx] if labels is not None else None
        print(f"[i] Data too large, sampled {X_plot.shape[0]} points out of {N} for plotting")
    else:
        X_plot = X_reduced
        labels_plot = labels

    # 4) Use t-SNE or PCA to reduce data to 2 dimensions for plotting
    if USE_TSNE:
        print("[i] Running t-SNE for 2D embedding...")
        X_2d = TSNE(n_components=2, init='random', learning_rate='auto', perplexity=30, random_state=42).fit_transform(X_plot)
    else:
        print("[i] Running PCA for 2D projection...")
        X_2d = PCA(n_components=2, random_state=42).fit_transform(X_plot)

    # 5) Plot the scatter diagram
    plt.figure(figsize=(8, 6))
    if labels_plot is not None:
        # Color points by cluster label if available
        num_clusters = len(np.unique(labels_plot))
        if num_clusters <= 10:
            cmap = 'tab10'
        elif num_clusters <= 20:
            cmap = 'tab20'
        else:
            cmap = 'Accent' #use a palette that supports many colors
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_plot, cmap=cmap, s=5, alpha=0.7)
        plt.colorbar(scatter, label='Cluster')  # =color bar shows cluster index
        plt.title('LSA Document Embeddings (colored by cluster)')
    else:
        # Plot without cluster labels, single-color points
        plt.scatter(X_2d[:, 0], X_2d[:, 1], color='blue', s=5, alpha=0.7)
        plt.title('LSA Document Embeddings')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_PATH)
    print(f"[i] Scatter plot saved to {OUTPUT_FIG_PATH}")

if __name__ == "__main__":
    main()