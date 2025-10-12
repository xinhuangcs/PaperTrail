import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

# Config
TFIDF_MATRIX_PATH = '/Users/jasonh/Desktop/02807/PaperTrail/data/tf_idf/tfidf_matrix.npz'
LSA_OUTPUT_PATH   = '/Users/jasonh/Desktop/02807/PaperTrail/data/lsa/lsa_reduced.npz'

# Number of latent dimensions for LSA
N_COMPONENTS = 100  # (Adjustable like 100 or 200)

def main():
    # 1) Load the TF-IDF sparse matrix
    X = sparse.load_npz(TFIDF_MATRIX_PATH)
    print(f"[i] TF-IDF matrix loaded: shape={X.shape}, nnz={X.nnz}")

    # 2) Perform LSA dimensionality reduction using TruncatedSVD
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    X_reduced = svd.fit_transform(X)
    print(f"[i] LSA reduction done: new shape={X_reduced.shape}")

    # print explained variance ratio sum
    if hasattr(svd, 'explained_variance_ratio_'):
        variance_ratio_sum = svd.explained_variance_ratio_.sum()
        print(f"[i] Explained variance by top {N_COMPONENTS} components: {variance_ratio_sum:.2%}")

    # 3)Save the reduced matrix to a compressed NPZ file
    np.savez_compressed(LSA_OUTPUT_PATH, X_reduced=X_reduced)
    print(f"[i] LSA reduced matrix saved to {LSA_OUTPUT_PATH}")

if __name__ == "__main__":
    main()