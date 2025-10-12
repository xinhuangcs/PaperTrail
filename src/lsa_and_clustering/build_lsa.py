
"""
Build LSA (TruncatedSVD) and save both reduced matrix + model + components
"""

import os
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import joblib

# Config
TFIDF_MATRIX_PATH = '/Users/jasonh/Desktop/02807/PaperTrail/data/tf_idf/tfidf_matrix.npz'
LSA_DIR           = '/Users/jasonh/Desktop/02807/PaperTrail/data/lsa'
LSA_OUTPUT_PATH   = os.path.join(LSA_DIR, 'lsa_reduced.npz')
SVD_MODEL_PATH    = os.path.join(LSA_DIR, 'svd_model.joblib')
LSA_COMP_PATH     = os.path.join(LSA_DIR, 'lsa_components.npz')

# Number of latent dimensions for LSA
N_COMPONENTS = 100

RANDOM_STATE = 42
N_ITER = 5

def bytes_to_mb(n_bytes: int) -> float:
    return n_bytes / (1024.0 * 1024.0)

def main():
    os.makedirs(LSA_DIR, exist_ok=True)

    # 1) Load the TF-IDF sparse matrix
    X = sparse.load_npz(TFIDF_MATRIX_PATH)  # shape: (N_docs, V)
    print(f"[i] TF-IDF matrix loaded: shape={X.shape}, nnz={X.nnz}")

    # 2) Perform LSA dimensionality reduction using TruncatedSVD
    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        n_iter=N_ITER,
        random_state=RANDOM_STATE
    )
    X_reduced = svd.fit_transform(X)  # dense matrix: (N_docs, N_COMPONENTS)
    print(f"LSA reduction done: new shape={X_reduced.shape}")

    # explained variance ratio
    if hasattr(svd, 'explained_variance_ratio_') and svd.explained_variance_ratio_ is not None:
        variance_ratio_sum = svd.explained_variance_ratio_.sum()
        print(f"Explained variance by top {N_COMPONENTS} components: {variance_ratio_sum:.2%}")

    # Memory footprint estimation for dense reduced matrix
    est_bytes = X_reduced.shape[0] * X_reduced.shape[1] * 4  # float32 ~4B; sklearn默认为float64(~8B)
    # X_reduced = X_reduced.astype(np.float32)
    print(f"Estimated dense matrix size ~ {bytes_to_mb(est_bytes):.1f} MB (float32 equivalent)")

    # 3) Save the reduced matrix to a compressed NPZ file
    np.savez_compressed(LSA_OUTPUT_PATH, X_reduced=X_reduced)
    print(f"LSA reduced matrix saved to {LSA_OUTPUT_PATH}")

    # 4) Save the full SVD model
    joblib.dump(svd, SVD_MODEL_PATH)
    print(f"SVD model saved to {SVD_MODEL_PATH}")

    # 5) Save lightweight components
    np.savez_compressed(
        LSA_COMP_PATH,
        components=svd.components_,                                # shape: (n_components, V)
        explained_variance_ratio=getattr(svd, "explained_variance_ratio_", None),
        singular_values=getattr(svd, "singular_values_", None),
    )
    print(f"LSA components saved to {LSA_COMP_PATH}")

    print("All LSA artifacts are ready: reduced matrix + model + components.")

if __name__ == "__main__":
    main()