from pathlib import Path
import json
import numpy as np
from scipy import sparse
import joblib

# 1) paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TFIDF_DIR = DATA_DIR / "tf_idf"
LSA_DIR = DATA_DIR / "lsa"
OUT_DIR = DATA_DIR / "similarity_results"/ "similarity_results_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2) inputs (already exist)
TFIDF_VECTORIZER_PATH = TFIDF_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = TFIDF_DIR / "tfidf_matrix.npz"
LSA_MODEL_PATH = LSA_DIR / "lsa_model.joblib"
LSA_MATRIX_PATH = LSA_DIR / "lsa_reduced.npz"  # expects key 'X_reduced'

# 3) outputs (write everything here, as you asked)
VOCAB_JSON = OUT_DIR / "vocab.json"                 # token -> col_idx
IDF_NPY = OUT_DIR / "idf.npy"                        # (V,)
USE_L2_TXT = OUT_DIR / "use_l2_norm.txt"             # "1"/"0"
SVD_COMPONENTS_NPY = OUT_DIR / "svd_components.npy"  # (k, V)
NCOMP_TXT = OUT_DIR / "n_components.txt"             # k
TFIDF_ROW_NORMS_NPY = OUT_DIR / "row_l2_norms.npy"   # (N,)
LSA_PRENORM_NPZ = OUT_DIR / "lsa_reduced_l2norm.npz" # 'Xr_norm'

def export_vectorizer():
    # dump vocab/idf/norm flag to plain files
    print("exporting vectorizer stuff...")
    vec = joblib.load(TFIDF_VECTORIZER_PATH)
    vocab = {k: int(v) for k, v in vec.vocabulary_.items()}
    VOCAB_JSON.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    np.save(IDF_NPY, vec.idf_)
    USE_L2_TXT.write_text("1" if getattr(vec, "norm", "l2") == "l2" else "0")
    print(f"saved: {VOCAB_JSON}")
    print(f"saved: {IDF_NPY}")
    print(f"saved: {USE_L2_TXT}")

def export_svd():
    # dump truncated svd components and k
    print("exporting svd stuff...")
    svd = joblib.load(LSA_MODEL_PATH)
    np.save(SVD_COMPONENTS_NPY, svd.components_)
    NCOMP_TXT.write_text(str(svd.n_components))
    print(f"saved: {SVD_COMPONENTS_NPY}")
    print(f"saved: {NCOMP_TXT}")

def export_tfidf_row_norms():
    # precompute tf-idf row l2 norms (for true cosine)
    if not TFIDF_MATRIX_PATH.exists():
        print("skip row norms (tfidf_matrix.npz not found)")
        return
    print("computing tfidf row norms...")
    X = sparse.load_npz(TFIDF_MATRIX_PATH)
    sq = X.multiply(X).sum(axis=1)
    norms = np.sqrt(np.asarray(sq).ravel())
    norms[norms == 0] = 1.0
    np.save(TFIDF_ROW_NORMS_NPY, norms)
    print(f"saved: {TFIDF_ROW_NORMS_NPY}")

def export_lsa_prenorm():
    # pre-normalize lsa doc matrix (faster cosine)
    if not LSA_MATRIX_PATH.exists():
        print("skip lsa prenorm (lsa_reduced.npz not found)")
        return
    d = np.load(LSA_MATRIX_PATH)
    if "X_reduced" not in d:
        print("skip lsa prenorm (X_reduced not found)")
        return
    print("normalizing lsa doc matrix...")
    Xr = d["X_reduced"]
    norms = np.linalg.norm(Xr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xr_norm = Xr / norms
    np.savez_compressed(LSA_PRENORM_NPZ, Xr_norm=Xr_norm)
    print(f"saved: {LSA_PRENORM_NPZ}")

def main():
    export_vectorizer()
    export_svd()
    export_tfidf_row_norms()
    export_lsa_prenorm()
    print("done. all files are in data/similarity_results_v2")

if __name__ == "__main__":
    main()
