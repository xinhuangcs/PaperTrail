import os
import numpy as np
from scipy import sparse
import joblib
import sys

from pathlib import Path



# 1) Config
ROOT_DIR = Path(__file__).resolve().parents[2]

DIR_A = ROOT_DIR / "data" / "tf_idf"
DIR_B = ROOT_DIR / "data" / "tf_idf_manual"


MATRIX_FILE = "tfidf_matrix.npz"
VECTORIZER_FILE = "tfidf_vectorizer.joblib"
IDS_FILE = "doc_ids.npy"
TITLES_FILE = "doc_titles.npy"

TOLERANCE = 1e-7



def compare_matrices(file_a, file_b):
    """compare two .npz sparse matrices"""
    try:
        A = sparse.load_npz(file_a)
        B = sparse.load_npz(file_b)
    except FileNotFoundError as e:
        print(f"  [error] file not found: {e.filename}")
        return False
    
    # 1. check shape
    if A.shape != B.shape:
        print(f"  [failed] shape mismatch: {A.shape} vs {B.shape}")
        return False
    
    # 2. check values
    try:
        C = A - B
    except Exception as e:
        print(f"  [failed] matrix cannot be subtracted (format may be different?): {e}")
        return False
        
    if C.nnz == 0:
        # no non-zero elements, means completely equal
        return True
        
    # check if all difference values are almost 0
    are_close = np.allclose(C.data, 0, atol=TOLERANCE)
    if not are_close:
        print(f"  [failed] matrix has {C.nnz} values with difference (greater than {TOLERANCE}).")
        print(f"  for example, the maximum difference value: {np.max(np.abs(C.data))}")
    return are_close

def compare_vectorizers(file_a, file_b):
    """compare two .joblib vectorizers"""
    try:
        A = joblib.load(file_a)
        B = joblib.load(file_b)
    except FileNotFoundError as e:
        print(f"  [error] file not found: {e.filename}")
        return False
    
    # 1. compare vocabulary (most important)
    if A.vocabulary_ != B.vocabulary_:
        print(f"  [failed] vocabulary (vocabulary_) is inconsistent.")
        len_a = len(A.vocabulary_)
        len_b = len(B.vocabulary_)
        if len_a != len_b:
            print(f"  vocabulary size is different: {len_a} vs {len_b}")
        return False
    
    # 2. compare feature names (order must also be consistent)
    try:
        feats_a = A.get_feature_names_out()
        feats_b = B.get_feature_names_out()
        if not np.array_equal(feats_a, feats_b):
            print(f"  [failed] feature names (feature_names_out) are inconsistent.")
            return False
    except Exception as e:
        print(f"  [warning] error comparing feature_names_out: {e}")
        
    return True

def compare_npy_arrays(file_a, file_b):
    try:
        A = np.load(file_a, allow_pickle=True) # allow_pickle 以防万一
        B = np.load(file_b, allow_pickle=True)
    except FileNotFoundError as e:
        print(f"  [error] file not found: {e.filename}")
        return False
        
    if not np.array_equal(A, B):
        print(f"  [failed] array content is inconsistent.")
        if A.shape != B.shape:
            print(f"  shape mismatch: {A.shape} vs {B.shape}")

        return False
    return True


def main():
    all_good = True
    
    print(f"--- Comparing directories ---")
    print(f"A: {DIR_A}")
    print(f"B: {DIR_B}")
    
    # 1. compare matrix
    print(f"\n[1] compare {MATRIX_FILE}...")
    path_a = os.path.join(DIR_A, MATRIX_FILE)
    path_b = os.path.join(DIR_B, MATRIX_FILE)
    if compare_matrices(path_a, path_b):
        print(f"  ✅ {MATRIX_FILE} is consistent.")
    else:
        print(f"  ❌ {MATRIX_FILE} is inconsistent.")
        all_good = False
        
    # 2. compare vectorizer
    print(f"\n[2] compare {VECTORIZER_FILE}...")
    path_a = os.path.join(DIR_A, VECTORIZER_FILE)
    path_b = os.path.join(DIR_B, VECTORIZER_FILE)
    if compare_vectorizers(path_a, path_b):
        print(f"  ✅ {VECTORIZER_FILE} is consistent.")
    else:
        print(f"  ❌ {VECTORIZER_FILE} is inconsistent.")
        all_good = False
        
    # 3. compare doc ids
    print(f"\n[3] compare {IDS_FILE}...")
    path_a = os.path.join(DIR_A, IDS_FILE)
    path_b = os.path.join(DIR_B, IDS_FILE)
    if compare_npy_arrays(path_a, path_b):
        print(f"  ✅ {IDS_FILE} is consistent.")
    else:
        print(f"  ❌ {IDS_FILE} is inconsistent.")
        all_good = False
        
    # 4. compare doc titles
    print(f"\n[4] compare {TITLES_FILE}...")
    path_a = os.path.join(DIR_A, TITLES_FILE)
    path_b = os.path.join(DIR_B, TITLES_FILE)
    if compare_npy_arrays(path_a, path_b):
        print(f"  ✅ {TITLES_FILE} is consistent.")
    else:
        print(f"  ❌ {TITLES_FILE} is inconsistent.")
        all_good = False

    print("\n" + "="*30)
    print("--- summary ---")
    if all_good:
        print("🎉 all TF-IDF related files in both directories are consistent.")
    else:
        print("⚠️ differences found in both directories. Please check the ❌ marks above.")
    print("="*30)

if __name__ == "__main__":
    main()

