from pathlib import Path
import re, time, json
import numpy as np
from typing import List, Tuple, Optional, Dict
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords



# 1) paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SBERT_DIR = DATA_DIR / "sbert_hdbscan_test"
OUT_DIR = DATA_DIR / "similarity_results" / "similarity_results_sbert"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2) inputs (from SBERT pipeline)
DOC_IDS_PATH = SBERT_DIR / "doc_ids.npy"
DOC_TITLES_PATH = SBERT_DIR / "doc_titles.npy"
SBERT_EMBEDDINGS_PATH = SBERT_DIR / "sbert_embeddings_norm.npy"
CLUSTER_LABELS_PATH = SBERT_DIR / "hdbscan_labels.npy"
CLUSTER_TOP_TERMS_PATH = SBERT_DIR / "cluster_top_terms.json"


# 3) SBERT model name
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
RAW_JSONL_PATH = DATA_DIR / "preprocess" / "arxiv-cs-data-with-citations-final-dataset.json"
CUSTOM_STOPWORDS_PATH = ROOT_DIR / "src" / "custom_stopwords.txt"


# --- Smart Stop Words Handling ---
# Download nltk stopwords if not already present
try:
    stopwords.words('english')
    print("[i] NLTK stopwords found.")
except LookupError:
    print("[i] NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    print("[i] Download complete.")


def load_stopwords():
    """Load stopwords from NLTK and custom stopwords file."""
    # Load NLTK stopwords
    nltk_stopwords = set(stopwords.words('english'))
    
    # Load custom stopwords
    custom_stopwords = set()
    if CUSTOM_STOPWORDS_PATH.exists():
        try:
            custom_stopwords = {
                line.strip().lower()
                for line in CUSTOM_STOPWORDS_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            print(f"[i] Loaded {len(custom_stopwords)} custom stopwords.")
        except Exception as exc:
            print(f"[warn] Failed to load custom stopwords: {exc}")
    else:
        print(f"[warn] Custom stopwords file not found: {CUSTOM_STOPWORDS_PATH}")
    
    # Merge stopwords
    all_stopwords = nltk_stopwords | custom_stopwords
    print(f"[i] Total stopwords: {len(all_stopwords)} (NLTK: {len(nltk_stopwords)}, custom: {len(custom_stopwords)})")
    return all_stopwords


STOP_WORDS = load_stopwords()




def preprocess_query(q: str) -> str:
    # Lowercase, remove non-alphanumeric, then filter stop words
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    # Filter stop words
    tokens = [token for token in q.split() if token not in STOP_WORDS]
    return " ".join(tokens)


def load_sbert_artifacts():
    # load SBERT embeddings and doc metadata
    print("[i] Loading SBERT artifacts...")
    doc_ids = np.load(DOC_IDS_PATH, allow_pickle=True)
    doc_titles = np.load(DOC_TITLES_PATH, allow_pickle=True)
    doc_embeddings = np.load(SBERT_EMBEDDINGS_PATH)
    print(f"[i] Loaded {len(doc_ids)} doc IDs/titles.")
    print(f"[i] Loaded doc embeddings matrix with shape: {doc_embeddings.shape}")
    cluster_labels: Optional[np.ndarray] = None
    if CLUSTER_LABELS_PATH.exists():
        cluster_labels = np.load(CLUSTER_LABELS_PATH)
        if cluster_labels.shape[0] != doc_ids.shape[0]:
            print(
                f"[warn] cluster labels length mismatch ({cluster_labels.shape[0]} vs {doc_ids.shape[0]}); ignoring"
            )
            cluster_labels = None
    else:
        print("[warn] cluster labels file not found; cluster context unavailable.")

    cluster_topics: Dict[int, List[str]] = {}
    if CLUSTER_TOP_TERMS_PATH.exists():
        try:
            raw = json.loads(CLUSTER_TOP_TERMS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key, value in raw.items():
                    try:
                        cid = int(key)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(value, list):
                        cluster_topics[cid] = [str(v) for v in value]
                    elif value is not None:
                        cluster_topics[cid] = [str(value)]
        except Exception as exc:
            print(f"[warn] failed to load cluster top terms: {exc}")
    else:
        print("[warn] cluster top terms file not found; cluster topics will be empty.")

    return doc_ids, doc_titles, doc_embeddings, cluster_labels, cluster_topics


def l2_normalize_dense(A: np.ndarray) -> np.ndarray:
    # row-wise l2 norm
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n


# Global cache for the SBERT model
_SBERT_MODEL = None

def get_sbert_model():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        print(f"[i] Loading SBERT model: {SBERT_MODEL_NAME}...")
        _SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME)
        print("[i] SBERT model loaded.")
    return _SBERT_MODEL


def search_sbert(query: str, top_k: int) -> List[Tuple[str, str, float, Optional[int], Optional[List[str]]]]:
    # sbert retrieval
    doc_ids, doc_titles, doc_embeddings, cluster_labels, cluster_topics = load_sbert_artifacts()

    # Preprocess and encode query
    processed_query = preprocess_query(query)
    print(f"[i] Original query: '{query}'")
    print(f"[i] Processed query: '{processed_query}'")

    model = get_sbert_model()
    q_vec = model.encode([processed_query], convert_to_numpy=True, show_progress_bar=False)
    q_vec = l2_normalize_dense(q_vec)

    # Compute cosine similarities
    # (since both query and doc embeddings are normalized, dot product is cosine similarity)
    sims = (q_vec @ doc_embeddings.T).ravel()

    # Get top_k results
    if top_k >= len(sims):
        top_indices = np.argsort(-sims)
    else:
        top_indices = np.argpartition(-sims, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

    results: List[Tuple[str, str, float, Optional[int], Optional[List[str]]]] = []
    for i in top_indices:
        # We can set a threshold if we want, e.g., > 0.3
        if sims[i] > 0.1:
            cluster_id: Optional[int] = None
            cluster_topics_list: Optional[List[str]] = None
            if cluster_labels is not None and i < cluster_labels.shape[0]:
                try:
                    lbl = int(cluster_labels[i])
                except (TypeError, ValueError):
                    lbl = -1
                if lbl >= 0:
                    cluster_id = lbl
                    cluster_topics_list = cluster_topics.get(lbl)
            results.append(
                (
                    str(doc_ids[i]),
                    str(doc_titles[i]),
                    float(sims[i]),
                    cluster_id,
                    cluster_topics_list,
                )
            )

    return results


def save_results_jsonl(
    query: str,
    method: str,
    results: List[Tuple[str, str, float, Optional[int], Optional[List[str]]]],
) -> Path:
    need_ids = {pid for (pid, _title, _s, _cid, _topics) in results}
    raw_meta = load_raw_meta(need_ids)

    ts = int(time.time())
    out_path = OUT_DIR / f"similarity_for_recommend_{method}_{ts}.json"

    with out_path.open("w", encoding="utf-8") as f:
        for rank, (pid, title, sc, cluster_id, cluster_topics_list) in enumerate(results, 1):
            base = raw_meta.get(pid, {})
            base.update({
                "sim_score": float(sc),
                "score": float(sc),
                "similarity": float(sc),
                "rank": rank,
                "query": query,
                "method": method,
                "sbert_cluster_id": int(cluster_id) if cluster_id is not None else None,
                "cluster_topics": cluster_topics_list if cluster_topics_list is not None else None,
            })
            f.write(json.dumps(base, ensure_ascii=False) + "\n")

    print(f"saved to: {out_path}")
    return out_path



def print_results(
    query: str,
    method: str,
    results: List[Tuple[str, str, float, Optional[int], Optional[List[str]]]],
):
    # simple pretty print
    print("\n" + "=" * 70)
    print(f"query: {query}")
    print(f"model: {method}")
    print(f"top {len(results)} here ↓")
    print("=" * 70)
    for i, (pid, title, sc, cluster_id, cluster_topics_list) in enumerate(results, 1):
        print(f"{i:2d}. [{pid}] {title}")
        cluster_str = "N/A" if cluster_id is None else str(cluster_id)
        print(f"    score: {sc:.4f} | cluster: {cluster_str}")
        if cluster_topics_list:
            pretty_topics = ", ".join(cluster_topics_list)
            print(f"    topics: {pretty_topics}")
    if not results:
        print("hmm no hit, maybe try other words")


def load_raw_meta(need_ids: set) -> dict:
    hit = {}
    if not RAW_JSONL_PATH.exists():
        print("raw file not found, skip extra fields")
        return hit
    found = 0
    target = len(need_ids)
    with RAW_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = obj.get("id")
            if pid in need_ids:
                hit[pid] = obj
                found += 1
                if found >= target:
                    break
    return hit


def main():
    # interactive cli
    print("SBERT Similarity Search")
    method = "sbert" # Hardcode to SBERT for simplicity in this script
    
    query = input("Type your query: ").strip()
    top_k_s = input("Top-k? (default 10): ").strip()
    top_k = int(top_k_s) if top_k_s.isdigit() and int(top_k_s) > 0 else 10

    t0 = time.time()
    results = search_sbert(query, top_k)
    dt = time.time() - t0

    print_results(query, method, results)
    out_file = save_results_jsonl(query, method, results)
    print(f"saved to: {out_file}")
    print(f"time: {dt:.2f}s")

if __name__ == "__main__":
    main()
