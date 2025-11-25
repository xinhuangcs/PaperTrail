from pathlib import Path
import json, re
from typing import List, Dict, Optional
from datetime import datetime, UTC
import argparse

#Config
ROOT_DIR = Path(__file__).resolve().parents[3]


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["application", "review", "trending", "theory"], required=True, help="Select Recommended Mode")
args = parser.parse_args()
mode = args.mode


timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M")
CONFIG = {
    #pick the newest recommend_*.json
    "INPUT_DIR": ROOT_DIR / "data" / "recommend",
    "INPUT_PATTERN": f"recommend_{mode}.json",
    "OUTPUT_DIR": ROOT_DIR / "data" / "ai_advice",
    "MAX_ABSTRACT_CHARS": 3000,
    "MAX_QUERY_CHARS": 200,
}
CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)


#tools
def pick_query(r: Dict, max_len: int) -> str:
    q = r.get("query")
    return normalize_text(q, max_len)

def newest_recommend_file(input_dir: Path, pattern: str) -> Path:
    files = sorted(input_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files match: {input_dir}/{pattern}")
    return files[0]

def normalize_text(s: Optional[str], max_len: int) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]

def extract_year(r: Dict) -> int:
    # prefer update_date (YYYY-MM-DD), else versions[0].created
    ud = (r.get("update_date") or "").strip()
    if len(ud) >= 4 and ud[:4].isdigit():
        return int(ud[:4])
    vers = r.get("versions") or []
    if vers and isinstance(vers, list):
        created = vers[0].get("created", "")
        m = re.search(r"\b(\d{4})\b", created or "")
        if m:
            return int(m.group(1))
    return 0

def to_minimal_record(r: Dict, max_abs_len: int, max_query_len: int) -> Dict:
    # make a small, stable schema for LLM input
    pid = r.get("id") or r.get("_id") or ""
    title = (r.get("title") or "").strip()
    abstract = normalize_text(r.get("abstract") or "", max_abs_len)
    year = extract_year(r)
    cit = int(r.get("citation_count") or 0)

    if r.get("_score") is not None:
        score = float(r["_score"])
    elif r.get("score") is not None:
        score = float(r["score"])
    elif r.get("sim_score") is not None:
        score = float(r["sim_score"])
    else:
        score = 0.0

    authors = r.get("authors") or r.get("authors_parsed")
    categories = r.get("categories")
    query = pick_query(r, max_query_len)

    rec = {
        "id": pid,
        "title": title,
        "abstract": abstract,
        "year": year,
        "citation_count": cit,
        "score": score,
        "authors": authors,
        "categories" : categories,
        "query": query,
    }
    return rec



# Main
def build_selected_papers() -> Path:
    latest = newest_recommend_file(CONFIG["INPUT_DIR"], CONFIG["INPUT_PATTERN"])
    seen = set()
    rows: List[Dict] = []

    with latest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("id") or r.get("_id")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            rows.append(
                to_minimal_record(
                    r,
                    max_abs_len=CONFIG["MAX_ABSTRACT_CHARS"],
                    max_query_len=CONFIG["MAX_QUERY_CHARS"],
                )
            )

    out_path = CONFIG["OUTPUT_DIR"] / f"standardize_input_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as wf:
        for rec in rows:
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Processed {len(rows)} papers from {latest.name} -> {out_path}")
    return out_path


if __name__ == "__main__":
    build_selected_papers()
