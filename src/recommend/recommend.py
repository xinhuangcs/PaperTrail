
"""
Re-ranking by user perspective (theoretical / application / review / trending)
"""

import os
import re
import json
import math
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

# CONFIG
CONFIG = {
    # candidate set (JSONL)
    "INPUT_JSONL": "/Users/jasonh/Desktop/02807/PaperTrail/data/similarity_results/similarity_for_recommend_lsa_1761492268.jsonl",

    # Output dir
    "OUTPUT_DIR": "/Users/jasonh/Desktop/02807/PaperTrail/data/recommend",

    # Keyword lexicons
    "KW_REVIEW": [
        "survey", "review", "overview", "tutorial", "comprehensive", "literature review",
        "systematic review", "meta-analysis", "benchmark", "comparative study"
    ],
    "KW_APPLICATION": [
        "application", "applied", "experiment", "experiments", "experimental",
        "evaluation", "real-world", "real world", "case study", "deployment",
        "dataset", "benchmark", "implementation", "industrial", "practical"
    ],
    "KW_THEORY": [
        "theorem", "theorems", "proof", "lemma", "corollary", "bound", "bounds",
        "formal", "convergence", "guarantee", "complexity", "approximation",
        "lower bound", "upper bound", "optimal", "tight"
    ],
    # Trending keywords (optional; we rely mostly on recency)
    "KW_TRENDING": [
        "emerging", "recent", "trend", "state-of-the-art", "sota", "novel"
    ],

    "RECENT_YEARS": 2,

    # Candidate text fields to scan for keywords
    "TEXT_FIELDS": ["processed_content", "abstract", "title"],

    # Candidate similarity score
    "SIM_FIELDS": ["sim_score", "score", "similarity"],

    # Weights per mode (sum ≈ 1)
    # Components:
    #   sim  : query-doc similarity from stage-5 (optional)
    #   cite : citation strength (log1p-scaled + normalized)
    #   rec  : recency (normalized year / exponential recency)
    #   kw_r : review keyword score
    #   kw_a : application keyword score
    #   kw_t : theory keyword score
    #   kw_tr: trending keyword score
    "WEIGHTS": {
        "theoretical": {"sim": 0.30, "cite": 0.45, "rec": 0.10, "kw_r": 0.00, "kw_a": 0.00, "kw_t": 0.15, "kw_tr": 0.00},
        "application": {"sim": 0.35, "cite": 0.20, "rec": 0.20, "kw_r": 0.00, "kw_a": 0.25, "kw_t": 0.00, "kw_tr": 0.00},
        "review":      {"sim": 0.25, "cite": 0.45, "rec": 0.05, "kw_r": 0.25, "kw_a": 0.00, "kw_t": 0.00, "kw_tr": 0.00},
        "trending":    {"sim": 0.35, "cite": 0.10, "rec": 0.45, "kw_r": 0.00, "kw_a": 0.00, "kw_t": 0.00, "kw_tr": 0.10},
    },

    # Guards for normalization (avoid zero-division)
    "EPS": 1e-9,
}

# Text helpers
def normalize_space(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def pick_first_existing(d: Dict, keys: List[str], default=None):
#Pick the first non-empty field in keys from dict
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return default

def safe_year(update_date: str) -> int:
    if not update_date:
        return 0
    try:
        return int(update_date[:4])
    except Exception:
        return 0

def now_year() -> int:
    return datetime.now().year

def count_keywords(text: str, keywords: List[str]) -> int:
    """Count how many keywords appear (出现次数，粗略匹配)"""
    if not text:
        return 0
    cnt = 0
    for kw in keywords:
        if kw in text:
            cnt += 1
    return cnt

def get_similarity(paper: Dict) -> float:
    for k in CONFIG["SIM_FIELDS"]:
        v = paper.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                continue
    return 0.0

def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

#  Normalization
def minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < CONFIG["EPS"]:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]

def log1p_then_minmax(values: List[float]) -> List[float]:
    logged = [math.log1p(max(0.0, v)) for v in values]
    return minmax_norm(logged)

# Exponential recency (optional)
def recency_score_by_year(years: List[int], tau: float = 3.0) -> List[float]:
    Y = now_year()
    raw = [math.exp(-(max(0, Y - (y or 0)) / max(CONFIG["EPS"], tau))) for y in years]
    return minmax_norm(raw)

# Scoring
def score_papers(papers: List[Dict], mode: str, top_n: int) -> Tuple[List[Dict], str]:

    mode = mode.lower().strip()
    if mode not in CONFIG["WEIGHTS"]:
        raise ValueError(f"未知模式/Unknown mode: {mode}; 可选：{list(CONFIG['WEIGHTS'].keys())}")

    W = CONFIG["WEIGHTS"][mode]

    # 1)Collect raw features
    cites = []
    years = []
    sims  = []
    kw_review = []
    kw_app    = []
    kw_theory = []
    kw_trend  = []
    texts_for_kw = []

    for p in papers:
        # citation count
        cites.append(float(p.get("citation_count", 0) or 0))

        # year
        years.append(safe_year(p.get("update_date", "") or p.get("year", "")))

        # similarity from stage-5
        sims.append(get_similarity(p))

        # build a single text bag for keyword scan
        text = ""
        for fld in CONFIG["TEXT_FIELDS"]:
            val = p.get(fld)
            if val:
                text += " " + str(val)
        text = normalize_space(text)

        texts_for_kw.append(text)
        kw_review.append(count_keywords(text, CONFIG["KW_REVIEW"]))
        kw_app.append(count_keywords(text, CONFIG["KW_APPLICATION"]))
        kw_theory.append(count_keywords(text, CONFIG["KW_THEORY"]))
        kw_trend.append(count_keywords(text, CONFIG["KW_TRENDING"]))

    # 2) ormalize features to comparable [0,1]
    cite_n = log1p_then_minmax(cites)
    rec_n  = recency_score_by_year(years, tau=3.0)
    sim_n  = minmax_norm(sims) if any(sims) else [0.0]*len(papers)

    kw_r_n = minmax_norm(kw_review)
    kw_a_n = minmax_norm(kw_app)
    kw_t_n = minmax_norm(kw_theory)
    kw_tr_n= minmax_norm(kw_trend)

    # 3) extra boost for recent papers under 'trending'
    recent_bonus = [0.0]*len(papers)
    if mode == "trending":
        Y = now_year()
        for i, y in enumerate(years):
            if y and (Y - y) <= CONFIG["RECENT_YEARS"]:
                recent_bonus[i] = 0.05  # 小幅奖励 small bonus（可调）

    # 4) Weighted linear combination
    scores = []
    breakdowns = []
    for i in range(len(papers)):
        comp = {
            "sim": sim_n[i],
            "cite": cite_n[i],
            "rec": rec_n[i],
            "kw_r": kw_r_n[i],
            "kw_a": kw_a_n[i],
            "kw_t": kw_t_n[i],
            "kw_tr": kw_tr_n[i],
            "recent_bonus": recent_bonus[i]
        }
        score = (
            W["sim"]  * comp["sim"]  +
            W["cite"] * comp["cite"] +
            W["rec"]  * comp["rec"]  +
            W["kw_r"] * comp["kw_r"] +
            W["kw_a"] * comp["kw_a"] +
            W["kw_t"] * comp["kw_t"] +
            W["kw_tr"]* comp["kw_tr"] +
            comp["recent_bonus"]
        )
        scores.append(score)
        breakdowns.append(comp)

    # 5)Rank & slice Top-N
    idx_sorted = sorted(range(len(papers)), key=lambda i: scores[i], reverse=True)
    top_n = min(top_n, len(idx_sorted))
    top_idx = idx_sorted[:top_n]

    # 6) build output rows with explanations
    Y = now_year()
    rows = []
    for rank, i in enumerate(top_idx, start=1):
        p = dict(papers[i])
        p["_rank"] = rank
        p["_mode"] = mode
        p["_final_score"] = float(scores[i])
        p["_score_breakdown"] = breakdowns[i]
        reasons = []
        if breakdowns[i]["sim"] > 0.66 and W["sim"] > 0:
            reasons.append("high semantic similarity")
        if breakdowns[i]["cite"] > 0.66 and W["cite"] > 0:
            reasons.append("strong citation impact")
        if (Y - (years[i] or 0)) <= CONFIG["RECENT_YEARS"] and W["rec"] > 0:
            reasons.append("recent years")
        if breakdowns[i]["kw_r"] > 0.33 and W["kw_r"] > 0:
            reasons.append("likely survey/review")
        if breakdowns[i]["kw_a"] > 0.33 and W["kw_a"] > 0:
            reasons.append("likely application-oriented")
        if breakdowns[i]["kw_t"] > 0.33 and W["kw_t"] > 0:
            reasons.append("likely theoretical")
        if breakdowns[i]["kw_tr"] > 0.33 and W["kw_tr"] > 0:
            reasons.append("mentions trend/novelty")
        p["_reasons"] = reasons
        rows.append(p)

    # 7) Save JSONL
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(CONFIG["OUTPUT_DIR"], f"recommend_top{top_n}_{mode}_{stamp}.jsonl")
    write_jsonl(out_path, rows)
    return rows, out_path

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Re-ranking by user perspective")
    parser.add_argument("--input", type=str, default=CONFIG["INPUT_JSONL"],
                        help="Path to stage-5 candidates JSONL file")
    parser.add_argument("--mode", type=str, default=None,
                        help="Perspective: theoretical/application/review/trending")
    parser.add_argument("--topn", type=int, default=None, help="Top-N recommendations")
    args = parser.parse_args()

    if args.input:
        CONFIG["INPUT_JSONL"] = args.input

    if args.mode is None:
        mode = input("Select (theoretical/application/review/trending): ").strip().lower()
    else:
        mode = args.mode.strip().lower()

    if args.topn is None:
        try:
            topn = int(input("Top N? ").strip())
        except Exception:
            topn = 20
    else:
        topn = int(args.topn)

    # 读取候选集合 / Load candidates
    if not os.path.exists(CONFIG["INPUT_JSONL"]):
        raise FileNotFoundError(f"输入文件不存在：{CONFIG['INPUT_JSONL']}")

    papers = read_jsonl(CONFIG["INPUT_JSONL"])
    if not papers:
        print("no candidates in input.")
        return

    rows, out_path = score_papers(papers, mode, topn)
    print(f"output Top-{len(rows)} recommended to：{out_path}")

if __name__ == "__main__":
    main()