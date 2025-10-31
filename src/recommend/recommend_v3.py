import re
import json
import gzip
import math
import random
from pathlib import Path
from time import perf_counter
from datetime import datetime
from collections import defaultdict

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
SIM_DIR = ROOT_DIR / "data" / "similarity_results" / "similarity_results_v2"

json_files = list(SIM_DIR.glob("similarity_for_recommend_lsa_*.json"))
if not json_files:
    raise FileNotFoundError("No similarity_for_recommend_lsa_*.json under similarity_results_v2")

INPUT_JSON = max(json_files, key=lambda p: p.stat().st_mtime)
OUTPUT_DIR = ROOT_DIR / "data" / "recommend"

FALLBACK_QUERY = "deep learning"
VIEWS = "all"           # all / default / review / application / theory / trending
TOPK = 200
MMR_LAMBDA = 0.75
SEED = 42

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _open_any(path):
    # handle Path and str; support gzip by extension
    path = str(path)
    if path.endswith(".gz") or path.endswith(".gzip"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def load_records(path):
    # support json array and jsonl
    with _open_any(path) as f:
        head = f.read(4096)
        f.seek(0)
        if head.lstrip().startswith("["):
            return json.load(f)
        recs = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                if line.endswith(","):
                    line = line[:-1]
                    recs.append(json.loads(line))
                else:
                    raise
        return recs

def to_dt(s):
    # parse to naive datetime
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt)
        except Exception:
            pass
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except Exception:
        return None

def clean_text(x):
    x = (x or "").lower()
    x = re.sub(r"[^a-z0-9\s]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def tokenize_title(x):
    x = clean_text(x)
    toks = [t for t in x.split() if len(t) > 2]
    return set(toks)

def rank_percentiles(values):
    # map value -> percentile [0,1]
    vals = [v for v in values if v is not None]
    if not vals:
        return defaultdict(lambda: 0.0)
    vals.sort()
    n = len(vals)
    first_idx = {}
    for i, v in enumerate(vals):
        if v not in first_idx:
            first_idx[v] = i
    return defaultdict(lambda: 0.0, {v: first_idx[v] / max(n - 1, 1) for v in first_idx})

def log1p(x):
    try:
        return math.log1p(max(float(x), 0.0))
    except Exception:
        return 0.0

# ------------------------------------------------------------
# Keyword lexicons
# ------------------------------------------------------------

P_REVIEW = [r"survey", r"review", r"overview", r"tutorial", r"systematic review", r"literature review"]
P_APP = [r"application", r"case study", r"real[-\s]?world", r"deployment", r"dataset", r"benchmark", r"experimental", r"experiments", r"implementation"]
P_THEORY = [r"theorem", r"proof", r"convergence", r"bound", r"complexity", r"guarantee", r"asymptotic"]

DL_TERMS = {"deep", "learning", "neural", "network", "networks", "vision", "nlp", "transformer"}
DEFAULT_GOOD_CATS = {"cs.lg", "cs.ai", "cs.cv", "cs.cl", "stat.ml"}

def kw_score(text, patterns, cap=2):
    # count capped matches, scaled to [0,1]
    if not text:
        return 0.0
    s = 0
    for p in patterns:
        if re.search(r"\b" + p + r"\b", text):
            s += 1
            if s >= cap:
                break
    return s / cap

# ------------------------------------------------------------
# Bayesian calibration for citations per year
# ------------------------------------------------------------

def bayes_citations_per_year(citations, age_years, alpha0=1.0, beta0=1.0):
    # posterior mean under Gamma-Poisson
    t = max(age_years, 1/12)
    c = max(0.0, float(citations or 0))
    return (alpha0 + c) / (beta0 + t)

# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

def build_features(records, fallback_query=None):
    # pass 1: parse dates and cache raw fields
    cache = []
    for r in records:
        sim = r.get("similarity", r.get("sim_score", r.get("score", 0.0)))
        title = r.get("title") or ""
        abstract = r.get("abstract") or ""
        cats = (r.get("categories") or "").lower()
        c = r.get("citation_count") or 0
        versions = r.get("versions") or []

        dt_first = to_dt(versions[0].get("created")) if versions else None
        dt_last  = to_dt(versions[-1].get("created")) if versions else None
        upd      = to_dt(r.get("update_date"))

        pub_date   = dt_first or upd or dt_last
        last_upd   = upd or dt_last or pub_date

        cache.append((r, sim, title, abstract, cats, c, pub_date, last_upd))

    # pick snapshot_date as max(last_update or pub_date)
    all_dates = []
    for (_r, _sim, _title, _abstract, _cats, _c, _pub, _last) in cache:
        if _last is not None:
            all_dates.append(_last)
        elif _pub is not None:
            all_dates.append(_pub)
    snapshot_date = max(all_dates) if all_dates else None

    # stats buffers
    sim_vals, cit_vals = [], []
    recency_ordinals = []

    for (_r, _sim, _title, _abstract, _cats, _c, _pub, _last) in cache:
        sim_vals.append(_sim)
        cit_vals.append(log1p(_c))
        if _pub is not None:
            recency_ordinals.append(_pub.toordinal())
        else:
            recency_ordinals.append(0)

    sim_pct   = rank_percentiles(sim_vals)
    cit_pct   = rank_percentiles(cit_vals)
    rec_pct   = rank_percentiles(recency_ordinals)

    # pass 2: compute age and bayesian cit/year to collect percentiles
    citpy_vals = []
    tmp2 = []
    for (r, sim, title, abstract, cats, c, pub_date, last_upd) in cache:
        if snapshot_date is not None and pub_date is not None:
            age_days = max((snapshot_date - pub_date).days, 1)
            age_years = max(age_days / 365.0, 1/12)
        else:
            age_years = 1.0
        cit_py_bayes = bayes_citations_per_year(c or 0, age_years, alpha0=1.0, beta0=1.0)
        citpy_vals.append(log1p(cit_py_bayes))
        tmp2.append((r, sim, title, abstract, cats, c, pub_date, age_years, cit_py_bayes))

    citpy_pct = rank_percentiles(citpy_vals)

    # pass 3: build final features
    feats = []
    for (r, sim, title, abstract, cats, c, pub_date, age_years, cit_py_bayes) in tmp2:
        title_c = clean_text(title or "")
        abs_c   = clean_text(abstract or "")
        cats_c  = clean_text(cats or "")

        q   = r.get("query") or fallback_query or ""
        q_c = clean_text(q)
        q_terms = set([t for t in q_c.split() if len(t) > 2])

        f_sim   = sim_pct[sim]
        f_title = 1.0 if any(t in title_c for t in q_terms) else 0.0
        abs_hits = sum(1 for t in q_terms if t in abs_c)
        f_abs   = min(abs_hits, 3) / 3.0

        f_cat = 1.0 if (DEFAULT_GOOD_CATS & set(cats_c.split())) and (DL_TERMS & q_terms or {"deep","learning"} & q_terms) else 0.0

        doc_text = f"{title_c} {abs_c}"
        f_rev = kw_score(doc_text, P_REVIEW, cap=2)
        f_app = kw_score(doc_text, P_APP, cap=2)
        f_the = kw_score(doc_text, P_THEORY, cap=2)

        f_cit   = cit_pct[log1p(c or 0)]
        f_citpy = citpy_pct[log1p(cit_py_bayes)]

        if pub_date is not None:
            f_rec = rec_pct[pub_date.toordinal()]
        else:
            f_rec = 0.0

        status = (r.get("citation_status") or "").lower()
        f_trust = 1.0 if ("ok_openalex" in status or "ok" in status) else 0.7

        qid = q_c if q_c else "__GLOBAL__"

        feats.append({
            "id": r.get("id"),
            "qid": qid,
            "f_sim": f_sim,
            "f_title": f_title,
            "f_abs": f_abs,
            "f_cat": f_cat,
            "f_is_review": f_rev,
            "f_is_application": f_app,
            "f_is_theory": f_the,
            "f_cit": f_cit,
            "f_cit_per_year": f_citpy,
            "f_recency": f_rec,
            "f_trust": f_trust,
            "age_years": age_years,
            "title_tokens": tokenize_title(title or ""),
            "_raw": r,
        })
    return feats

# ------------------------------------------------------------
# Weak supervision labels
# ------------------------------------------------------------

def weak_label(feat, view="default"):
    y = 0.6 * feat["f_sim"] + 0.2 * feat["f_title"] + 0.2 * feat["f_cat"]
    y += 0.1 * feat["f_abs"]

    if view == "review":
        y += 0.4 * feat["f_is_review"] + 0.15 * feat["f_cit"]
    elif view == "application":
        y += 0.35 * feat["f_is_application"] + 0.15 * feat["f_abs"]
    elif view == "theory":
        y += 0.35 * feat["f_is_theory"] + 0.10 * feat["f_cat"]
    elif view == "trending":
        y += 0.35 * feat["f_recency"] + 0.25 * feat["f_cit_per_year"]
    else:
        y += 0.15 * feat["f_cit_per_year"] + 0.10 * feat["f_recency"]

    y *= (0.85 + 0.15 * feat["f_trust"])
    return float(y)

# ------------------------------------------------------------
# Pairwise logistic with non-negative weights
# ------------------------------------------------------------

FEATURE_KEYS = [
    "f_sim",
    "f_title",
    "f_abs",
    "f_cat",
    "f_cit_per_year",
    "f_recency",
    "f_is_review",
    "f_is_application",
    "f_is_theory",
]

def vectorize(feat):
    return [feat[k] for k in FEATURE_KEYS]

def pairwise_logistic_loss(w, pairs, lam=1e-3):
    loss = 0.0
    for xi, xj in pairs:
        si = sum(w[k] * xi[k] for k in range(len(w)))
        sj = sum(w[k] * xj[k] for k in range(len(w)))
        z = si - sj
        if z > 0:
            loss += math.log1p(math.exp(-z))
        else:
            loss += (-z) + math.log1p(math.exp(-z))
    loss += lam * sum(wk * wk for wk in w)
    return loss

def build_pairs(features, labels, max_pairs_per_query=2000):
    by_q = defaultdict(list)
    for f, y in zip(features, labels):
        by_q[f["qid"]].append((f, y))

    pairs = []
    for qid, lst in by_q.items():
        lst.sort(key=lambda t: t[1], reverse=True)
        local_pairs = 0
        for i in range(len(lst)):
            xi = vectorize(lst[i][0])
            for j in range(i + 1, min(i + 10, len(lst))):
                xj = vectorize(lst[j][0])
                pairs.append((xi, xj))
                local_pairs += 1
                if local_pairs >= max_pairs_per_query:
                    break
            if local_pairs >= max_pairs_per_query:
                break
    random.shuffle(pairs)
    return pairs

def coordinate_ascent_nonneg(w, pairs, iters=300, step=0.05, lam=1e-3, seed=42):
    random.seed(seed)
    d = len(w)
    best_w = w[:]
    best_val = pairwise_logistic_loss(best_w, pairs, lam)

    for _ in range(iters):
        k = random.randrange(d)
        candidates = []
        for delta in (+step, -step):
            nw = best_w[:]
            nw[k] = max(0.0, nw[k] + delta)
            val = pairwise_logistic_loss(nw, pairs, lam)
            candidates.append((val, nw))
        candidates.sort(key=lambda t: t[0])
        if candidates and candidates[0][0] < best_val:
            best_val, best_w = candidates[0]

    s = sum(best_w)
    if s > 0:
        best_w = [wk / s for wk in best_w]
    return best_w

# ------------------------------------------------------------
# Intent gating (simple rules) and mixing
# ------------------------------------------------------------

def intent_prior_from_query(query_text):
    q = clean_text(query_text or "")
    pri = {"review": 0.25, "application": 0.25, "theory": 0.25, "trending": 0.25}
    if not q:
        return pri
    if any(t in q for t in ["survey", "overview", "tutorial", "review"]):
        pri["review"] += 0.3
    if any(t in q for t in ["dataset", "benchmark", "case", "real world", "deployment", "application"]):
        pri["application"] += 0.3
    if any(t in q for t in ["theorem", "bound", "proof", "convergence", "complexity"]):
        pri["theory"] += 0.3
    if any(t in q for t in ["latest", "2024", "2025", "new", "state of the art", "sota", "recent"]):
        pri["trending"] += 0.3
    s = sum(pri.values())
    for k in pri:
        pri[k] /= s
    return pri

def mix_weights(prior, w_dict):
    d = len(next(iter(w_dict.values())))
    w = [0.0] * d
    for k, p in prior.items():
        wk = w_dict.get(k)
        if not wk:
            continue
        for i in range(d):
            w[i] += p * wk[i]
    s = sum(w)
    if s > 0:
        w = [x / s for x in w]
    return w

# ------------------------------------------------------------
# Scoring and MMR
# ------------------------------------------------------------

def score_with_weights(feat, w):
    x = vectorize(feat)
    base = sum(w[i] * x[i] for i in range(len(w)))
    base *= (0.85 + 0.15 * feat["f_trust"])
    return base

def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union

def mmr_rerank(items, scores, k=50, lambda_=0.75):
    selected, selected_idx = [], set()
    cand_idx = list(range(len(items)))
    while len(selected) < min(k, len(items)) and cand_idx:
        best_i, best_val = None, -1e9
        for i in cand_idx:
            rel = scores[i]
            if not selected:
                div = 0.0
            else:
                sim_to_S = max(jaccard(items[i]["title_tokens"], items[j]["title_tokens"]) for j in selected_idx)
                div = sim_to_S
            val = lambda_ * rel - (1 - lambda_) * div
            if val > best_val:
                best_val, best_i = val, i
        selected.append(items[best_i])
        selected_idx.add(best_i)
        cand_idx.remove(best_i)
    return selected

# ------------------------------------------------------------
# Training and pipeline
# ------------------------------------------------------------

def train_expert_weights(features, view, seed=42):
    labels = [weak_label(f, view=view) for f in features]
    pairs = build_pairs(features, labels, max_pairs_per_query=2000)
    if not pairs:
        return [1.0 / len(FEATURE_KEYS)] * len(FEATURE_KEYS)
    w0 = [1.0 / len(FEATURE_KEYS)] * len(FEATURE_KEYS)
    w = coordinate_ascent_nonneg(w0, pairs, iters=300, step=0.05, lam=1e-3, seed=seed)
    return w

def learn_or_default_weights(features, views=("review", "application", "theory", "trending"), seed=42):
    w_dict = {}
    for v in views:
        w_dict[v] = train_expert_weights(features, v, seed=seed)
    return w_dict

def run_pipeline():
    print("running recommend_v3")
    print(f"input file: {INPUT_JSON}")
    ensure_dir(OUTPUT_DIR)

    records = load_records(INPUT_JSON)
    print(f"loaded records: {len(records)}")

    feats = build_features(records, fallback_query=FALLBACK_QUERY)
    print(f"built features: {len(feats)}")

    random.seed(SEED)
    w_dict = learn_or_default_weights(feats, seed=SEED)
    print("learned expert weights:")
    for v, wv in w_dict.items():
        parts = [f"{FEATURE_KEYS[i]}={wv[i]:.3f}" for i in range(len(wv))]
        print(f"  {v}: " + ", ".join(parts))

    if VIEWS == "all":
        target_views = ["default", "review", "application", "theory", "trending"]
    else:
        target_views = [VIEWS]

    by_q = defaultdict(list)
    for f in feats:
        by_q[f["qid"]].append(f)

    for view in target_views:
        out_all = []
        for qid, group in by_q.items():
            if view == "default":
                prior = intent_prior_from_query(qid)
                w = mix_weights(prior, w_dict)
            else:
                w = w_dict.get(view)
                if w is None:
                    w = [1.0 / len(FEATURE_KEYS)] * len(FEATURE_KEYS)

            scores = [score_with_weights(f, w) for f in group]

            if TOPK > 0:
                reranked = mmr_rerank(group, scores, k=TOPK, lambda_=MMR_LAMBDA)
            else:
                reranked = [x for _, x in sorted(zip(scores, group), key=lambda t: t[0], reverse=True)]

            rank = 1
            for f in reranked:
                rec = dict(f["_raw"])
                x = vectorize(f)
                contrib = {FEATURE_KEYS[i]: round(w[i] * x[i], 6) for i in range(len(w))}
                rec["_view"] = view
                rec["_qid"] = qid
                rec["_score"] = round(score_with_weights(f, w), 6)
                rec["_rank"] = rank
                rec["_weights"] = {FEATURE_KEYS[i]: round(w[i], 6) for i in range(len(w))}
                rec["_features"] = {k: round(f[k], 6) for k in FEATURE_KEYS}
                rec["_contrib"] = contrib
                out_all.append(rec)
                rank += 1

        out_path = OUTPUT_DIR / f"recommend_{view}.json"
        with open(out_path, "w", encoding="utf-8") as fo:
            for r in out_all:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"saved: {out_path} ({len(out_all)} lines)")



if __name__ == "__main__":
    t0 = perf_counter()
    run_pipeline()
    print(f"finished in: {perf_counter() - t0:.2f}s")
