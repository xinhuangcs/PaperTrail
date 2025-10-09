"""
Retry and process records where citation_count is -1

Dependencies:
  pip install requests tqdm
"""

import json
import os
import sys
import time
import shutil
import signal
from typing import Optional, Tuple, Dict

import requests
from difflib import SequenceMatcher
from tqdm import tqdm


# 1) Configuration
CONFIG = {
    "INPUT_FILE": "/work3/s242644/PaperTrail/processed/40/arxiv-cs-data-with-citations-refreshed-2.json",
    "OUTPUT_FILE": "/work3/s242644/PaperTrail/processed/40/arxiv-cs-data-with-citations-refreshed.json",
    "CACHE_FILE": "/work3/s242644/PaperTrail/processed/40/citation_cache.json",
    "SLEEP_SECS": 1,
    "SAVE_EVERY": 1000,
    "TITLE_SIM_RATIO": 0.90,
}


# 2) Utility functions
def normalize_doi(raw: Optional[str]) -> Optional[str]:
    if not raw or not isinstance(raw, str):
        return None
    doi = raw.strip()
    if not doi:
        return None
    doi = doi.replace(" ", "")
    low = doi.lower()
    if low.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    elif low.startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/"):]
    return doi.lower()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def get_first_n_words(title: str, n: int = 5) -> str:
    """Extract first n words from title"""
    if not title or not isinstance(title, str):
        return ""
    words = title.strip().split()
    return " ".join(words[:n])

#3) OpenAlex
def openalex_get_by_doi(doi: str, session: requests.Session,
                        sleep_secs: float = 0.2,
                        max_retries: int = 5,
                        timeout: float = 20.0) -> Tuple[int, str]:
    base = "https://api.openalex.org/works"
    params = {"filter": f"doi:{doi}"}
    attempt = 0
    while True:
        try:
            resp = session.get(base, params=params, timeout=timeout)
            code = resp.status_code
            if code == 200:
                data = resp.json()
                results = data.get("results", [])
                nd = normalize_doi(doi)
                for item in results:
                    item_doi = normalize_doi(item.get("doi"))
                    if item_doi and item_doi == nd:
                        cbc = item.get("cited_by_count")
                        if isinstance(cbc, int):
                            time.sleep(sleep_secs)
                            return cbc, "ok_openalex_doi"
                        else:
                            return -1, "openalex_no_cited_by_count"
                return -1, "openalex_not_found_doi"
            elif code in (429, 500, 502, 503, 504):
                attempt += 1
                if attempt > max_retries:
                    return -1, f"openalex_http_{code}_fail"
                backoff = sleep_secs * (2 ** attempt) + 0.05 * attempt
                time.sleep(backoff)
                continue
            else:
                return -1, f"openalex_http_{code}"
        except requests.RequestException as e:
            attempt += 1
            if attempt > max_retries:
                return -1, f"openalex_exception:{type(e).__name__}"
            backoff = sleep_secs * (2 ** attempt) + 0.05 * attempt
            time.sleep(backoff)


def openalex_get_by_title(title: str, session: requests.Session,
                          sleep_secs: float = 0.2,
                          timeout: float = 20.0,
                          min_ratio: float = 0.9) -> Tuple[int, str]:
    base = "https://api.openalex.org/works"
    params = {"filter": f"title.search:{title}", "per-page": 5}
    try:
        resp = session.get(base, params=params, timeout=timeout)
        if resp.status_code != 200:
            return -1, f"openalex_title_http_{resp.status_code}"
        data = resp.json()
        results = data.get("results", [])
        best_item, best_sim = None, 0.0
        for item in results:
            cand_title = item.get("title", "") or ""
            sim = title_similarity(title, cand_title)
            if sim > best_sim:
                best_sim = sim
                best_item = item
        if best_item and best_sim >= min_ratio:
            cbc = best_item.get("cited_by_count")
            if isinstance(cbc, int):
                time.sleep(sleep_secs)
                return cbc, "ok_openalex_title"
        return -1, "openalex_title_not_found"
    except requests.RequestException as e:
        return -1, f"openalex_title_exception:{type(e).__name__}"


# 4) Cache
def load_cache(cache_path: Optional[str]) -> Dict[str, Dict[str, object]]:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def atomic_save_json(obj, path: str):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    shutil.move(tmp, path)


# 5) Main process (Second-pass for records with citation_count == -1)
def main():
    INPUT_FILE      = CONFIG["INPUT_FILE"]
    OUTPUT_FILE     = CONFIG["OUTPUT_FILE"]
    CACHE_FILE      = CONFIG["CACHE_FILE"]
    SLEEP_SECS      = float(CONFIG["SLEEP_SECS"])
    SAVE_EVERY      = int(CONFIG["SAVE_EVERY"])
    TITLE_SIM_RATIO = float(CONFIG["TITLE_SIM_RATIO"])

    print(f"[i] Parameters:")
    print(f"    INPUT_FILE   = {INPUT_FILE}")
    print(f"    OUTPUT_FILE  = {OUTPUT_FILE}")
    print(f"    CACHE_FILE   = {CACHE_FILE}")
    print(f"    SLEEP_SECS   = {SLEEP_SECS}")
    print(f"    TITLE_SIM    = {TITLE_SIM_RATIO}")

    # Exit handling
    interrupted = {"flag": False}
    def handle_sigint(signum, frame):
        interrupted["flag"] = True
        print("\n Interrupted, saving cache", file=sys.stderr)
    signal.signal(signal.SIGINT, handle_sigint)

    # Cache
    cache = load_cache(CACHE_FILE)
    print(f"[i] Cache loaded: {len(cache)} entries")

    # First pass: count how many records need re-check (citation_count == -1)
    neg_count = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("citation_count", None) == -1:
                neg_count += 1

    if neg_count == 0:
        print("[i] No records with citation_count == -1. Copying through without changes...")
        with open(INPUT_FILE, "r", encoding="utf-8") as in_f, \
             open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            for line in in_f:
                out_f.write(line)
        print("Done. Nothing to update.")
        return

    print(f"[i] Records to re-check (citation_count == -1): {neg_count}")

    # Output preparation
    out_dir = os.path.dirname(os.path.abspath(OUTPUT_FILE))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_f = open(OUTPUT_FILE, "w", encoding="utf-8")

    session = requests.Session()
    session.headers.update({"User-Agent": "dtuproject-citation-enricher/second-pass/1.0"})

    processed_neg = 0      # number of -1 records processed
    written_total = 0      # total lines written
    last_cache_dump = 0

    # Second pass: stream and only re-query where citation_count == -1
    with open(INPUT_FILE, "r", encoding="utf-8") as in_f, \
         tqdm(total=neg_count, unit="rec", desc="Re-check (-1 only)") as pbar:

        for line in in_f:
            if interrupted["flag"]:
                break

            raw = line.strip()
            if not raw:
                out_f.write(line)
                continue

            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                out_f.write(line)
                continue

            # If this record does NOT need re-check, write through as-is and do not advance the bar
            if rec.get("citation_count", None) != -1:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written_total += 1
                continue

            # For -1 records: try again using DOI first, then title
            doi = normalize_doi(rec.get("doi"))
            title = (rec.get("title") or "").strip()

            new_count, new_status = -1, rec.get("citation_status", "missing_id")

            if doi:
                if doi in cache:
                    new_count = cache[doi]["count"]
                    new_status = cache[doi]["status"]
                else:
                    new_count, new_status = openalex_get_by_doi(doi, session, sleep_secs=SLEEP_SECS)
                    cache[doi] = {"count": new_count, "status": new_status}
            else:
                if title:
                    cache_key = f"title::{title}"
                    if cache_key in cache:
                        new_count = cache[cache_key]["count"]
                        new_status = cache[cache_key]["status"]
                    else:
                        # First try with full title
                        new_count, new_status = openalex_get_by_title(
                            title, session, sleep_secs=SLEEP_SECS, min_ratio=TITLE_SIM_RATIO
                        )
                        
                        # If not found, try with first 5 words
                        if new_count == -1:
                            short_title = get_first_n_words(title, 5)
                            if short_title and short_title != title:
                                cache_key_short = f"title::{short_title}"
                                if cache_key_short in cache:
                                    new_count = cache[cache_key_short]["count"]
                                    new_status = cache[cache_key_short]["status"]
                                else:
                                    new_count, new_status = openalex_get_by_title(
                                        short_title, session, sleep_secs=SLEEP_SECS, min_ratio=TITLE_SIM_RATIO
                                    )
                                    cache[cache_key_short] = {"count": new_count, "status": new_status}
                        
                        cache[cache_key] = {"count": new_count, "status": new_status}
                else:
                    pass

            # If found (new_count != -1), update citation_count and citation_status; otherwise keep original
            if isinstance(new_count, int) and new_count != -1:
                rec["citation_count"] = new_count
                rec["citation_status"] = new_status

            # write record (updated or unchanged)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written_total += 1

            # update counters and progress bar for -1 records only
            processed_neg += 1
            pbar.update(1)

            # periodic cache flush (based on -1 processed count)
            if CACHE_FILE and (processed_neg - last_cache_dump) >= SAVE_EVERY:
                atomic_save_json(cache, CACHE_FILE)
                last_cache_dump = processed_neg

    # final cache flush
    if CACHE_FILE:
        atomic_save_json(cache, CACHE_FILE)

    out_f.close()
    print(f"Second pass done. -1 processed={processed_neg}, total written={written_total}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
