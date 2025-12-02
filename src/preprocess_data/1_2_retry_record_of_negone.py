
"""
Retry and process records where citation_count is -1

Dependencies:
  pip install requests tqdm

Enter your email address and purpose in lines 307-309 to join the openalex polite pool.
"""

import json
import os
import sys
import time
import shutil
import signal
import re
from typing import Optional, Tuple, Dict

import requests
from difflib import SequenceMatcher
from tqdm import tqdm
from pathlib import Path



# 1) Config
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG = {
    "INPUT_FILE": ROOT_DIR / "data" / "preprocess" / "arxiv-cs-data-with-citations_merged_zrk_5.json",
    "OUTPUT_FILE": ROOT_DIR / "data" / "preprocess" / "arxiv-cs-data-with-citations_final_dataset_odd.json",
    "CACHE_FILE": ROOT_DIR / "data" / "preprocess" / "citation_cache.json",
    "SLEEP_SECS": 0.25,
    "SAVE_EVERY": 100,
    "TITLE_SIM_RATIO": 0.80,
}
CONFIG["OUTPUT_FILE"].parent.mkdir(parents=True, exist_ok=True)
CONFIG["CACHE_FILE"].parent.mkdir(parents=True, exist_ok=True)


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



def normalize_search_term(text: Optional[str], max_words: int) -> Optional[str]:
    """
    Cleans text by lowercasing, removing special characters/punctuation,
    and truncating it to the specified number of words.
    This creates a canonical search fragment optimized for API matching.
    """
    if not text or not isinstance(text, str):
        return None

    text = text.lower().strip()
    text = text.replace('\n', ' ').replace('\t', ' ')
    cleaned_text = re.sub(r'[^\w\s]', '', text)

    tokens = [t for t in cleaned_text.split() if t]
    if not tokens:
        return None
    truncated_tokens = tokens[:max_words]
    return " ".join(truncated_tokens)



def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

#3) OpenAlex
def openalex_get_by_doi(doi: str, session: requests.Session,
                        sleep_secs: float = 0.2,
                        max_retries: int = 3,
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
    params = {"filter": f"title.search:{title}", "per-page": 15}
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



def openalex_get_by_truncated_title(original_title: str, truncated_term: str,
                                    session: requests.Session,
                                    sleep_secs: float = 0.2,
                                    timeout: float = 20.0,
                                    min_ratio: float = 0.9) -> Tuple[int, str]:

    base = "https://api.openalex.org/works"
    # Use title.search for targeted title matching with the normalized term
    params = {"filter": f"title.search:{truncated_term}", "per-page": 15}

    try:
        resp = session.get(base, params=params, timeout=timeout)
        if resp.status_code != 200:
            return -1, f"openalex_title4_http_{resp.status_code}"

        data = resp.json()
        results = data.get("results", [])
        best_item, best_sim = None, 0.0

        for item in results:
            cand_title = item.get("title", "") or ""
            sim = title_similarity(original_title, cand_title)
            if sim > best_sim:
                best_sim = sim
                best_item = item

        if best_item and best_sim >= min_ratio:
            cbc = best_item.get("cited_by_count")
            if isinstance(cbc, int):
                time.sleep(sleep_secs)
                return cbc, "ok_openalex_title_4"  # New status code

        return -1, "openalex_title_4_not_found"
    except requests.RequestException as e:
        return -1, f"openalex_title_4_exception:{type(e).__name__}"


def openalex_get_by_truncated_abstract(original_title: str, truncated_term: str,
                                       session: requests.Session,
                                       sleep_secs: float = 0.2,
                                       timeout: float = 20.0,
                                       min_ratio: float = 0.8) -> Tuple[int, str]:

    base = "https://api.openalex.org/works"
    params = {"filter": f"abstract.search:{truncated_term}", "per-page": 15}

    try:
        resp = session.get(base, params=params, timeout=timeout)
        if resp.status_code != 200:
            return -1, f"openalex_abstract_http_{resp.status_code}"

        data = resp.json()
        results = data.get("results", [])
        best_item, best_sim = None, 0.0

        for item in results:
            cand_title = item.get("title", "") or ""
            sim = title_similarity(original_title, cand_title)
            if sim > best_sim:
                best_sim = sim
                best_item = item

        if best_item and best_sim >= min_ratio:
            cbc = best_item.get("cited_by_count")
            if isinstance(cbc, int):
                time.sleep(sleep_secs)
                return cbc, "ok_openalex_abstract"  # New status code

        return -1, "openalex_abstract_not_found"
    except requests.RequestException as e:
        return -1, f"openalex_abstract_exception:{type(e).__name__}"





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
    session.headers.update({
        "User-Agent": "XXX",
        "mailto": "XXX"
    })

    #Count how many HTTP requests are sent
    request_counter = {"count": 0}
    _original_get = session.get
    def counted_get(*args, **kwargs):
        request_counter["count"] += 1
        return _original_get(*args, **kwargs)
    session.get = counted_get

    processed_neg = 0      # number of -1 records processed
    written_total = 0      # total lines written
    last_cache_dump = 0
    #statistics
    stats = {
        "doi_attempt": 0, "title_attempt": 0, "title4_attempt": 0, "abs_attempt": 0,
        "doi_ok": 0, "title_ok": 0, "title4_ok": 0, "abs_ok": 0,
        "doi_nf": 0, "title_nf": 0, "title4_nf": 0, "abs_nf": 0,
        # Skip Count (No Content)
        "title4_skipped": 0, "abs_skipped": 0,
        "final_ok": 0, "final_neg1": 0,
    }


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

            # For -1 records: try   4-step retrieval pipeline
            doi = normalize_doi(rec.get("doi"))
            title = (rec.get("title") or "").strip()
            abstract = (rec.get("abstract") or "").strip()

            new_count, new_status = -1, rec.get("citation_status", "missing_id")

            # PRIORITY 1: DOI Search
            if doi:
                stats["doi_attempt"] += 1
                if doi in cache:
                    new_count = cache[doi]["count"]
                    new_status = cache[doi]["status"]
                else:
                    new_count, new_status = openalex_get_by_doi(doi, session, sleep_secs=SLEEP_SECS)
                    cache[doi] = {"count": new_count, "status": new_status}
                if new_status == "ok_openalex_doi" and isinstance(new_count, int) and new_count != -1:
                    stats["doi_ok"] += 1
                elif doi:
                    stats["doi_nf"] += 1

            #PRIORITY 2:Title Search
            if new_count == -1 and title:
                stats["title_attempt"] += 1
                cache_key = f"title::{title}"
                if cache_key in cache:
                    new_count = cache[cache_key]["count"]
                    new_status = cache[cache_key]["status"]
                else:
                    new_count, new_status = openalex_get_by_title(
                        title, session, sleep_secs=SLEEP_SECS, min_ratio=TITLE_SIM_RATIO
                    )
                    cache[cache_key] = {"count": new_count, "status": new_status}
                if new_status == "ok_openalex_title" and isinstance(new_count, int) and new_count != -1:
                    stats["title_ok"] += 1
                elif new_count == -1:
                    stats["title_nf"] += 1

            #PRIORITY 3 : Truncated Title Search
            if new_count == -1 and title:
                search_term_4 = normalize_search_term(title, max_words=4)
                if not search_term_4:
                    stats["title4_skipped"] += 1

                if search_term_4:
                    stats["title4_attempt"] += 1
                    cache_key_4 = f"title_4::{search_term_4}"  # New cache key format

                    if cache_key_4 in cache:
                        new_count = cache[cache_key_4]["count"]
                        new_status = cache[cache_key_4]["status"]
                    else:
                        new_count, new_status = openalex_get_by_truncated_title(
                            original_title=title,
                            truncated_term=search_term_4,
                            session=session,
                            sleep_secs=SLEEP_SECS,
                            min_ratio=TITLE_SIM_RATIO
                        )
                        cache[cache_key_4] = {"count": new_count, "status": new_status}
                    if new_status == "ok_openalex_title_4" and isinstance(new_count, int) and new_count != -1:
                        stats["title4_ok"] += 1
                    elif new_count == -1:
                        stats["title4_nf"] += 1

            #PRIORITY 4 :Truncated Abstract Search
            if new_count == -1 and abstract:
                search_term_ab = normalize_search_term(abstract, max_words=150)
                if not search_term_ab:
                    stats["abs_skipped"] += 1

                if search_term_ab:
                    stats["abs_attempt"] += 1
                    cache_key_ab = f"abstract::{search_term_ab}"

                    if cache_key_ab in cache:
                        new_count = cache[cache_key_ab]["count"]
                        new_status = cache[cache_key_ab]["status"]
                    else:
                        new_count, new_status = openalex_get_by_truncated_abstract(
                            original_title=title,
                            truncated_term=search_term_ab,
                            session=session,
                            sleep_secs=SLEEP_SECS,
                            min_ratio=TITLE_SIM_RATIO
                        )
                        cache[cache_key_ab] = {"count": new_count, "status": new_status}
                    if new_status == "ok_openalex_abstract" and isinstance(new_count, int) and new_count != -1:
                        stats["abs_ok"] += 1
                    elif new_count == -1:
                        stats["abs_nf"] += 1

            rec["citation_status"] = new_status

            # Statistics of Final Results
            if isinstance(new_count, int) and new_count != -1:
                rec["citation_count"] = new_count
                stats["final_ok"] += 1
            else:
                stats["final_neg1"] += 1


            # write record (updated or unchanged)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written_total += 1

            # update counters and progress bar for -1 records only
            processed_neg += 1
            pbar.update(1)

            if pbar.n % 50 == 0:
                pbar.set_postfix({
                    "requests": request_counter["count"],
                    "ok": stats["final_ok"],
                    "neg1": stats["final_neg1"]
                }, refresh=True)

            # periodic cache flush (based on -1 processed count)
            if CACHE_FILE and (processed_neg - last_cache_dump) >= SAVE_EVERY:
                atomic_save_json(cache, CACHE_FILE)
                last_cache_dump = processed_neg

    print(
        "Match summary:\n"
        f"  DOI:     attempt={stats['doi_attempt']}, ok={stats['doi_ok']}, nf={stats['doi_nf']}\n"
        f"  Title:   attempt={stats['title_attempt']}, ok={stats['title_ok']}, nf={stats['title_nf']}\n"
        f"  Title4:  attempt={stats['title4_attempt']}, ok={stats['title4_ok']}, nf={stats['title4_nf']}, skipped={stats['title4_skipped']}\n"
        f"  Abstract:attempt={stats['abs_attempt']}, ok={stats['abs_ok']}, nf={stats['abs_nf']}, skipped={stats['abs_skipped']}\n"
        f"  Final:   ok={stats['final_ok']}, neg1={stats['final_neg1']}"
    )

    # final cache flush
    if CACHE_FILE:
        atomic_save_json(cache, CACHE_FILE)

    out_f.close()
    print(f"Second pass done. -1 processed={processed_neg}, total written={written_total}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
