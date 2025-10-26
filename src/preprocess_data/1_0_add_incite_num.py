"""
Add citation numbers (OpenAlex):
- With DOI: use OpenAlex (DOI)
- Without DOI: use OpenAlex title search (similarity matching)
- If citation count is not available, write -1 and specify citation_status
- Process line by line (streaming)
- Slice
- Use tqdm to show progress / speed / estimated remaining time for each slice

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
from pathlib import Path



# 1) Config
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG = {
    "INPUT_FILE": ROOT_DIR / "data" / "preprocess" / "arxiv-cs-data.json",
    "OUTPUT_FILE_BASE": ROOT_DIR / "data" / "preprocess" / "arxiv-cs-data-with-citations",
    "CACHE_FILE": ROOT_DIR / "data" / "preprocess" / "citation_cache.json",

    # Slicing (starting from 0, e.g.: split into 10 slices, run the 2nd slice → SLICE_COUNT=10, SLICE_INDEX=1)
    "SLICE_COUNT": 20,
    "SLICE_INDEX": 6,


    # Rate control (to avoid throttling)
    "SLEEP_SECS": 0.2,  # Pause in seconds after each request (recommended 0.15–0.25)
    "SAVE_EVERY": 1000,  # Save cache after processing this many records
    "TITLE_SIM_RATIO": 0.90,  # Title similarity threshold
}
CONFIG["OUTPUT_FILE"].parent.mkdir(parents=True, exist_ok=True)

# 2) Utility functions
def normalize_doi(raw: Optional[str]) -> Optional[str]:
    #Normalize DOI: trim whitespace, remove (https)://doi.org/ prefix, convert to lowercase
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
    #Title similarity [0,1]
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


#3) OpenAlex
def openalex_get_by_doi(doi: str, session: requests.Session,
                        sleep_secs: float = 0.2,
                        max_retries: int = 3,
                        timeout: float = 20.0) -> Tuple[int, str]:
    #Use DOI to query OpenAlex, return (citation_count, status)
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
    #Search OpenAlex by title (return citation count of the closest match)
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
    #Load cache JSON: {key: {"count": int, "status": str}},  where key can be a DOI or 'title::<title>'
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


# 5) Main process
def main():
    # Load configuration
    INPUT_FILE      = CONFIG["INPUT_FILE"]
    OUTPUT_FILE_BASE= CONFIG["OUTPUT_FILE_BASE"]
    CACHE_FILE      = CONFIG["CACHE_FILE"]
    SLICE_COUNT     = int(CONFIG["SLICE_COUNT"])
    SLICE_INDEX     = int(CONFIG["SLICE_INDEX"])
    SLEEP_SECS      = float(CONFIG["SLEEP_SECS"])
    SAVE_EVERY      = int(CONFIG["SAVE_EVERY"])
    TITLE_SIM_RATIO = float(CONFIG["TITLE_SIM_RATIO"])

    # Output file is automatically named based on slice index
    OUTPUT_FILE = f"{OUTPUT_FILE_BASE}_slice{SLICE_INDEX}.jsonl"

    print(f"[i] Parameters:")
    print(f"    INPUT_FILE  = {INPUT_FILE}")
    print(f"    OUTPUT_FILE = {OUTPUT_FILE}")
    print(f"    CACHE_FILE  = {CACHE_FILE}")
    print(f"    SLICE       = {SLICE_INDEX}/{SLICE_COUNT - 1}")
    print(f"    SLEEP_SECS  = {SLEEP_SECS}")

    # Exit handling
    interrupted = {"flag": False}

    def handle_sigint(signum, frame):
        interrupted["flag"] = True
        print("\n Interrupted, saving cache", file=sys.stderr)

    signal.signal(signal.SIGINT, handle_sigint)

    # Cache
    cache = load_cache(CACHE_FILE)
    print(f"[i] Cache loaded: {len(cache)} entries")
    # Count total lines and compute slice range
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    if SLICE_COUNT < 1:
        SLICE_COUNT = 1
    if SLICE_INDEX < 0 or SLICE_INDEX >= SLICE_COUNT:
        print(f"SLICE_INDEX out of range (0..{SLICE_COUNT - 1})", file=sys.stderr)
        sys.exit(2)
    lines_per_slice = total_lines // SLICE_COUNT
    start_line = SLICE_INDEX * lines_per_slice + 1
    end_line = (SLICE_INDEX + 1) * lines_per_slice if SLICE_INDEX < SLICE_COUNT - 1 else total_lines
    total_this_slice = end_line - start_line + 1
    print(f"[i] Total lines={total_lines}, this slice range={start_line} ~ {end_line} (total {total_this_slice} lines)")

    # Output preparation
    out_dir = os.path.dirname(os.path.abspath(OUTPUT_FILE))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_f = open(OUTPUT_FILE, "w", encoding="utf-8")

    session = requests.Session()
    session.headers.update({"User-Agent": "dtuproject-citation-enricher/1.0"})

    processed, written, last_cache_dump = 0, 0, 0

    # tqdm only covers the current slice
    with open(INPUT_FILE, "r", encoding="utf-8") as in_f, \
         tqdm(total=total_this_slice, unit="rec", desc=f"Slice {SLICE_INDEX}/{SLICE_COUNT-1}") as pbar:
        for lineno, line in enumerate(in_f, start=1):
            if lineno < start_line:
                continue
            if lineno > end_line:
                break
            if interrupted["flag"]:
                break

            line = line.strip()
            if not line:
                pbar.update(1)
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                pbar.update(1)
                continue

            doi = normalize_doi(rec.get("doi"))
            title = (rec.get("title") or "").strip()

            citation_count, citation_status = -1, "missing_id"

            if doi:
                # Use DOI as cache key
                if doi in cache:
                    citation_count = cache[doi]["count"]
                    citation_status = cache[doi]["status"]
                else:
                    # OpenAlex (DOI only)
                    citation_count, citation_status = openalex_get_by_doi(doi, session, sleep_secs=SLEEP_SECS)
                    cache[doi] = {"count": citation_count, "status": citation_status}
            else:
                # No DOI → search by title (OpenAlex), cache key uses title::
                if title:
                    cache_key = f"title::{title}"
                    if cache_key in cache:
                        citation_count = cache[cache_key]["count"]
                        citation_status = cache[cache_key]["status"]
                    else:
                        citation_count, citation_status = openalex_get_by_title(
                            title, session, sleep_secs=SLEEP_SECS, min_ratio=TITLE_SIM_RATIO
                        )
                        cache[cache_key] = {"count": citation_count, "status": citation_status}
                else:
                    citation_count, citation_status = -1, "no_doi_no_title"

            # Write back
            rec["citation_count"] = citation_count
            rec["citation_status"] = citation_status
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            written += 1
            processed += 1

            # Periodically save cache
            if CACHE_FILE and (processed - last_cache_dump) >= CONFIG["SAVE_EVERY"]:
                atomic_save_json(cache, CACHE_FILE)
                last_cache_dump = processed

            pbar.update(1)

        # Final cache save
        if CACHE_FILE:
            atomic_save_json(cache, CACHE_FILE)

        out_f.close()
        print(f"Processing completed: processed={processed}, written={written}")
        print(f"Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
