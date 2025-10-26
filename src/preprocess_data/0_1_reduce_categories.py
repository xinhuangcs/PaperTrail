
# Filter arXiv snapshots to retain only cs.* / stat.* / eess.*
import os
import gzip
import json
from collections import Counter
from tqdm import tqdm
from pathlib import Path




# Configuration:
ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH  = ROOT_DIR / "data" / "preprocess" / "arxiv-metadata-oai-snapshot.json"
OUTPUT_PATH = ROOT_DIR / "data" / "preprocess" / "arxiv-cs-data.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

LIMIT       = 0
PRECOUNT    = True              # True: Count total rows to let tqdm estimate remaining time
CHUNK_BYTES = 8 * 1024 * 1024
USE_ORJSON  = True
KEEP_PREFIXES = ("cs.", "stat.", "eess.")

if USE_ORJSON:
    try:
        import orjson as _fastjson
        def json_loads(s):
            return _fastjson.loads(s if isinstance(s, (bytes, bytearray)) else s.encode("utf-8"))
    except Exception:
        def json_loads(s): return json.loads(s)
else:
    def json_loads(s): return json.loads(s)

def open_text(path, mode):
    return gzip.open(path, mode + "t", encoding="utf-8", newline="") if path.endswith(".gz") else open(path, mode, encoding="utf-8", newline="")

def open_bin(path, mode):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)

def parse_categories(s):
    return [t.strip() for t in s.split()] if s else []

def is_kept(cat_str):
    for c in parse_categories(cat_str):
        if c.startswith(KEEP_PREFIXES):
            return True
    return False

def tally(cat_str, counter: Counter):
    for c in parse_categories(cat_str):
        if   c.startswith("cs.")  : counter["Computer Science"] += 1
        elif c.startswith("stat."): counter["Statistics"] += 1
        elif c.startswith("eess."): counter["EESS"] += 1

def count_lines(path):
    total = 0
    with open_text(path, "r") as f:
        for _ in f: total += 1
    return total

def iter_lines_chunked(path, chunk_bytes):
    with open_bin(path, "rb") as fb:
        rem = ""
        while True:
            buf = fb.read(chunk_bytes)
            if not buf: break
            s = rem + buf.decode("utf-8", errors="ignore")
            parts = s.split("\n")
            rem = parts.pop()
            for ln in parts: yield ln
        if rem: yield rem

def iter_lines_default(path):
    with open_text(path, "r") as f:
        for ln in f: yield ln.rstrip("\n")

def filter_file(in_path, out_path, limit=0, precount=True, chunk_bytes=8*1024*1024):
    total = count_lines(in_path) if precount else None
    line_iter = iter_lines_chunked(in_path, chunk_bytes) if chunk_bytes > 0 else iter_lines_default(in_path)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    kept = skipped = total_read = 0
    by_domain = Counter()

    with open_text(out_path, "w") as fout, tqdm(total=total, unit="line", desc="过滤中", mininterval=0.5) as pbar:
        for raw in line_iter:
            total_read += 1
            line = raw.strip()
            if not line:
                pbar.update(1); continue
            try:
                obj = json_loads(line)
            except Exception:
                skipped += 1; pbar.update(1); continue

            cats = obj.get("categories", "")
            if is_kept(cats):
                tally(cats, by_domain)
                fout.write(line if line.endswith("\n") else line + "\n")
                kept += 1
                if limit and kept >= limit:
                    pbar.update(1)
                    break
            else:
                skipped += 1
            pbar.update(1)

    return {
        "input": in_path, "output": out_path, "total_read": total_read,
        "kept": kept, "skipped": skipped, "by_domain": dict(by_domain)
    }

def main():
    stats = filter_file(INPUT_PATH, OUTPUT_PATH, LIMIT, PRECOUNT, CHUNK_BYTES)
    print("\n=== Summary ===")
    print(f"Input:     {stats['input']}")
    print(f"Output:    {stats['output']}")
    print(f"Read:      {stats['total_read']:,}")
    print(f"Kept:      {stats['kept']:,}")
    print(f"Skipped:   {stats['skipped']:,}")
    if stats["by_domain"]:
        print("By category:")
        for k, v in sorted(stats["by_domain"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {k}: {v:,}")


if __name__ == "__main__":
    main()
