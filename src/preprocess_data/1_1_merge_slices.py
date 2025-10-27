"""
Merge JSONL slices

Dependencies:
  pip install tqdm
"""

import argparse
import glob
import os
import sys
from tqdm import tqdm
from pathlib import Path



# 1) Config
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG = {
    # Glob pattern to match all slice files
    "input_glob": ROOT_DIR / "data" / "preprocess" / "arxiv-cs-data-with-citations-final-dataset-*.json",
    "output_file": ROOT_DIR / "data" / "preprocess" / "arxiv-cs-data-with-citations-final-dataset.json",

    "show_progress": True,
}
CONFIG["output_file"].parent.mkdir(parents=True, exist_ok=True)


def parse_args():
    #CLI args to override CONFIG
    p = argparse.ArgumentParser(description="Merge JSONL slices as-is (no order checks)")
    p.add_argument("--input_glob", default=CONFIG["input_glob"], help="Glob for slice files")
    p.add_argument("--output_file", default=CONFIG["output_file"], help="Merged output file")
    p.add_argument("--no_progress", action="store_true", help="Disable progress bar")
    return p.parse_args()


def main():
    args = parse_args()
    show_progress = CONFIG["show_progress"] and (not args.no_progress)
    files = glob.glob(str(CONFIG["input_glob"]))
    if not files:
        print(f"No files matched: {args.input_glob}", file=sys.stderr)
        sys.exit(2)
    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    total_lines = 0
    total_files = 0
    #Merge
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        iterator = files
        if show_progress:
            iterator = tqdm(files, desc="Merging (any order)", unit="file")

        for fp in iterator:
            total_files += 1
            # Stream read & write to save memory
            try:
                with open(fp, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_lines += 1
            except Exception as e:
                # Log and continue on file read errors
                print(f"Failed to read: {fp} ({type(e).__name__}: {e})", file=sys.stderr)
                continue

    print(f"Merge done → {args.output_file}")
    print(f"[i] Files: {total_files} | Lines: {total_lines}")


if __name__ == "__main__":
    main()
