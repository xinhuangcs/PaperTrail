"""
Run the full PaperTrail pipeline on the server (GitHub Actions, or local)
and save the generated learning plan to website/plans/<issue>.json.

Steps:
  1) src/search/similarity_search_v4.py
  2) src/recommend/recommend_v3.py
  3) src/ai_advice/v2/standardize_input.py
  4) src/ai_advice/v2/generate_plan_v3.py

Important:
  - Each step already has its own input/output paths configured inside the file.
    Here we just call them in the right order, so relative paths keep working.
  - The last step is expected to write the final plan JSON to:
        data/ai_advice/plan_latest.json
    Please make sure generate_plan_v3.py uses this path.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# project root = repo root (one level above scripts/)
ROOT = Path(__file__).resolve().parents[1]

# where we expect the final plan from step 4
FINAL_PLAN_PATH = ROOT / "data" / "ai_advice" / "plan_latest.json"

# where GitHub Pages will read plans from
PLANS_DIR = ROOT / "website" / "plans"
PLANS_DIR.mkdir(parents=True, exist_ok=True)


def run_step(label: str, cmd: list[str]) -> None:
    """
    Run a single pipeline step as a subprocess.

    label: short name for printing logs
    cmd:   list like [python, script_path, ...]
    """
    print("\n" + "=" * 70)
    print(f"[step] {label}")
    print("=" * 70)
    print(" ".join(cmd))
    print("=" * 70)

    # run from project root so all relative paths inside scripts still work
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[error] Step '{label}' failed with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[ok] Step '{label}' finished.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True, help="User query / learning goal")
    parser.add_argument("--top_k", type=int, default=10, help="How many papers to retrieve (default: 10)")
    parser.add_argument(
        "--method",
        type=str,
        default="lsa_lsh",
        choices=["lsa_lsh", "lsa", "tfidf"],
        help="Search method for similarity_search_v4 (default: lsa_lsh)",
    )
    parser.add_argument(
        "--issue",
        required=True,
        help="GitHub issue number; used as the plan id and filename",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="trending",
        choices=["trending", "review", "application", "theory"],
        help="Which paper view to use when standardizing input"
    )
    args = parser.parse_args()

    issue_id = str(args.issue).strip()
    if not issue_id:
        print("[error] --issue must not be empty", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Running PaperTrail pipeline")
    print("=" * 70)
    print(f"Goal:   {args.goal}")
    print(f"Top-K:  {args.top_k}")
    print(f"Method: {args.method}")
    print(f"Issue:  {issue_id}")
    print("=" * 70)

    python_exe = sys.executable

    # 1) similarity_search_v4.py
    sim_script = ROOT / "src" / "search" / "similarity_search_v4.py"
    sim_cmd = [
        python_exe,
        str(sim_script),
        "--query",
        args.goal,
        "--top_k",
        str(args.top_k),
        "--method",
        args.method,
    ]
    run_step("similarity_search_v4", sim_cmd)

    # 2) recommend_v3.py
    rec_script = ROOT / "src" / "recommend" / "recommend_v3.py"
    rec_cmd = [python_exe, str(rec_script)]
    run_step("recommend_v3", rec_cmd)

    # 3) standardize_input.py
    std_script = ROOT / "src" / "ai_advice" / "v2" / "standardize_input.py"
    std_cmd = [
        python_exe,
        str(std_script),
        "--mode",
        args.mode,
    ]
    run_step("standardize_input", std_cmd)

    # 4) generate_plan_v3.py
    plan_script = ROOT / "src" / "ai_advice" / "v2" / "generate_plan_v3.py"
    # 如果 generate_plan_v3 支持 --goal / --issue，可以顺便传进去，方便写 metadata
    plan_cmd = [
        python_exe,
        str(plan_script),
        "--goal",
        args.goal,
        "--issue",
        issue_id,
    ]
    run_step("generate_plan_v3", plan_cmd)

    # After step 4, we expect FINAL_PLAN_PATH to exist
    if not FINAL_PLAN_PATH.exists():
        print(f"\n[error] Expected final plan file not found: {FINAL_PLAN_PATH}", file=sys.stderr)
        sys.exit(1)

    # copy to website/plans/<issue>.json for GitHub Pages
    out_path = PLANS_DIR / f"{issue_id}.json"
    shutil.copy2(FINAL_PLAN_PATH, out_path)
    print(f"\n[ok] Copied final plan to -> {out_path}")

    # update website/plans/index.json for listing (optional, but nice to have)
    index_path = PLANS_DIR / "index.json"
    try:
        listing = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        listing = []

    # remove any existing entry for this issue id
    listing = [x for x in listing if str(x.get("id")) != issue_id]

    # prepend new record
    listing.insert(
        0,
        {
            "id": int(issue_id),
            "goal": args.goal,
            "top_k": args.top_k,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    index_path.write_text(json.dumps(listing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Updated index -> {index_path}")

    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
