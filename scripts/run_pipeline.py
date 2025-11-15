#!/usr/bin/env python3
# Temporary simple pipeline: just write a fake plan json for testing.

import argparse, json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PLANS_DIR = ROOT / "website" / "plans"
PLANS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", required=True)
    ap.add_argument("--top_k", default="10")
    ap.add_argument("--issue", required=True)
    args = ap.parse_args()

    plan = {
        "goal": args.goal,
        "study_level": "intermediate",
        "source_papers": ["dummy-001", "dummy-002"],
        "plan_overview": (
            "This is a fake learning plan for testing the end-to-end pipeline.\n"
            "Later this JSON will be replaced by real output from your LLM pipeline."
        ),
        "reading_order": [
            {
                "paper_id": "dummy-001",
                "why_first": "Just a placeholder paper to test rendering.",
                "key_questions": [
                    "What is the main idea?",
                    "How does this help me reach my goal?"
                ]
            },
            {
                "paper_id": "dummy-002",
                "why_first": "Second placeholder paper.",
                "key_questions": [
                    "What is different from the first one?"
                ]
            }
        ],
        "actions": [
            {
                "label": "Read the first dummy paper",
                "how_to": "Pretend to read it and take notes.",
                "expected_outcome": "You confirm the page renders correctly."
            }
        ],
        "metrics": [
            "Plan JSON can be loaded in plan.html",
            "Sections (overview, actions, etc.) are visible"
        ],
        "timeline_weeks": [
            {"week": 1, "focus": "Just testing", "deliverable": "Visual check of the page"}
        ],
        "risks": [
            {"risk": "Real pipeline not wired yet", "mitigation": "Your teammate will replace this script later."}
        ],
        "metadata": {
            "prompt_version": "test-v0",
            "model": "fake-model",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    }

    out_path = PLANS_DIR / f"{args.issue}.json"
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    # optional index
    index_path = PLANS_DIR / "index.json"
    try:
        listing = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        listing = []
    listing = [x for x in listing if str(x.get("id")) != str(args.issue)]
    listing.insert(0, {
        "id": int(args.issue),
        "goal": args.goal,
        "top_k": int(args.top_k),
        "ts_utc": datetime.now(timezone.utc).isoformat()
    })
    index_path.write_text(json.dumps(listing, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote fake plan -> {out_path}")

if __name__ == "__main__":
    main()
