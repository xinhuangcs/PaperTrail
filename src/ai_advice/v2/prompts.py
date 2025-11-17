from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime, UTC


PROMPT_VERSION = "PT-20251102-1"

#config
ROOT_DIR = Path(__file__).resolve().parents[3]
data_dir = ROOT_DIR / "data" / "ai_advice"



latest_standardize_input = max(
    data_dir.glob("standardize_input_*.json"),
    key=lambda f: f.stat().st_mtime
)
latest_plan_schema = max(
    data_dir.glob("plan_schema_*.json"),
    key=lambda f: f.stat().st_mtime
)

timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M")
CONFIG = {
    "SELECTED_PAPERS_FILE": latest_standardize_input,
    "PLAN_SCHEMA_FILE": latest_plan_schema,
    #output：
    "PROMPT_DOC_FILE": ROOT_DIR / "data" / "ai_advice"/ "docs" / f"prompt_design_{timestamp}.md",
    "SYSTEM_PROMPT_DUMP": ROOT_DIR / "data" / "ai_advice" / f"system_prompt_{timestamp}.txt",
    "USER_PROMPT_PREVIEW": ROOT_DIR / "data" / "ai_advice" / f"user_prompt_preview_{timestamp}.txt",
}
CONFIG["PROMPT_DOC_FILE"].parent.mkdir(parents=True, exist_ok=True)


# System Prompt (role + strict rules)
SYSTEM_PROMPT = """
You are a senior computer scientist and mentor.
Transform a small set of computer science papers into a structured, actionable learning plan.

Hard requirements (read carefully):
- Output MUST be a single JSON object ONLY (no commentary, no markdown, no code fences).
- Follow the JSON schema fields EXACTLY:
  goal, study_level, source_papers, metadata, plan_overview, reading_order, actions, metrics, timeline_weeks, risks.
- Use ONLY the provided papers. Do NOT invent or cite any paper that is not in the provided list.
- All values in reading_order[].paper_id MUST be chosen from source_papers,
  and source_papers MUST include ALL provided paper IDs (no missing IDs, no invented IDs).
- Be concise and execution-oriented (clear steps, measurable outcomes).
- If information is insufficient, use fewer items and add a risk item explaining the limitation.
- Do NOT add any extra fields not defined by the schema.

Language and style:
- When referring to the learner, always address them directly as “You”.
- Do NOT describe the learner in the third person (e.g., “a junior computer science student”).

Schema guidance:
- goal: copy the user goal faithfully in meaning.
- study_level: choose from ["beginner","intermediate","advanced"]; if unsure, prefer "intermediate".
- source_papers: MUST contain all paper_id strings from the provided list (do not drop any, do not invent IDs).
- metadata:
  - prompt_version: provided by tooling.
  - model: provided by tooling.
  - created_at: current UTC in ISO8601 (e.g., 2025-11-01T12:34:56Z).
  - tokens_estimated: include only if values are provided; otherwise omit.
- plan_overview: 1–2 short paragraphs explaining rationale and overall strategy.
- reading_order: {paper_id, why_first, key_questions[]} with concrete technical questions.
- actions: 3–10 items of {label, how_to, expected_outcome}, focusing on reproducible tasks.
- metrics: 2–8 measurable indicators.
- timeline_weeks: 2–12 items; week starts at 1.
- risks: 0–8 items; realistic, technical risks only.

Output rule:
- Return ONLY the JSON object that conforms to the provided JSON schema.
""".strip()

# ---- Helpers ----
def clip(text: str, max_chars: int) -> str:
    # Clip long text to avoid context explosion
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."

def norm_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

def load_plan_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_response_format(schema: Dict[str, Any]) -> Dict[str, Any]:
    # Build the Responses API response_format payload
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.get("title", "PaperTrailLearningPlan"),
            "schema": schema,
            "strict": True,
        },
    }

#User Prompt Builder
def build_user_prompt(

    goal: str,

    papers: List[Dict[str, Any]],
    *,
    audience: str = "you",
    study_level_hint: Optional[str] = None,  # "beginner" | "intermediate" | "advanced" | None
    model_id: str = "gpt-5",
    prompt_version: str = PROMPT_VERSION,
    max_abstract_chars: int = 1200,
) -> str:
    # Build the user prompt
    if study_level_hint not in {"beginner", "intermediate", "advanced", None}:
        study_level_hint = None
    if not goal:
        # fallback to first non-empty 'query' from papers (step 1 output)
        for p in papers:
            q = (p.get("query") or "").strip()
            if q:
                goal = q
                break

    lines: List[str] = []
    lines.append(f"Goal: {goal}")
    lines.append(f"Audience: {audience}")
    if study_level_hint:
        lines.append(f"StudyLevelHint: {study_level_hint}")
    lines.append("Selected research papers (id, title, authors, year, citation_count, categories, abstract):")
    allowed_ids = [norm_str(p.get("id")) for p in papers if p.get("id")]
    lines.append("AllowedPaperIDsJSON: " + json.dumps(allowed_ids, ensure_ascii=False))

    # Paper list with clipped abstracts
    for idx, p in enumerate(papers, start=1):
        pid = norm_str(p.get("id"))
        title = norm_str(p.get("title"))
        authors = norm_str(p.get("authors"))
        year = norm_str(p.get("year"))
        cites = norm_str(p.get("citation_count"))
        cats = norm_str(p.get("categories"))
        abstract = clip(norm_str(p.get("abstract")), max_abstract_chars)

        lines.append(f"{idx}) ID: {pid}")
        lines.append(f"   Title: {title}")
        lines.append(f"   Authors: {authors}")
        lines.append(f"   Year: {year}")
        lines.append(f"   Citations: {cites}")
        lines.append(f"   Categories: {cats}")
        lines.append(f"   Abstract: {abstract}")

    # Explicit schema fields to return (aligned with plan_schema.json)
    # We also pass runtime hints so the model can fill metadata correctly.
    lines.append(
        "Return a single JSON object that strictly conforms to the provided JSON schema. "
        "Set source_papers to contain ALL IDs listed in AllowedPaperIDsJSON (do not drop any, do not invent IDs). "
        "Design reading_order so that EVERY paper_id from source_papers appears at least once, "
        "and the sequence forms a coherent learning path for the user."
    )

    return "\n".join(lines)

def build_messages(
    goal: str,
    papers: List[Dict[str, Any]],
    *,
    audience: str = "you",
    study_level_hint: Optional[str] = None,
    model_id: str = "gpt-4.1",
    prompt_version: str = PROMPT_VERSION,
    max_abstract_chars: int = 1200,
) -> List[Dict[str, str]]:
    # Build messages array for OpenAI Responses API
    user_prompt = build_user_prompt(
        goal,
        papers,
        audience=audience,
        study_level_hint=study_level_hint,
        model_id=model_id,
        prompt_version=prompt_version,
        max_abstract_chars=max_abstract_chars,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

def write_prompt_doc(doc_path: Path, schema: Dict[str, Any]) -> None:
    #Write docs/prompt_design.md for version tracking
    doc = []
    doc.append(f"# Prompt Design\n")
    doc.append(f"**Prompt Version:** {PROMPT_VERSION}\n")
    doc.append(f"**Last Updated:** {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)\n")
    doc.append(f"\n## Overview\nDefines the current system prompt template used for structured learning plan generation.\n")
    doc.append(f"\n## System Prompt\n\n```\n{SYSTEM_PROMPT}\n```\n")
    doc.append(f"\n## Schema\n- Title: {schema.get('title','(no title)')}\n")
    doc.append("## Required Fields\n")
    req = schema.get("required", [])
    for r in req:
        doc.append(f"- {r}")
    doc.append("\n## Notes\n- Output must strictly follow the schema.\n- Use only paper IDs listed in source_papers.\n- Keep all content concise, reproducible, and task-oriented.\n")
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(doc), encoding="utf-8")


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    # load schema
    schema_path = CONFIG["PLAN_SCHEMA_FILE"]
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    schema = load_plan_schema(str(schema_path))

    # write prompt doc so we can track changes and explain the design
    write_prompt_doc(CONFIG["PROMPT_DOC_FILE"], schema)

    # save system prompt as plain text for quick review/diff
    dump_text(CONFIG["SYSTEM_PROMPT_DUMP"], SYSTEM_PROMPT)

    # build a user prompt preview
    sp = CONFIG["SELECTED_PAPERS_FILE"]
    if not sp.exists():
        raise FileNotFoundError(f"Selected papers file not found: {sp}")

    papers = []
    with sp.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                papers.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"bad json at line {i}: {e}") from e
    if not papers:
        raise ValueError(f"no papers found in {sp}")

    # build the preview; leave goal empty to fallback to the first non-empty 'query'
    preview = build_user_prompt(
        goal="",                  # empty on purpose; function will fallback to the first 'query'
        papers=papers,
        study_level_hint=None,
        max_abstract_chars=600,
    )
    dump_text(CONFIG["USER_PROMPT_PREVIEW"], preview)

    print(f"wrote docs -> {CONFIG['PROMPT_DOC_FILE']}")
    print(f"dumped system prompt -> {CONFIG['SYSTEM_PROMPT_DUMP']}")
    print(f"user prompt preview -> {CONFIG['USER_PROMPT_PREVIEW']}")
