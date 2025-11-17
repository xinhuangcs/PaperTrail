from __future__ import annotations
import os, json, time, uuid, math, sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# Add project root to Python path for imports
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

# local prompt builder
from src.ai_advice.v2.prompts import (
    PROMPT_VERSION,
    build_messages,
    build_response_format,
    load_plan_schema,
)

timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")

# config
data_dir = ROOT_DIR / "data" / "ai_advice"
PLAN_LATEST_PATH = data_dir / "plan_latest.json"


def get_latest_standardize_input() -> Path:
    files = list(data_dir.glob("standardize_input_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No standardize_input_*.json files found in {data_dir}. "
            "Please run the standardize step first."
        )
    return max(files, key=lambda f: f.stat().st_mtime)


def get_latest_plan_schema() -> Path:
    files = list(data_dir.glob("plan_schema_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No plan_schema_*.json files found in {data_dir}. "
            "Please run schema_contract.py first."
        )
    return max(files, key=lambda f: f.stat().st_mtime)


CONFIG = {
    "MODEL": os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
    "TEMPERATURE": float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
    "TIMEOUT_SEC": int(os.environ.get("OPENAI_TIMEOUT", "60")),
    "MAX_RETRIES": int(os.environ.get("OPENAI_MAX_RETRIES", "5")),

    "PAPERS_FILE": None,
    "SCHEMA_FILE": None,

    "ARTIFACT_DIR": ROOT_DIR / "data" / "ai_advice" / "artifacts",
    "LOG_DIR": ROOT_DIR / "data" / "ai_advice" / "logs",
}

CONFIG["ARTIFACT_DIR"].mkdir(parents=True, exist_ok=True)
CONFIG["LOG_DIR"].mkdir(parents=True, exist_ok=True)

# Recommend file path for extracting topics
RECOMMEND_DIR = ROOT_DIR / "data" / "recommend"

def read_papers_jsonl(path: Path) -> List[Dict[str, Any]]:
    papers = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                papers.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"bad json at line {i}: {e}") from e
    return papers


def pick_goal_from_query(papers: List[Dict[str, Any]]) -> str:
    for p in papers:
        q = (p.get("query") or "").strip()
        if q:
            return q
    raise ValueError("no 'query' found in input papers")


def assert_reading_order_from_source(plan: dict) -> None:
    src = set(plan.get("source_papers") or [])
    bad = [
        it.get("paper_id")
        for it in plan.get("reading_order", [])
        if it.get("paper_id") not in src
    ]
    if bad:
        raise ValueError(f"reading_order contains IDs not in source_papers: {bad}")


def collect_topics_from_recommend(prefer_view: str = "default") -> Optional[str]:
    if prefer_view:
        preferred_file = RECOMMEND_DIR / f"recommend_{prefer_view}.json"
        if preferred_file.exists():
            recommend_file = preferred_file
        else:
            recommend_files = sorted(
                RECOMMEND_DIR.glob("recommend_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not recommend_files:
                return None
            recommend_file = recommend_files[0]
    else:
        recommend_files = sorted(
            RECOMMEND_DIR.glob("recommend_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not recommend_files:
            return None
        recommend_file = recommend_files[0]

    # Collect all unique topics
    topics_set = set()
    try:
        with recommend_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    topics = rec.get("topics")
                    if topics and isinstance(topics, list):
                        topics_set.update(t for t in topics if t)
                except (json.JSONDecodeError, Exception):
                    continue
    except Exception:
        return None

    if not topics_set:
        return None

    # Format topics as comma-separated string
    topics_list = sorted(list(topics_set))
    topics_str = ", ".join(topics_list)

    return (
        "The selected papers may also cover research areas including: "
        f"{topics_str}. You may gain a broader understand of the field "
        "by exploring these related research directions"
    )


def make_trace_id() -> str:
    # simple trace id (UTC date + short uuid)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def log_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# OpenAI call

def _normalize_text_format(fmt: Dict[str, Any]) -> Dict[str, Any]:

    if not isinstance(fmt, dict):
        raise ValueError("text_format must be a dict")

    # Already flat
    if fmt.get("type") == "json_schema" and "schema" in fmt:
        if "name" not in fmt:
            fmt["name"] = "plan_schema"
        fmt.setdefault("strict", True)
        return fmt

    # Legacy nested shape
    if fmt.get("type") == "json_schema" and isinstance(fmt.get("json_schema"), dict):
        js = fmt["json_schema"]
        name = js.get("name") or "plan_schema"
        strict = js.get("strict", True)
        schema = js.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("text_format.json_schema.schema must be an object")
        return {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": strict,
        }

    raise ValueError("Unsupported text_format shape for Responses API")


def call_with_retries(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    text_format: Dict[str, Any],
    temperature: float,
    timeout_sec: int,
    max_retries: int,
):
    # Convert chat-style messages into Responses API input format
    inputs = [
        {
            "role": m["role"],
            "content": [
                {"type": "input_text", "text": m["content"]},
            ],
        }
        for m in messages
    ]

    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            resp = client.responses.create(
                model=model,
                input=inputs,
                temperature=temperature,
                # Structured outputs for Responses API
                text={
                    "format": text_format,
                },
                timeout=timeout_sec,
            )
            latency = time.time() - start
            return resp, latency

        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise
            # simple backoff
            time.sleep(2 * attempt)

    raise last_err  # should not reach here


def extract_text(resp) -> str:
    """Extract text from Responses API response."""
    try:
        txt = getattr(resp, "output_text", None)
        if txt:
            return txt

        if hasattr(resp, "output") and resp.output:
            first = resp.output[0]
            content = getattr(first, "content", None)
            if content:
                part = content[0]
                part_text = getattr(part, "text", None)

                if isinstance(part_text, str):
                    return part_text

                if hasattr(part_text, "value"):
                    return part_text.value

                if isinstance(part_text, dict):
                    if "value" in part_text:
                        return str(part_text["value"])
                    if "text" in part_text:
                        return str(part_text["text"])

                if isinstance(part, dict) and "text" in part:
                    return str(part["text"])

        return ""
    except Exception:
        return ""


def validate_against_schema(schema: dict, data: dict) -> None:
    try:
        Draft202012Validator(schema).validate(data)
    except ValidationError as e:
        loc = " → ".join([str(p) for p in e.path]) or "<root>"
        raise ValueError(f"schema fail at [{loc}]: {e.message}") from e


def estimate_cost_usd(model: str, in_tokens: Optional[int], out_tokens: Optional[int]) -> Optional[float]:
    try:
        in_price = float(os.environ.get("PRICE_IN_USD_PER_1K", "0"))
        out_price = float(os.environ.get("PRICE_OUT_USD_PER_1K", "0"))
        if in_tokens is None or out_tokens is None:
            return None
        return (in_tokens / 1000.0) * in_price + (out_tokens / 1000.0) * out_price
    except Exception:
        return None


def main():
    # 0) load inputs (lazy load files)
    CONFIG["SCHEMA_FILE"] = get_latest_plan_schema()
    CONFIG["PAPERS_FILE"] = get_latest_standardize_input()

    schema = load_plan_schema(str(CONFIG["SCHEMA_FILE"]))
    papers = read_papers_jsonl(CONFIG["PAPERS_FILE"])
    if not papers:
        raise ValueError("no papers")

    goal = pick_goal_from_query(papers)

    # 1) build prompts + structured output format
    messages = build_messages(
        goal=goal,
        papers=papers,
        study_level_hint=None,
        max_abstract_chars=800,  # short to keep context light
    )
    text_format = build_response_format(schema)
    text_format = _normalize_text_format(text_format)

    # 2) call OpenAI with retries
    client = OpenAI()  # reads OPENAI_API_KEY from env
    trace_id = make_trace_id()

    resp, latency = call_with_retries(
        client=client,
        model=CONFIG["MODEL"],
        messages=messages,
        text_format=text_format,
        temperature=CONFIG["TEMPERATURE"],
        timeout_sec=CONFIG["TIMEOUT_SEC"],
        max_retries=CONFIG["MAX_RETRIES"],
    )

    # 3) parse model output (must be a single JSON object)
    text = extract_text(resp).strip()
    try:
        plan = json.loads(text)
    except json.JSONDecodeError as e:
        debug_path = CONFIG["ARTIFACT_DIR"] / f"bad_output_{trace_id}.txt"
        debug_path.write_text(text, encoding="utf-8")
        raise ValueError(f"model did not return valid JSON (see {debug_path})") from e

    all_ids: List[str] = [p.get("id") for p in papers if p.get("id")]
    plan["source_papers"] = all_ids

    meta = plan.get("metadata") or {}
    meta["prompt_version"] = PROMPT_VERSION
    meta["model"] = CONFIG["MODEL"]
    meta["created_at"] = datetime.now(timezone.utc).isoformat()
    plan["metadata"] = meta

    # 4) validate schema + runtime constraints
    validate_against_schema(schema, plan)
    assert_reading_order_from_source(plan)

    # 5) Add topics suggestion (hardcoded, extracted from recommend JSON)
    topics_suggestion = collect_topics_from_recommend(prefer_view="default")
    if topics_suggestion:
        plan["_topics_suggestion"] = topics_suggestion

    # 6) write artifact
    out_path = CONFIG["ARTIFACT_DIR"] / f"plan_{trace_id}.json"
    out_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # also write a fixed "latest" file for the pipeline
    PLAN_LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_LATEST_PATH.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 7) collect usage + cost (best-effort)
    usage = getattr(resp, "usage", None)
    in_tokens = None
    out_tokens = None
    if usage is not None:
        # Responses API usually uses input_tokens / output_tokens
        in_tokens = getattr(usage, "input_tokens", None) or getattr(
            usage, "prompt_tokens", None
        )
        out_tokens = getattr(usage, "output_tokens", None) or getattr(
            usage, "completion_tokens", None
        )

    cost = estimate_cost_usd(CONFIG["MODEL"], in_tokens, out_tokens)

    # 8) log one line per run
    log_obj = {
        "trace_id": trace_id,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "model": CONFIG["MODEL"],
        "temperature": CONFIG["TEMPERATURE"],
        "latency_sec": round(latency, 3) if latency is not None else None,
        "tokens_in": in_tokens,
        "tokens_out": out_tokens,
        "cost_usd_estimate": cost,
        "artifact": str(out_path),
        "schema_file": str(CONFIG["SCHEMA_FILE"]),
        "papers_file": str(CONFIG["PAPERS_FILE"]),
        "status": "ok",
    }
    log_path = CONFIG["LOG_DIR"] / f"inference_{timestamp}.json"
    log_jsonl(log_path, log_obj)

    # 9) print a short summary to console
    print("\n== Run Summary ==")
    print(f"trace_id: {trace_id}")
    print(f"model: {CONFIG['MODEL']}  temp: {CONFIG['TEMPERATURE']}")
    print(
        f"latency: {round(latency, 2) if latency is not None else 'n/a'}s  "
        f"tokens(in/out): {in_tokens}/{out_tokens}  cost~: {cost}"
    )
    print(f"artifact: {out_path}")
    print(f"latest:   {PLAN_LATEST_PATH}")


if __name__ == "__main__":
    main()
