from __future__ import annotations
import os, json, time, uuid, math
from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


# local prompt builder
from src.ai_advice.v2.prompts import(
    PROMPT_VERSION,
    build_messages,
    build_response_format,
    load_plan_schema,
)
timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M")
# config
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
CONFIG = {
    "MODEL": os.environ.get("OPENAI_MODEL", "gpt-5"),
    "TEMPERATURE": float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
    "TIMEOUT_SEC": int(os.environ.get("OPENAI_TIMEOUT", "60")),
    "MAX_RETRIES": int(os.environ.get("OPENAI_MAX_RETRIES", "5")),

    "PAPERS_FILE": latest_standardize_input,
    "SCHEMA_FILE": latest_plan_schema,

    "ARTIFACT_DIR": ROOT_DIR / "data" / "ai_advice" / "artifacts",
    "LOG_DIR": ROOT_DIR / "data" / "ai_advice" / "logs",
}

CONFIG["ARTIFACT_DIR"].mkdir(parents=True, exist_ok=True)
CONFIG["LOG_DIR"].mkdir(parents=True, exist_ok=True)

#tool
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
    # get user topic from first non-empty 'query'
    for p in papers:
        q = (p.get("query") or "").strip()
        if q:
            return q
    raise ValueError("no 'query' found in input papers")

def assert_reading_order_from_source(plan: dict) -> None:
    # enforce reading_order[].paper_id belong to source_papers
    src = set(plan.get("source_papers") or [])
    bad = [it.get("paper_id") for it in plan.get("reading_order", []) if it.get("paper_id") not in src]
    if bad:
        raise ValueError(f"reading_order contains IDs not in source_papers: {bad}")

def make_trace_id() -> str:
    # simple trace id (UTC date + short uuid)
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"

def log_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

#main
def call_with_retries(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    response_format: Dict[str, Any],
    temperature: float,
    timeout_sec: int,
    max_retries: int,
):
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,                 # chat.completions expects 'messages'
                response_format=response_format,   # JSON schema constraint
                timeout=timeout_sec,
            )
            latency = time.time() - t0
            return resp, latency
        except Exception as e:
            if attempt >= max_retries:
                raise
            sleep_s = min(10, 0.5 * (2 ** (attempt - 1)))
            time.sleep(sleep_s)


def extract_text(resp) -> str:
    # chat.completions: first choice, message content
    try:
        return resp.choices[0].message.content or ""
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
        return (in_tokens/1000.0)*in_price + (out_tokens/1000.0)*out_price
    except Exception:
        return None

def main():
    # 0) load inputs
    schema = load_plan_schema(str(CONFIG["SCHEMA_FILE"]))
    papers = read_papers_jsonl(CONFIG["PAPERS_FILE"])
    if not papers:
        raise ValueError("no papers")

    goal = pick_goal_from_query(papers)

    # 1) build prompts + response_format
    messages = build_messages(
        goal=goal,
        papers=papers,
        study_level_hint=None,
        max_abstract_chars=800,  # short to keep context light
    )
    response_format = build_response_format(schema)

    # 2) call OpenAI with retries
    client = OpenAI()  # reads OPENAI_API_KEY from env
    trace_id = make_trace_id()

    resp, latency = call_with_retries(
        client=client,
        model=CONFIG["MODEL"],
        messages=messages,
        response_format=response_format,
        temperature=CONFIG["TEMPERATURE"],
        timeout_sec=CONFIG["TIMEOUT_SEC"],
        max_retries=CONFIG["MAX_RETRIES"],
    )

    # 3) parse model output (must be a single JSON object)
    text = extract_text(resp).strip()
    try:
        plan = json.loads(text)
    except json.JSONDecodeError as e:
        # if it is not json, write debug snippet and fail
        debug_path = CONFIG["ARTIFACT_DIR"] / f"bad_output_{trace_id}.txt"
        debug_path.write_text(text, encoding="utf-8")
        raise ValueError(f"model did not return valid JSON (see {debug_path})") from e

    # 4) validate schema + runtime constraints
    validate_against_schema(schema, plan)
    assert_reading_order_from_source(plan)

    # 5) write artifact
    out_path = CONFIG["ARTIFACT_DIR"] / f"plan_{trace_id}.json"
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    # 6) collect usage + cost (best-effort)
    usage = getattr(resp, "usage", None)
    in_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    out_tokens = getattr(usage, "completion_tokens", None) if usage else None
    cost = estimate_cost_usd(CONFIG["MODEL"], in_tokens, out_tokens)

    # 7) log one line per run
    log_obj = {
        "trace_id": trace_id,
        "ts_utc": datetime.now(UTC).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "model": CONFIG["MODEL"],
        "temperature": CONFIG["TEMPERATURE"],
        "latency_sec": round(latency, 3),
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

    # 8) print a short summary to console
    print("\n== Run Summary ==")
    print(f"trace_id: {trace_id}")
    print(f"model: {CONFIG['MODEL']}  temp: {CONFIG['TEMPERATURE']}")
    print(f"latency: {round(latency, 2)}s  tokens(in/out): {in_tokens}/{out_tokens}  cost~: {cost}")
    print(f"artifact: {out_path}")

if __name__ == "__main__":
    main()
