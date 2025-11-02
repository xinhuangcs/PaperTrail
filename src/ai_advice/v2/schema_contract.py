from pathlib import Path
import json
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from datetime import datetime, UTC


#Config
ROOT_DIR = Path(__file__).resolve().parents[3]
latest_file = max(
    (ROOT_DIR / "data" / "ai_advice").glob("standardize_input_*.json"),
    key=lambda f: f.stat().st_mtime
)
timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M")
CONFIG = {
    "INPUT_SELECTED_FILE": latest_file,

    # the JSON Schema file to be written for downstream use (LLM response_format)
    "SCHEMA_FILE": ROOT_DIR / "data" / "ai_advice" / f"plan_schema_{timestamp}.json",

    #for quick validation in future steps
    "EXAMPLE_OUTPUT_FILE": ROOT_DIR / "data" / "ai_advice" / f"plan_example_{timestamp}.json",
}
Path(CONFIG["SCHEMA_FILE"]).parent.mkdir(parents=True, exist_ok=True)
Path(CONFIG["EXAMPLE_OUTPUT_FILE"]).parent.mkdir(parents=True, exist_ok=True)

# 2) Plan JSON Schema
PLAN_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "description": "Structured learning plan generated from selected papers for a given goal.",
    "title": "PaperTrailLearningPlan",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "plan_overview",
        "reading_order",
        "actions",
        "metrics",
        "timeline_weeks",
        "risks",
        "goal",
        "study_level",
        "source_papers",
        "metadata"
    ],
    "properties": {
        "goal": {"type": "string", "minLength": 1, "maxLength": 2000},
        "study_level": {
            "type": "string",
            "enum": ["beginner", "intermediate", "advanced"]
        },
        "source_papers": {
            "type": "array",
            "minItems": 1,
            "maxItems": 200,
            "items": {"type": "string", "minLength": 1}
        },
        "metadata": {
            "type": "object",
            "additionalProperties": False,
            "required": ["prompt_version", "model", "created_at"],
            "properties": {
                "prompt_version": {"type": "string", "minLength": 1, "maxLength": 100},
                "model": {"type": "string", "minLength": 1, "maxLength": 100},
                "created_at": {"type": "string", "format": "date-time"}
             }
        },


        "plan_overview": {
            "type": "string",
            "minLength": 1,
            "maxLength": 5000
        },
        "reading_order": {
            "type": "array",
            "minItems": 1,
            "maxItems": 50,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["paper_id", "why_first", "key_questions"],
                "properties": {
                    "paper_id": {"type": "string", "minLength": 1},
                    "why_first": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "key_questions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 6,
                        "items": {"type": "string", "minLength": 1, "maxLength": 500}
                    }
                }
            }
        },
        "actions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 50,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "how_to", "expected_outcome"],
                "properties": {
                    "label": {"type": "string", "minLength": 1, "maxLength": 200},
                    "how_to": {"type": "string", "minLength": 1, "maxLength": 3000},
                    "expected_outcome": {"type": "string", "minLength": 1, "maxLength": 2000}
                }
            }
        },
        "metrics": {
            "type": "array",
            "minItems": 1,
            "maxItems": 30,
            "items": {"type": "string", "minLength": 1, "maxLength": 200}
        },
        "timeline_weeks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 52,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["week", "focus", "deliverable"],
                "properties": {
                    "week": {"type": "integer", "minimum": 1, "maximum": 104},
                    "focus": {"type": "string", "minLength": 1, "maxLength": 500},
                    "deliverable": {"type": "string", "minLength": 1, "maxLength": 1000}
                }
            }
        },
        "risks": {
            "type": "array",
            "minItems": 0,
            "maxItems": 30,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["risk", "mitigation"],
                "properties": {
                    "risk": {"type": "string", "minLength": 1, "maxLength": 500},
                    "mitigation": {"type": "string", "minLength": 1, "maxLength": 1000}
                }
            }
        }
    }
}


#tools
def write_schema(path: Path = CONFIG["SCHEMA_FILE"]) -> None:
    # write the JSON Schema
    Draft202012Validator.check_schema(PLAN_SCHEMA)
    with path.open("w", encoding="utf-8") as f:
        json.dump(PLAN_SCHEMA, f, ensure_ascii=False, indent=2)

def validate_plan(plan: dict) -> None:
    try:
        Draft202012Validator(PLAN_SCHEMA).validate(plan)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e.message}") from e


def quick_example() -> dict:
    # A valid example for schema validation
    return {
        "goal": "Read and understand what 'model complexity' really means in deep learning and how it connects to generalization or model design.",
        "study_level": "intermediate",
        "source_papers": [
            "2103.05127"
        ],
        "plan_overview": (
            "My plan is to go through the paper 'Model Complexity of Deep Learning: A Survey' "
            "and try to make sense of how researchers define and measure model complexity. "
            "I want to take notes on the main ideas like expressive capacity and effective model "
            "complexity, and see how they relate to things I already know about overfitting or "
            "optimization. After that, I’ll try to make a small example or diagram to help me remember the concepts."
        ),
        "reading_order": [
            {
                "paper_id": "2103.05127",
                "why_first": (
                    "It's a survey paper that summarizes different views on model complexity, "
                    "so it should be a good way to get an overview before diving into more detailed studies."
                ),
                "key_questions": [
                    "What are the main factors that define model complexity?",
                    "How does model complexity affect generalization or performance?",
                    "What open questions or future directions does the paper mention?"
                ]
            }
        ],
        "actions": [
            {
                "label": "Read the paper and highlight key terms",
                "how_to": (
                    "Read the abstract, introduction, and the sections about expressive capacity "
                    "and effective model complexity. Highlight terms I don’t understand and write "
                    "short summaries in my own words."
                ),
                "expected_outcome": (
                    "A few pages of notes with highlighted definitions and simple explanations."
                )
            },
            {
                "label": "Summarize examples and frameworks",
                "how_to": (
                    "Look at how the paper categorizes complexity by model size, optimization, and data. "
                    "Maybe draw a small chart showing the relationships."
                ),
                "expected_outcome": (
                    "One diagram or table summarizing the main framework used in the survey."
                )
            },
            {
                "label": "Make a small illustrative test",
                "how_to": (
                    "Try a quick example, maybe comparing a small vs. large neural net on a simple dataset, "
                    "to see how size and training affect performance."
                ),
                "expected_outcome": (
                    "A short notebook or plot showing how complexity can change results."
                )
            }
        ],
        "metrics": [
            "Finish reading the survey and write at least 500 words of notes",
            "Make one visual summary (chart or diagram)",
            "Do one short experiment or comparison"
        ],
        "timeline_weeks": [
            {
                "week": 1,
                "focus": "Reading and note-taking",
                "deliverable": "Notes and highlighted PDF"
            },
            {
                "week": 2,
                "focus": "Summarizing and small experiment",
                "deliverable": "Diagram and short notebook"
            }
        ],
        "risks": [
            {
                "risk": "The paper is long and dense with theory",
                "mitigation": "Focus on main sections first and skip heavy math if needed"
            },
            {
                "risk": "Hard to find data for testing complexity ideas",
                "mitigation": "Use simple datasets like MNIST or toy data from sklearn"
            }
        ],
        "metadata": {
            "prompt_version": "student-draft-v2",
            "model": "manual-draft-by-student",
            "created_at": "2025-10-31T20:00:00Z",
        }
    }



if __name__ == "__main__":
    write_schema()

    example = quick_example()
    validate_plan(example)

    # save example output for later reference
    with open(CONFIG["EXAMPLE_OUTPUT_FILE"], "w", encoding="utf-8") as f:
        json.dump(example, f, ensure_ascii=False, indent=2)

    print(f"Schema written -> {CONFIG['SCHEMA_FILE']}")
    print(f"Example plan saved -> {CONFIG['EXAMPLE_OUTPUT_FILE']}")
