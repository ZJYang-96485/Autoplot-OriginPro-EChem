
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

AI_WORKFLOW_MODEL = os.getenv("AI_WORKFLOW_MODEL", os.getenv("AI_MODEL", "gpt-5.5-pro"))

WORKFLOW_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "workflow_name": {"type": "string"},
        "workflow_type": {"type": "string"},
        "preset_name": {"type": "string"},
        "summary": {"type": "string"},
        "requires_user_confirmation": {"type": "boolean"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "step_id": {"type": "integer"},
                    "action": {"type": "string", "enum": ["inspect_classify_execute"]},
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "input_files": {"type": "array", "items": {"type": "string"}},
                    "input_dataset_ids": {"type": "array", "items": {"type": "string"}},
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "user_request": {"type": "string"},
                            "combine_uploaded_files": {"type": "boolean"},
                            "combine_selected_datasets": {"type": "boolean"},
                            "clean_before_combine": {"type": "boolean"},
                            "generate_plots": {"type": "boolean"},
                            "max_plots": {"type": "integer"}
                        },
                        "required": [
                            "user_request",
                            "combine_uploaded_files",
                            "combine_selected_datasets",
                            "clean_before_combine",
                            "generate_plots",
                            "max_plots"
                        ]
                    },
                    "output_name": {"type": "string"},
                    "user_confirmation_needed": {"type": "array", "items": {"type": "string"}}
                },
                "required": [
                    "step_id",
                    "action",
                    "title",
                    "rationale",
                    "input_files",
                    "input_dataset_ids",
                    "parameters",
                    "output_name",
                    "user_confirmation_needed"
                ]
            }
        }
    },
    "required": [
        "workflow_name",
        "workflow_type",
        "preset_name",
        "summary",
        "requires_user_confirmation",
        "assumptions",
        "warnings",
        "steps"
    ]
}


def _clean_text(value):
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in {"none", "null", "nan", "na", "n/a"}:
        return ""

    return value


def _file_name(item):
    if isinstance(item, str):
        return Path(item).name

    if not isinstance(item, dict):
        return ""

    for key in ["file_name", "original_name", "saved_name", "name"]:
        value = _clean_text(item.get(key))

        if value:
            return value

    return ""


def compact_file_summaries(items):
    compact = []

    for item in items or []:
        if isinstance(item, str):
            compact.append({"file_name": Path(item).name})
        elif isinstance(item, dict):
            compact.append({
                "dataset_id": item.get("dataset_id", ""),
                "file_name": item.get("file_name", ""),
                "file_extension": item.get("file_extension", ""),
                "rows": item.get("rows", 0),
                "columns": item.get("columns", 0),
                "column_names": item.get("column_names", []),
                "dataset_type": item.get("dataset_type", "")
            })

    return compact


def _dataset_ids(items):
    ids = []

    for item in items or []:
        if not isinstance(item, dict):
            continue

        value = _clean_text(item.get("dataset_id"))

        if value:
            ids.append(value)

    return ids


def _requested_plot_count(user_request):
    text = _clean_text(user_request).lower()

    if any(term in text for term in ["no plot", "without plot", "do not plot", "don't plot"]):
        return 0

    wants_plot = any(term in text for term in ["plot", "figure", "chart", "graph", "visual"])

    if not wants_plot:
        return 0

    if any(term in text for term in ["all plots", "several", "multiple", "every", "each", "separate plots"]):
        return 4

    return 1


def _extract_output_text(response):
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    chunks = []

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)

    if chunks:
        return "".join(chunks)

    raise RuntimeError("No text output returned by the model.")


def _create_default_workflow_plan(user_request, file_summaries, current_datasets=None, preset_name=None):
    user_request = _clean_text(user_request)
    file_summaries = compact_file_summaries(file_summaries)
    current_datasets = compact_file_summaries(current_datasets)
    input_dataset_ids = _dataset_ids(current_datasets)
    max_plots = _requested_plot_count(user_request)

    return {
        "workflow_name": "Generic parser-registry workflow",
        "workflow_type": "inspect_classify_execute",
        "preset_name": _clean_text(preset_name) or "auto_parser_registry",
        "summary": (
            "Inspect uploaded files, classify their data shape, route them to a supported deterministic parser, "
            "and create processed datasets with diagnostics. This is intended to avoid adding a new hardcoded "
            "router for every new experiment."
        ),
        "requires_user_confirmation": True,
        "assumptions": [
            "CSV/TXT/XLSX files may be long tables, wide paired tables, or other tabular exports.",
            "DTA files with CURVE tables use the Gamry DTA parser.",
            "The executor should stop with diagnostics if mixed or unsupported data shapes are detected."
        ],
        "warnings": [],
        "steps": [
            {
                "step_id": 1,
                "action": "inspect_classify_execute",
                "title": "Inspect files, classify data shape, and execute the matching parser",
                "rationale": "Use data-shape detection rather than hardcoding every dataset type.",
                "input_files": [_file_name(item) for item in file_summaries],
                "input_dataset_ids": input_dataset_ids,
                "parameters": {
                    "user_request": user_request,
                    "combine_uploaded_files": True,
                    "combine_selected_datasets": bool(input_dataset_ids),
                    "clean_before_combine": True,
                    "generate_plots": max_plots > 0,
                    "max_plots": max_plots
                },
                "output_name": "processed_dataset",
                "user_confirmation_needed": [
                    "Review file inspection, parser classification, selected columns, and recommended plot mapping."
                ]
            }
        ]
    }


def _repair_workflow_plan(plan, default_plan, user_request, file_summaries, current_datasets):
    plan = validate_workflow_plan(plan)
    file_summaries = compact_file_summaries(file_summaries)
    current_datasets = compact_file_summaries(current_datasets)
    allowed_dataset_ids = set(_dataset_ids(current_datasets))
    default_step = default_plan["steps"][0]
    step = plan["steps"][0]

    step["action"] = "inspect_classify_execute"
    step["input_files"] = [_file_name(item) for item in file_summaries]
    step["input_dataset_ids"] = [
        value
        for value in step.get("input_dataset_ids", [])
        if value in allowed_dataset_ids
    ]

    if allowed_dataset_ids and not step["input_dataset_ids"]:
        step["input_dataset_ids"] = default_step["input_dataset_ids"]

    parameters = step.get("parameters")

    if not isinstance(parameters, dict):
        parameters = {}

    default_parameters = default_step["parameters"]
    parameters["user_request"] = _clean_text(user_request)
    parameters["combine_uploaded_files"] = bool(file_summaries)
    parameters["combine_selected_datasets"] = bool(step["input_dataset_ids"])
    parameters["clean_before_combine"] = True
    parameters["generate_plots"] = bool(parameters.get("generate_plots", default_parameters["generate_plots"]))

    try:
        max_plots = int(parameters.get("max_plots", default_parameters["max_plots"]))
    except (TypeError, ValueError):
        max_plots = default_parameters["max_plots"]

    parameters["max_plots"] = max(0, min(max_plots, 8))

    if not parameters["generate_plots"]:
        parameters["max_plots"] = 0

    step["parameters"] = parameters

    if not plan.get("workflow_name"):
        plan["workflow_name"] = default_plan["workflow_name"]

    if not plan.get("workflow_type"):
        plan["workflow_type"] = default_plan["workflow_type"]

    if not plan.get("preset_name"):
        plan["preset_name"] = default_plan["preset_name"]

    plan["requires_user_confirmation"] = True

    return plan


def create_workflow_plan(user_request, file_summaries, current_datasets=None, preset_name=None):
    default_plan = _create_default_workflow_plan(
        user_request=user_request,
        file_summaries=file_summaries,
        current_datasets=current_datasets,
        preset_name=preset_name
    )

    if not os.getenv("OPENAI_API_KEY"):
        return default_plan

    compact_files = compact_file_summaries(file_summaries)
    compact_datasets = compact_file_summaries(current_datasets)

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.responses.create(
            model=AI_WORKFLOW_MODEL,
            instructions=(
                "Create a safe workflow plan for a data-cleaning and plotting app. "
                "The executor supports one action: inspect_classify_execute. "
                "Use uploaded files and selected datasets only from the provided lists. "
                "Plan for cleaning before combining, combine compatible files or selected datasets, "
                "and generate plots only when the request asks for plots, figures, charts, graphs, or visuals. "
                "Keep the plan concise and outcome-focused for a researcher. "
                "Do not invent files, dataset IDs, columns, conversions, or unsupported actions."
            ),
            input=(
                f"User request: {_clean_text(user_request)}\n\n"
                f"Uploaded file summaries JSON:\n{json.dumps(compact_files, indent=2)}\n\n"
                f"Selected dataset summaries JSON:\n{json.dumps(compact_datasets, indent=2)}\n\n"
                f"Default safe plan JSON:\n{json.dumps(default_plan, indent=2)}"
            ),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "workflow_plan",
                    "schema": WORKFLOW_PLAN_SCHEMA,
                    "strict": True
                }
            }
        )
        plan = json.loads(_extract_output_text(response))

        return _repair_workflow_plan(plan, default_plan, user_request, file_summaries, current_datasets)
    except Exception as error:
        fallback = dict(default_plan)
        fallback["warnings"] = list(default_plan.get("warnings", [])) + [
            f"AI workflow planner unavailable; deterministic safe planner used instead: {error}"
        ]
        return fallback


def validate_workflow_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("Workflow plan must be a JSON object.")

    if "steps" not in plan or not isinstance(plan["steps"], list) or not plan["steps"]:
        raise ValueError("Workflow plan contains no steps.")

    plan["preset_name"] = _clean_text(plan.get("preset_name")) or "auto_parser_registry"

    return plan


def describe_workflow_plan(plan):
    plan = validate_workflow_plan(plan)
    lines = [
        f"Workflow: {plan.get('workflow_name', 'Generic parser-registry workflow')}",
        f"Workflow type: {plan.get('workflow_type', 'inspect_classify_execute')}",
        f"Preset: {plan.get('preset_name', 'auto_parser_registry')}",
        plan.get("summary", ""),
        "",
        "Steps:"
    ]

    for step in plan.get("steps", []):
        lines.append(f"{step.get('step_id', '?')}. {step.get('title', step.get('action', 'step'))} ({step.get('action', '')})")

        if step.get("rationale"):
            lines.append(f"   {step['rationale']}")

    return "\n".join(lines)
