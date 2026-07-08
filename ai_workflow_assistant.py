
from pathlib import Path


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


def create_workflow_plan(user_request, file_summaries, current_datasets=None, preset_name=None):
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
