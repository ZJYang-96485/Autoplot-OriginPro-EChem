
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
                "file_name": item.get("file_name", ""),
                "file_extension": item.get("file_extension", ""),
                "rows": item.get("rows", 0),
                "columns": item.get("columns", 0),
                "column_names": item.get("column_names", []),
                "dataset_type": item.get("dataset_type", "")
            })

    return compact


def create_workflow_plan(user_request, file_summaries, current_datasets=None, preset_name=None):
    user_request = _clean_text(user_request)
    file_summaries = compact_file_summaries(file_summaries)

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
                "input_dataset_ids": [],
                "parameters": {
                    "user_request": user_request
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
