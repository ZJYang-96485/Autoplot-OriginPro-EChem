
from pathlib import Path
import re


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


def _extension(name):
    return Path(str(name)).suffix.lower().lstrip(".")


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


def _infer_condition(file_name):
    lower = str(file_name).lower()

    if "no3" in lower or "nano3" in lower or "pbsno3" in lower or "pbs+no3" in lower:
        return "PBS+NaNO3"

    if "pbs" in lower:
        return "PBS"

    stem = Path(str(file_name)).stem
    stem = re.sub(r"#\s*\d+.*$", "", stem)
    stem = re.sub(r"\(\d+\)$", "", stem)
    stem = stem.strip("_- ")

    return stem or Path(str(file_name)).stem


def _sequence_number(file_name):
    stem = Path(str(file_name)).stem
    match = re.search(r"#\s*(\d+)", stem)

    if match:
        return int(match.group(1))

    numbers = re.findall(r"\d+", stem)

    return int(numbers[-1]) if numbers else 0


def _is_b_sequence(file_name):
    lower = str(file_name).lower()
    return "chronoa_b" in lower or "_b_" in lower


def _sequence_prefix(file_name):
    stem = Path(str(file_name)).stem
    stem = re.sub(r"\(\d+\)$", "", stem)
    stem = re.sub(r"#\s*\d+.*$", "", stem)
    stem = re.sub(r"CHRONOA[_-]*B$", "CHRONOA", stem, flags=re.IGNORECASE)
    stem = stem.strip("_- ")
    return stem or Path(str(file_name)).stem


def _dataset_label(file_name):
    condition = _infer_condition(file_name)

    if condition == "PBS":
        return "PBS_stitched_sequence"

    prefix = re.sub(r"[^A-Za-z0-9]+", "_", _sequence_prefix(file_name)).strip("_")

    return prefix or condition


def _infer_preset(file_summaries):
    names = [_file_name(item) for item in file_summaries or []]
    names = [name for name in names if name]
    extensions = {_extension(name) for name in names if _extension(name)}

    if not names:
        return "auto_detect"

    if extensions and extensions.issubset({"dta"}):
        if any("#" in name for name in names):
            return "dta_segmented_ca"
        return "dta_curve_auto"

    if extensions and extensions.issubset({"csv", "txt", "dat", "tsv"}):
        return "tabular_auto"

    if extensions and extensions.issubset({"xlsx", "xls"}):
        return "spreadsheet_auto"

    if extensions & {"dta", "csv", "txt", "dat", "tsv", "xlsx", "xls"}:
        return "mixed_supported_auto"

    return "inspect_only"


def _mapping_rows(file_summaries):
    rows = []

    for item in file_summaries or []:
        file_name = _file_name(item)

        if not file_name:
            continue

        rows.append({
            "file_name": file_name,
            "condition": _infer_condition(file_name),
            "dataset_label": _dataset_label(file_name)
        })

    return sorted(
        rows,
        key=lambda row: (
            row["condition"],
            row["dataset_label"],
            int(_is_b_sequence(row["file_name"])),
            _sequence_number(row["file_name"]),
            row["file_name"]
        )
    )


def create_workflow_plan(user_request, file_summaries, current_datasets=None, preset_name=None):
    user_request = _clean_text(user_request)
    file_summaries = compact_file_summaries(file_summaries)
    current_datasets = compact_file_summaries(current_datasets)

    preset = _clean_text(preset_name) or _infer_preset(file_summaries)
    mapping = _mapping_rows(file_summaries)

    return {
        "workflow_name": "Automatic data inspection and processing workflow",
        "workflow_type": "auto_detect",
        "preset_name": preset,
        "summary": (
            "The workflow inspects uploaded files, chooses the safest supported deterministic route, "
            "creates a combined dataset, and creates an averaged dataset when valid X/Y/condition/replicate information is available."
        ),
        "requires_user_confirmation": True,
        "assumptions": [
            "DTA files are parsed directly from CURVE tables when possible.",
            "Segmented files are stitched by sequence numbers in filenames.",
            "Tabular files use automatic X/Y/condition/replicate column inference.",
            "If safe processing is not possible, execution returns diagnostics instead of producing misleading data."
        ],
        "warnings": [],
        "steps": [
            {
                "step_id": 1,
                "action": "auto_detect_process",
                "title": "Inspect files and process with the safest compatible route",
                "rationale": "Use deterministic data inspection and processing rather than a free-form AI-generated workflow.",
                "input_files": [row["file_name"] for row in mapping],
                "input_dataset_ids": [],
                "parameters": {
                    "preset_name": preset,
                    "file_condition_map": mapping,
                    "user_request": user_request
                },
                "output_name": "auto_processed_dataset",
                "user_confirmation_needed": [
                    "Review detected preset, condition mapping, selected columns, and diagnostics after execution."
                ]
            }
        ]
    }


def validate_workflow_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("Workflow plan must be a JSON object.")

    if "steps" not in plan or not isinstance(plan["steps"], list) or not plan["steps"]:
        raise ValueError("Workflow plan contains no steps.")

    plan["preset_name"] = _clean_text(plan.get("preset_name")) or "auto_detect"

    return plan


def describe_workflow_plan(plan):
    plan = validate_workflow_plan(plan)
    lines = [
        f"Workflow: {plan.get('workflow_name', 'Automatic workflow')}",
        f"Preset: {plan.get('preset_name', 'auto_detect')}",
        plan.get("summary", ""),
        "",
        "Steps:"
    ]

    for step in plan.get("steps", []):
        lines.append(f"{step.get('step_id', '?')}. {step.get('title', step.get('action', 'step'))} ({step.get('action', '')})")

        if step.get("rationale"):
            lines.append(f"   {step['rationale']}")

        needed = step.get("user_confirmation_needed", [])

        if needed:
            lines.append("   Needs confirmation: " + "; ".join(needed))

    return "\n".join(lines)
