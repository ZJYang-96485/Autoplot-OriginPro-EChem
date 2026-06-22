
import re
from pathlib import Path


ALLOWED_ACTIONS = {
    "clean_files",
    "stitch_sequence",
    "combine_datasets",
    "convert_variables",
    "average_replicates",
    "plot"
}


def _clean_text(value):
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in {"none", "null", "nan", "na", "n/a"}:
        return ""

    return value


def _file_name_from_summary(item):
    if isinstance(item, str):
        return Path(item).name

    if not isinstance(item, dict):
        return ""

    for key in ["file_name", "original_name", "saved_name", "name"]:
        value = _clean_text(item.get(key))

        if value:
            return value

    return ""


def _infer_condition(file_name):
    lower = file_name.lower()

    if "no3" in lower or "nano3" in lower or "pbsno3" in lower or "pbs+no3" in lower:
        return "PBS+NaNO3"

    if "pbs" in lower:
        return "PBS"

    return "Unknown"


def _sequence_number(file_name):
    stem = Path(file_name).stem

    match = re.search(r"#\s*(\d+)", stem)

    if match:
        return int(match.group(1))

    numbers = re.findall(r"\d+", stem)

    if numbers:
        return int(numbers[-1])

    return 0


def _sequence_prefix(file_name):
    stem = Path(file_name).stem
    stem = re.sub(r"\(\d+\)$", "", stem)
    stem = re.sub(r"#\s*\d+.*$", "", stem)
    stem = stem.strip("_- ")

    return stem or Path(file_name).stem


def _infer_dataset_label(file_name):
    condition = _infer_condition(file_name)
    prefix = _sequence_prefix(file_name)

    if condition == "PBS":
        return "PBS_stitched_sequence"

    if condition == "PBS+NaNO3":
        safe_prefix = re.sub(r"[^A-Za-z0-9]+", "_", prefix).strip("_")
        return safe_prefix or "PBS_NaNO3_stitched_sequence"

    return prefix or "unknown_sequence"


def _sort_mapping_key(item):
    name = item["file_name"]
    condition = item["condition"]
    lower = name.lower()
    is_b = 1 if "chronoa_b" in lower or "_b_" in lower else 0

    if condition == "PBS":
        return (0, is_b, _sequence_number(name), name)

    if condition == "PBS+NaNO3":
        return (1, _sequence_prefix(name), _sequence_number(name), name)

    return (9, name)


def _build_mapping(file_summaries):
    rows = []

    for item in file_summaries or []:
        file_name = _file_name_from_summary(item)

        if not file_name:
            continue

        condition = _infer_condition(file_name)
        dataset_label = _infer_dataset_label(file_name)

        rows.append({
            "file_name": file_name,
            "condition": condition,
            "dataset_label": dataset_label
        })

    return sorted(rows, key=_sort_mapping_key)


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


def create_workflow_plan(user_request, file_summaries, current_datasets=None):
    user_request = _clean_text(user_request)

    if not user_request:
        raise ValueError("Missing workflow request.")

    file_summaries = compact_file_summaries(file_summaries)
    mapping = _build_mapping(file_summaries)

    unknown = [row["file_name"] for row in mapping if row["condition"] == "Unknown"]

    warnings = []

    if unknown:
        warnings.append(
            "Some files could not be mapped to PBS or PBS+NaNO3 by filename rules: "
            + ", ".join(unknown[:10])
        )

    if not mapping and current_datasets:
        warnings.append(
            "No newly uploaded files were provided. The workflow may run on selected/current datasets if available."
        )

    return {
        "workflow_name": "DTA chronoamperometry cleaning, stitching, and replicate averaging",
        "workflow_type": "replicate_average_to_plot",
        "summary": (
            "Deterministic DTA workflow preset. It cleans raw DTA files, converts time/current/potential, "
            "combines files with filename-based PBS/PBS+NaNO3 mapping, stitches sequential 5-minute segments, "
            "and averages replicates by condition. It does not generate a plot."
        ),
        "requires_user_confirmation": True,
        "assumptions": [
            "Each DTA file is one local chronoamperometry segment.",
            "PBS CHRONOA files #1-#16 followed by PBS CHRONOA_B files #1-#16 are one continuous PBS sequence.",
            "NO3 files are stitched by numerical file index within each sequence prefix.",
            "Current is in A unless the detected current column explicitly indicates mA.",
            "Potential conversion uses reference_offset_v = -1 and pH = 0."
        ],
        "warnings": warnings,
        "steps": [
            {
                "step_id": 1,
                "action": "clean_files",
                "title": "Clean raw DTA files",
                "rationale": "Convert raw DTA uploads into cleaned CSV datasets.",
                "input_files": [row["file_name"] for row in mapping],
                "input_dataset_ids": [],
                "parameters": {
                    "dataset_name": "cleaned_dta_files",
                    "file_condition_map": []
                },
                "output_name": "cleaned_dta_files",
                "user_confirmation_needed": []
            },
            {
                "step_id": 2,
                "action": "convert_variables",
                "title": "Convert time, current density, and potential",
                "rationale": "Create global_time_min, j_mA_cm2, and E_RHE columns needed for combining and averaging.",
                "input_files": [],
                "input_dataset_ids": [],
                "parameters": {
                    "dataset_name": "converted_dta_files",
                    "time_col": "",
                    "convert_current": True,
                    "current_col": "",
                    "current_density_output_col": "j_mA_cm2",
                    "electrode_area_cm2": 0.283,
                    "convert_potential": True,
                    "potential_col": "",
                    "potential_output_col": "E_RHE",
                    "reference_offset_v": -1,
                    "ph_value": 0
                },
                "output_name": "converted_dta_files",
                "user_confirmation_needed": []
            },
            {
                "step_id": 3,
                "action": "combine_datasets",
                "title": "Combine and stitch sequential segments",
                "rationale": (
                    "Use deterministic filename mapping. PBS files are one stitched PBS sequence; "
                    "NO3 files are stitched by sequence prefix and number."
                ),
                "input_files": [],
                "input_dataset_ids": [],
                "parameters": {
                    "dataset_name": "combined_stitched_dta",
                    "file_condition_map": mapping,
                    "condition_col": "condition",
                    "replicate_col": "dataset_label",
                    "x_col": "global_time_min",
                    "y_col": "j_mA_cm2"
                },
                "output_name": "combined_stitched_dta",
                "user_confirmation_needed": [
                    "Review the file-to-condition mapping table before execution."
                ]
            },
            {
                "step_id": 4,
                "action": "average_replicates",
                "title": "Average replicates by condition",
                "rationale": "Average stitched condition curves into one final dataset for plotting.",
                "input_files": [],
                "input_dataset_ids": [],
                "parameters": {
                    "dataset_name": "final_averaged_dataset",
                    "x_col": "global_time_min",
                    "y_col": "j_mA_cm2",
                    "condition_col": "condition",
                    "replicate_col": "dataset_label",
                    "averaging_method": "interpolate",
                    "x_grid_method": "overlap",
                    "grid_points": 500,
                    "x_round_decimals": 6,
                    "min_replicates": 1,
                    "electrode_area_cm2": 0.283
                },
                "output_name": "final_averaged_dataset",
                "user_confirmation_needed": []
            }
        ]
    }


def validate_workflow_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("Workflow plan must be a JSON object.")

    steps = plan.get("steps", [])

    if not steps:
        raise ValueError("Workflow plan contains no steps.")

    for step in steps:
        action = _clean_text(step.get("action"))

        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported workflow action: {action}")

        step["action"] = action
        step["title"] = _clean_text(step.get("title")) or action.replace("_", " ").title()
        step["rationale"] = _clean_text(step.get("rationale"))
        step["input_files"] = step.get("input_files") if isinstance(step.get("input_files"), list) else []
        step["input_dataset_ids"] = step.get("input_dataset_ids") if isinstance(step.get("input_dataset_ids"), list) else []
        step["parameters"] = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        step["output_name"] = _clean_text(step.get("output_name")) or step["title"].lower().replace(" ", "_")
        step["user_confirmation_needed"] = (
            step.get("user_confirmation_needed")
            if isinstance(step.get("user_confirmation_needed"), list)
            else []
        )

    plan["workflow_name"] = _clean_text(plan.get("workflow_name")) or "DTA workflow"
    plan["workflow_type"] = _clean_text(plan.get("workflow_type")) or "replicate_average_to_plot"
    plan["summary"] = _clean_text(plan.get("summary"))
    plan["requires_user_confirmation"] = bool(plan.get("requires_user_confirmation", True))
    plan["assumptions"] = plan.get("assumptions") if isinstance(plan.get("assumptions"), list) else []
    plan["warnings"] = plan.get("warnings") if isinstance(plan.get("warnings"), list) else []

    return plan


def describe_workflow_plan(plan):
    plan = validate_workflow_plan(plan)
    lines = [f"Workflow: {plan['workflow_name']}", plan["summary"], "", "Steps:"]

    for step in plan["steps"]:
        lines.append(f"{step['step_id']}. {step['title']} ({step['action']})")

        if step.get("rationale"):
            lines.append(f"   {step['rationale']}")

        if step.get("user_confirmation_needed"):
            lines.append("   Needs confirmation: " + "; ".join(step["user_confirmation_needed"]))

    if plan.get("warnings"):
        lines.append("")
        lines.append("Warnings:")

        for warning in plan["warnings"]:
            lines.append(f"- {warning}")

    return "\n".join(lines)
