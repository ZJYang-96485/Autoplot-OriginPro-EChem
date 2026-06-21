import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

AI_WORKFLOW_MODEL = os.getenv("AI_WORKFLOW_MODEL", os.getenv("AI_MODEL", "gpt-5.5"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ALLOWED_ACTIONS = {
    "clean_files",
    "stitch_sequence",
    "combine_datasets",
    "convert_variables",
    "average_replicates",
    "plot"
}

WORKFLOW_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "workflow_name": {"type": "string"},
        "workflow_type": {
            "type": "string",
            "enum": [
                "plot_only",
                "single_file_to_plot",
                "multi_file_to_plot",
                "replicate_average_to_plot",
                "unknown"
            ]
        },
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
                    "action": {
                        "type": "string",
                        "enum": [
                            "clean_files",
                            "stitch_sequence",
                            "combine_datasets",
                            "convert_variables",
                            "average_replicates",
                            "plot"
                        ]
                    },
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "input_files": {"type": "array", "items": {"type": "string"}},
                    "input_dataset_ids": {"type": "array", "items": {"type": "string"}},
                    "output_name": {"type": "string"},
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "dataset_name": {"type": "string"},
                            "condition_label": {"type": "string"},
                            "file_condition_map": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "file_name": {"type": "string"},
                                        "condition": {"type": "string"},
                                        "dataset_label": {"type": "string"}
                                    },
                                    "required": ["file_name", "condition", "dataset_label"]
                                }
                            },
                            "x_col": {"type": "string"},
                            "y_col": {"type": "string"},
                            "group_col": {"type": "string"},
                            "condition_col": {"type": "string"},
                            "replicate_col": {"type": "string"},
                            "time_col": {"type": "string"},
                            "step_variable_col": {"type": "string"},
                            "sequence_duration_s": {"type": "number"},
                            "sequence_regex": {"type": "string"},
                            "convert_potential": {"type": "boolean"},
                            "potential_col": {"type": "string"},
                            "potential_output_col": {"type": "string"},
                            "reference_offset_v": {"type": "number"},
                            "ph_value": {"type": "number"},
                            "convert_current": {"type": "boolean"},
                            "current_col": {"type": "string"},
                            "current_density_output_col": {"type": "string"},
                            "electrode_area_cm2": {"type": "number"},
                            "averaging_method": {"type": "string"},
                            "x_grid_method": {"type": "string"},
                            "grid_points": {"type": "integer"},
                            "x_round_decimals": {"type": "integer"},
                            "min_replicates": {"type": "integer"},
                            "plot_style": {"type": "string"},
                            "plot_type": {"type": "string"},
                            "x_label": {"type": "string"},
                            "y_label": {"type": "string"},
                            "plot_title": {"type": "string"},
                            "show_legend": {"type": "boolean"},
                            "legend_title": {"type": "string"},
                            "legend_location": {"type": "string"},
                            "legend_frame": {"type": "boolean"},
                            "use_step_axis": {"type": "boolean"},
                            "step_axis_label": {"type": "string"},
                            "step_axis_custom_labels": {"type": "string"},
                            "step_axis_custom_positions": {"type": "string"}
                        },
                        "required": []
                    },
                    "review_required": {"type": "boolean"},
                    "user_confirmation_needed": {"type": "array", "items": {"type": "string"}}
                },
                "required": [
                    "step_id",
                    "action",
                    "title",
                    "rationale",
                    "input_files",
                    "input_dataset_ids",
                    "output_name",
                    "parameters",
                    "review_required",
                    "user_confirmation_needed"
                ]
            }
        },
        "final_plot_prompt": {"type": "string"}
    },
    "required": [
        "workflow_name",
        "workflow_type",
        "summary",
        "requires_user_confirmation",
        "assumptions",
        "warnings",
        "steps",
        "final_plot_prompt"
    ]
}


def _clean_text(value):
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"none", "null", "n/a", "na"}:
        return ""

    return text


def _extract_output_text(response):
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    texts = []

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)

            if text:
                texts.append(text)

    return "\n".join(texts).strip()


def compact_file_summaries(file_summaries):
    compact = []

    for item in file_summaries or []:
        if isinstance(item, str):
            item = {
                "file_name": item,
                "description": item
            }

        if not isinstance(item, dict):
            continue

        compact.append({
            "file_name": _clean_text(item.get("file_name") or item.get("name")),
            "file_extension": _clean_text(item.get("file_extension") or item.get("extension")),
            "rows": item.get("rows", 0),
            "columns": item.get("columns", 0),
            "column_names": item.get("column_names", []),
            "numeric_columns": item.get("numeric_columns", []),
            "categorical_columns": item.get("categorical_columns", []),
            "dataset_type": _clean_text(item.get("dataset_type")),
            "description": _clean_text(item.get("description")),
            "preview": str(item.get("preview", ""))[:1200]
        })

    return compact


def normalize_file_condition_map(value):
    if value is None:
        return []

    if isinstance(value, dict):
        value = [value]

    if isinstance(value, str):
        return [
            {
                "file_name": value,
                "condition": "",
                "dataset_label": ""
            }
        ]

    if not isinstance(value, list):
        return []

    normalized = []

    for item in value:
        if isinstance(item, str):
            normalized.append({
                "file_name": item,
                "condition": "",
                "dataset_label": ""
            })
        elif isinstance(item, dict):
            normalized.append({
                "file_name": _clean_text(item.get("file_name")),
                "condition": _clean_text(item.get("condition")),
                "dataset_label": _clean_text(item.get("dataset_label"))
            })

    return normalized


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
        step["title"] = _clean_text(step.get("title"))
        step["rationale"] = _clean_text(step.get("rationale"))
        step["output_name"] = _clean_text(step.get("output_name"))
        step["input_files"] = step.get("input_files") or []
        step["input_dataset_ids"] = step.get("input_dataset_ids") or []
        step["parameters"] = step.get("parameters") or {}

        if not isinstance(step["parameters"], dict):
            step["parameters"] = {}

        step["parameters"]["file_condition_map"] = normalize_file_condition_map(
            step["parameters"].get("file_condition_map")
        )
        step["review_required"] = bool(step.get("review_required", True))
        step["user_confirmation_needed"] = [
            _clean_text(item)
            for item in step.get("user_confirmation_needed", [])
            if _clean_text(item)
        ]

    plan["workflow_name"] = _clean_text(plan.get("workflow_name")) or "AI Workflow"
    plan["workflow_type"] = _clean_text(plan.get("workflow_type")) or "unknown"
    plan["summary"] = _clean_text(plan.get("summary"))
    plan["requires_user_confirmation"] = bool(plan.get("requires_user_confirmation", True))
    plan["assumptions"] = [_clean_text(item) for item in plan.get("assumptions", []) if _clean_text(item)]
    plan["warnings"] = [_clean_text(item) for item in plan.get("warnings", []) if _clean_text(item)]
    plan["final_plot_prompt"] = _clean_text(plan.get("final_plot_prompt"))

    return plan


def create_workflow_plan(user_request, file_summaries, current_datasets=None):
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")

    user_request = _clean_text(user_request)

    if not user_request:
        raise ValueError("Missing workflow request.")

    file_summaries = compact_file_summaries(file_summaries)
    current_datasets = compact_file_summaries(current_datasets)

    response = client.responses.create(
        model=AI_WORKFLOW_MODEL,
        instructions=(
            "You are an AI workflow planner for a Flask app that processes electrochemical and scientific data. "
            "Create a structured workflow plan using only the allowed actions in the schema. "
            "Do not generate Python code, shell commands, or arbitrary transformations. "
            "The backend will execute only approved existing functions. "
            "Use clean_files for raw uploaded files that need cleaning. "
            "Use stitch_sequence when multiple files belong to one time sequence. "
            "Use convert_variables for potential and current-density conversion. "
            "Use combine_datasets when datasets need condition and replicate labels. "
            "Use average_replicates only after condition and replicate labels exist. "
            "Use plot only after the required plotting columns exist. "
            "Do not skip variable conversion when the requested downstream columns are missing. "
            "If uploaded raw files do not contain global_time_min or j_mA_cm2, add a convert_variables step before combine_datasets and average_replicates. "
            "If the files contain a time column such as T_s, Time_s, time_s, T, Time, or time, set time_col so the backend can create global_time_min. "
            "If the files contain current and potential columns, set current_col and potential_col using the detected column names. "
            "If electrode area, reference offset, or pH are not provided, still include the convert_variables step, but set requires_user_confirmation=true and list these missing values in user_confirmation_needed. "
            "For DTA/Gamry-style data, likely candidate columns include T_s or Time_s for time, Im_A or Im for current, and Vf_V_vs_Ref, Vf, or Ewe/V for potential. "
            "Do not assume electrode area, reference offset, pH, condition mapping, or replicate mapping unless the user provides them. "
            "If these are unclear, set requires_user_confirmation to true and list what must be confirmed. "
            "For averaged replicate datasets, plot y_mean grouped by condition and do not use summary mode. "
            "For publication plots, prefer legend_frame=false and legend_location='auto' when the user asks to avoid data. "
            "Use Matplotlib mathtext labels when appropriate, such as '$j$ / mA cm$^{-2}$', '$E$ / V vs. RHE', and '$t$ / min. "
            "Never return the literal string 'none', 'null', or 'n/a' for visible text fields; use an empty string."
        ),
        input=(
            "User request:\n"
            f"{user_request}\n\n"
            "Uploaded or selected file summaries:\n"
            f"{json.dumps(file_summaries, indent=2)}\n\n"
            "Planning rule:\n"
            "If the target plotting/averaging columns are absent from uploaded raw files, include convert_variables before combine/average. "
            "Do not write 'no conversion is planned' unless global_time_min and j_mA_cm2 already exist in the input selected for averaging.\n\n"
            "Current registered dataset summaries if available:\n"
            f"{json.dumps(current_datasets, indent=2)}"
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "workflow_plan",
                "schema": WORKFLOW_PLAN_SCHEMA,
                "strict": False
            }
        }
    )

    output_text = _extract_output_text(response)

    if not output_text:
        raise ValueError("The AI workflow planner returned an empty response.")

    plan = json.loads(output_text)

    return validate_workflow_plan(plan)


def describe_workflow_plan(plan):
    plan = validate_workflow_plan(plan)
    lines = [f"Workflow: {plan['workflow_name']}", plan["summary"], "", "Steps:"]

    for step in plan["steps"]:
        lines.append(f"{step['step_id']}. {step['title']} ({step['action']})")
        lines.append(f"   {step['rationale']}")

        needed = step.get("user_confirmation_needed", [])

        if needed:
            lines.append("   Needs confirmation: " + "; ".join(needed))

    if plan["warnings"]:
        lines.append("")
        lines.append("Warnings:")

        for warning in plan["warnings"]:
            lines.append(f"- {warning}")

    return "\n".join(lines)
