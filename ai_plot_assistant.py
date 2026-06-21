import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

STYLE_PROFILES = [
    "default",
    "publication",
    "nature",
    "science",
    "acs",
    "rsc",
    "elsevier",
    "ieee",
    "thesis",
    "presentation",
    "poster",
    "monochrome",
    "colorblind",
    "dark"
]

STYLE_PRESETS = {
    "default": {},
    "publication": {
        "figure_width": 6.0,
        "figure_height": 4.0,
        "figure_dpi": 300,
        "axis_label_size": 14,
        "tick_label_size": 11,
        "title_size": 14,
        "legend_font_size": 10,
        "spine_width": 1.1,
        "tick_width": 1.1,
        "tick_length": 4,
        "tick_direction": "in",
        "line_width": 1.8,
        "marker_size": 12,
        "opacity": 1.0,
        "show_grid": False,
        "show_full_frame": True,
        "show_top_ticks": True,
        "show_bottom_ticks": True,
        "show_left_ticks": True,
        "show_right_ticks": True
    },
    "nature": {
        "figure_width": 3.5,
        "figure_height": 2.6,
        "figure_dpi": 300,
        "axis_label_size": 7,
        "tick_label_size": 6,
        "title_size": 7,
        "legend_font_size": 6,
        "spine_width": 0.8,
        "tick_width": 0.8,
        "tick_length": 3,
        "tick_direction": "in",
        "line_width": 1.0,
        "marker_size": 8,
        "opacity": 1.0,
        "show_grid": False,
        "show_full_frame": True,
        "show_top_ticks": True,
        "show_bottom_ticks": True,
        "show_left_ticks": True,
        "show_right_ticks": True
    },
    "science": {
        "figure_width": 3.5,
        "figure_height": 2.6,
        "figure_dpi": 300,
        "axis_label_size": 7,
        "tick_label_size": 6,
        "title_size": 7,
        "legend_font_size": 6,
        "spine_width": 0.8,
        "tick_width": 0.8,
        "tick_length": 3,
        "tick_direction": "in",
        "line_width": 1.0,
        "marker_size": 8,
        "opacity": 1.0,
        "show_grid": False,
        "show_full_frame": True,
        "show_top_ticks": True,
        "show_bottom_ticks": True,
        "show_left_ticks": True,
        "show_right_ticks": True
    },
    "acs": {
        "figure_width": 3.25,
        "figure_height": 2.5,
        "figure_dpi": 300,
        "axis_label_size": 8,
        "tick_label_size": 7,
        "title_size": 8,
        "legend_font_size": 7,
        "spine_width": 0.8,
        "tick_width": 0.8,
        "tick_length": 3,
        "tick_direction": "in",
        "line_width": 1.0,
        "marker_size": 8,
        "opacity": 1.0,
        "show_grid": False,
        "show_full_frame": True,
        "show_top_ticks": True,
        "show_bottom_ticks": True,
        "show_left_ticks": True,
        "show_right_ticks": True
    },
    "rsc": {
        "figure_width": 3.3,
        "figure_height": 2.5,
        "figure_dpi": 300,
        "axis_label_size": 8,
        "tick_label_size": 7,
        "title_size": 8,
        "legend_font_size": 7,
        "spine_width": 0.8,
        "tick_width": 0.8,
        "tick_length": 3,
        "tick_direction": "in",
        "line_width": 1.0,
        "marker_size": 8,
        "opacity": 1.0,
        "show_grid": False,
        "show_full_frame": True,
        "show_top_ticks": True,
        "show_bottom_ticks": True,
        "show_left_ticks": True,
        "show_right_ticks": True
    },
    "elsevier": {
        "figure_width": 3.5,
        "figure_height": 2.6,
        "figure_dpi": 300,
        "axis_label_size": 7,
        "tick_label_size": 7,
        "title_size": 8,
        "legend_font_size": 7,
        "spine_width": 0.8,
        "tick_width": 0.8,
        "tick_length": 3,
        "tick_direction": "in",
        "line_width": 1.0,
        "marker_size": 8,
        "opacity": 1.0,
        "show_grid": False,
        "show_full_frame": True,
        "show_top_ticks": True,
        "show_bottom_ticks": True,
        "show_left_ticks": True,
        "show_right_ticks": True
    },
    "ieee": {
        "figure_width": 3.5,
        "figure_height": 2.4,
        "figure_dpi": 300,
        "axis_label_size": 8,
        "tick_label_size": 7,
        "title_size": 8,
        "legend_font_size": 7,
        "spine_width": 0.8,
        "tick_width": 0.8,
        "tick_length": 3,
        "tick_direction": "in",
        "line_width": 1.0,
        "marker_size": 8,
        "opacity": 1.0,
        "show_grid": False,
        "show_full_frame": True,
        "show_top_ticks": True,
        "show_bottom_ticks": True,
        "show_left_ticks": True,
        "show_right_ticks": True
    },
    "thesis": {
        "figure_width": 6.5,
        "figure_height": 4.5,
        "figure_dpi": 300,
        "axis_label_size": 16,
        "tick_label_size": 12,
        "title_size": 16,
        "legend_font_size": 11,
        "spine_width": 1.2,
        "tick_width": 1.2,
        "tick_length": 5,
        "tick_direction": "in",
        "line_width": 2.0,
        "marker_size": 14,
        "opacity": 1.0,
        "show_grid": False,
        "show_full_frame": True,
        "show_top_ticks": True,
        "show_bottom_ticks": True,
        "show_left_ticks": True,
        "show_right_ticks": True
    },
    "presentation": {
        "figure_width": 9.0,
        "figure_height": 5.5,
        "figure_dpi": 180,
        "axis_label_size": 22,
        "tick_label_size": 16,
        "title_size": 22,
        "legend_font_size": 14,
        "spine_width": 1.4,
        "tick_width": 1.4,
        "tick_length": 6,
        "tick_direction": "in",
        "line_width": 2.6,
        "marker_size": 20,
        "opacity": 1.0,
        "show_grid": True,
        "show_full_frame": True,
        "show_top_ticks": True,
        "show_bottom_ticks": True,
        "show_left_ticks": True,
        "show_right_ticks": True
    },
    "poster": {
        "figure_width": 10.0,
        "figure_height": 6.0,
        "figure_dpi": 200,
        "axis_label_size": 26,
        "tick_label_size": 18,
        "title_size": 26,
        "legend_font_size": 16,
        "spine_width": 1.6,
        "tick_width": 1.6,
        "tick_length": 7,
        "tick_direction": "in",
        "line_width": 3.0,
        "marker_size": 24,
        "opacity": 1.0,
        "show_grid": True,
        "show_full_frame": True,
        "show_top_ticks": True,
        "show_bottom_ticks": True,
        "show_left_ticks": True,
        "show_right_ticks": True
    },
    "monochrome": {
        "marker_color": "#000000",
        "line_color": "#000000",
        "second_marker_color": "#666666",
        "second_line_color": "#666666",
        "show_grid": False,
        "show_full_frame": True
    },
    "colorblind": {
        "marker_color": "#0072B2",
        "line_color": "#0072B2",
        "second_marker_color": "#D55E00",
        "second_line_color": "#D55E00",
        "show_grid": False,
        "show_full_frame": True
    },
    "dark": {
        "marker_color": "#FFB000",
        "line_color": "#FFB000",
        "second_marker_color": "#56B4E9",
        "second_line_color": "#56B4E9",
        "show_grid": True,
        "show_full_frame": True
    }
}

DEFAULT_VALUES = {
    "plot_title": "",
    "x_column": "",
    "y_column": "none",
    "plot_type": "line",
    "x_label": "",
    "y_label": "",
    "x_min": None,
    "x_max": None,
    "y_min": None,
    "y_max": None,
    "style_profile": "default",
    "marker_color": "#FF5F05",
    "line_color": "#13294B",
    "primary_label": "",
    "group_column": "none",
    "group_label": "",
    "group_color_mode": "same",
    "data_reduction": "raw",
    "summary_group_column": "none",
    "x_summary_method": "mean",
    "y_summary_method": "mean_tail",
    "tail_fraction": 0.20,
    "fit_guide": "connect",
    "smooth_window": 5,
    "use_step_axis": False,
    "step_axis_mode": "auto_data",
    "step_axis_placement": "uniform",
    "step_axis_value_column": "none",
    "step_axis_group_column": "none",
    "step_axis_label": "",
    "step_axis_max_ticks": 12,
    "step_axis_decimal_places": 1,
    "step_axis_label_stride": 2,
    "step_axis_custom_labels": "",
    "step_axis_custom_positions": "",
    "step_axis_label_rotation": 0,
    "step_axis_label_pad": 14,
    "bottom_margin": None,
    "figure_width": 8.0,
    "figure_height": 5.0,
    "figure_dpi": 150,
    "x_tick_mode": "auto",
    "x_major_interval": None,
    "x_minor_interval": None,
    "x_custom_ticks": "",
    "x_custom_tick_labels": "",
    "y_tick_mode": "auto",
    "y_major_interval": None,
    "y_minor_interval": None,
    "y_custom_ticks": "",
    "y_custom_tick_labels": "",
    "line_order": "original",
    "opacity": 1.0,
    "line_width": 2.2,
    "marker_size": 18,
    "show_markers": True,
    "show_legend": False,
    "axis_label_size": 18,
    "tick_label_size": 13,
    "axis_label_weight": "bold",
    "title_size": 18,
    "legend_font_size": 11,
    "spine_width": 1.2,
    "tick_width": 1.2,
    "tick_length": 5,
    "tick_direction": "in",
    "show_full_frame": True,
    "show_top_ticks": True,
    "show_bottom_ticks": True,
    "show_left_ticks": True,
    "show_right_ticks": True,
    "show_grid": False,
    "secondary_mode": "none",
    "x2_column": "none",
    "y2_column": "none",
    "top_x_label": "",
    "right_y_label": "",
    "secondary_plot_type": "line",
    "second_label": "",
    "second_marker_color": "#9A4DFF",
    "second_line_color": "#9A4DFF",
    "bottom_annotation": "",
    "export_format": "png",
    "notes": ""
}

PLOT_CONFIG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "plot_title": {"type": "string"},
        "x_column": {"type": "string"},
        "y_column": {"type": "string"},
        "plot_type": {"type": "string", "enum": ["line", "scatter", "line_scatter", "bar", "histogram", "box", "violin", "heatmap"]},
        "x_label": {"type": "string"},
        "y_label": {"type": "string"},
        "x_min": {"type": ["number", "null"]},
        "x_max": {"type": ["number", "null"]},
        "y_min": {"type": ["number", "null"]},
        "y_max": {"type": ["number", "null"]},
        "style_profile": {"type": "string", "enum": STYLE_PROFILES},
        "marker_color": {"type": ["string", "null"]},
        "line_color": {"type": ["string", "null"]},
        "primary_label": {"type": "string"},
        "group_column": {"type": "string"},
        "group_label": {"type": "string"},
        "group_color_mode": {"type": "string", "enum": ["same", "auto"]},
        "data_reduction": {"type": "string", "enum": ["raw", "summary"]},
        "summary_group_column": {"type": "string"},
        "x_summary_method": {"type": "string", "enum": ["mean", "median", "first", "last", "min", "max"]},
        "y_summary_method": {"type": "string", "enum": ["mean_tail", "mean", "median", "first", "last", "min", "max"]},
        "tail_fraction": {"type": ["number", "null"]},
        "fit_guide": {"type": "string", "enum": ["none", "connect", "smooth"]},
        "smooth_window": {"type": ["integer", "null"]},
        "use_step_axis": {"type": ["boolean", "null"]},
        "step_axis_mode": {"type": "string", "enum": ["auto_data", "uniform_custom"]},
        "step_axis_placement": {"type": "string", "enum": ["uniform", "data_positions", "custom_positions"]},
        "step_axis_value_column": {"type": "string"},
        "step_axis_group_column": {"type": "string"},
        "step_axis_label": {"type": "string"},
        "step_axis_max_ticks": {"type": ["integer", "null"]},
        "step_axis_decimal_places": {"type": ["integer", "null"]},
        "step_axis_label_stride": {"type": ["integer", "null"]},
        "step_axis_custom_labels": {"type": "string"},
        "step_axis_custom_positions": {"type": "string"},
        "step_axis_label_rotation": {"type": ["number", "null"]},
        "step_axis_label_pad": {"type": ["number", "null"]},
        "bottom_margin": {"type": ["number", "null"]},
        "figure_width": {"type": ["number", "null"]},
        "figure_height": {"type": ["number", "null"]},
        "figure_dpi": {"type": ["integer", "null"]},
        "x_tick_mode": {"type": "string", "enum": ["auto", "uniform", "custom"]},
        "x_major_interval": {"type": ["number", "null"]},
        "x_minor_interval": {"type": ["number", "null"]},
        "x_custom_ticks": {"type": "string"},
        "x_custom_tick_labels": {"type": "string"},
        "y_tick_mode": {"type": "string", "enum": ["auto", "uniform", "custom"]},
        "y_major_interval": {"type": ["number", "null"]},
        "y_minor_interval": {"type": ["number", "null"]},
        "y_custom_ticks": {"type": "string"},
        "y_custom_tick_labels": {"type": "string"},
        "line_order": {"type": "string", "enum": ["original", "sort_x"]},
        "opacity": {"type": ["number", "null"]},
        "line_width": {"type": ["number", "null"]},
        "marker_size": {"type": ["number", "null"]},
        "show_markers": {"type": ["boolean", "null"]},
        "show_legend": {"type": ["boolean", "null"]},
        "axis_label_size": {"type": ["integer", "null"]},
        "tick_label_size": {"type": ["integer", "null"]},
        "axis_label_weight": {"type": "string", "enum": ["bold", "normal"]},
        "title_size": {"type": ["integer", "null"]},
        "legend_font_size": {"type": ["integer", "null"]},
        "spine_width": {"type": ["number", "null"]},
        "tick_width": {"type": ["number", "null"]},
        "tick_length": {"type": ["number", "null"]},
        "tick_direction": {"type": "string", "enum": ["in", "out", "inout"]},
        "show_full_frame": {"type": ["boolean", "null"]},
        "show_top_ticks": {"type": ["boolean", "null"]},
        "show_bottom_ticks": {"type": ["boolean", "null"]},
        "show_left_ticks": {"type": ["boolean", "null"]},
        "show_right_ticks": {"type": ["boolean", "null"]},
        "show_grid": {"type": ["boolean", "null"]},
        "secondary_mode": {"type": "string", "enum": ["none", "same_y_different_x", "same_x_different_y"]},
        "x2_column": {"type": "string"},
        "y2_column": {"type": "string"},
        "top_x_label": {"type": "string"},
        "right_y_label": {"type": "string"},
        "secondary_plot_type": {"type": "string", "enum": ["line", "scatter"]},
        "second_label": {"type": "string"},
        "second_marker_color": {"type": ["string", "null"]},
        "second_line_color": {"type": ["string", "null"]},
        "bottom_annotation": {"type": "string"},
        "export_format": {"type": "string", "enum": ["png", "pdf", "both"]},
        "notes": {"type": "string"}
    },
    "required": [
        "plot_title",
        "x_column",
        "y_column",
        "plot_type",
        "x_label",
        "y_label",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "style_profile",
        "marker_color",
        "line_color",
        "primary_label",
        "group_column",
        "group_label",
        "group_color_mode",
        "data_reduction",
        "summary_group_column",
        "x_summary_method",
        "y_summary_method",
        "tail_fraction",
        "fit_guide",
        "smooth_window",
        "use_step_axis",
        "step_axis_mode",
        "step_axis_placement",
        "step_axis_value_column",
        "step_axis_group_column",
        "step_axis_label",
        "step_axis_max_ticks",
        "step_axis_decimal_places",
        "step_axis_label_stride",
        "step_axis_custom_labels",
        "step_axis_custom_positions",
        "step_axis_label_rotation",
        "step_axis_label_pad",
        "bottom_margin",
        "figure_width",
        "figure_height",
        "figure_dpi",
        "x_tick_mode",
        "x_major_interval",
        "x_minor_interval",
        "x_custom_ticks",
        "x_custom_tick_labels",
        "y_tick_mode",
        "y_major_interval",
        "y_minor_interval",
        "y_custom_ticks",
        "y_custom_tick_labels",
        "line_order",
        "opacity",
        "line_width",
        "marker_size",
        "show_markers",
        "show_legend",
        "axis_label_size",
        "tick_label_size",
        "axis_label_weight",
        "title_size",
        "legend_font_size",
        "spine_width",
        "tick_width",
        "tick_length",
        "tick_direction",
        "show_full_frame",
        "show_top_ticks",
        "show_bottom_ticks",
        "show_left_ticks",
        "show_right_ticks",
        "show_grid",
        "secondary_mode",
        "x2_column",
        "y2_column",
        "top_x_label",
        "right_y_label",
        "secondary_plot_type",
        "second_label",
        "second_marker_color",
        "second_line_color",
        "bottom_annotation",
        "export_format",
        "notes"
    ]
}

def _clean_columns(column_names):
    return [str(column) for column in column_names if str(column).strip()]

def _match_column(value, column_names, allow_none=True):
    if value is None:
        return "none" if allow_none else ""

    value = str(value).strip()

    if allow_none and value.lower() in {"", "none", "null", "no", "n/a"}:
        return "none"

    for column in column_names:
        if value == column:
            return column

    for column in column_names:
        if value.lower() == column.lower():
            return column

    if allow_none:
        return "none"

    raise ValueError(f"Invalid column: {value}")

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

def _merge_profile_defaults(config):
    merged = dict(DEFAULT_VALUES)

    profile_name = config.get("style_profile") or "default"
    profile_values = STYLE_PRESETS.get(profile_name, {})

    merged.update(profile_values)

    for key, value in config.items():
        if value is not None:
            merged[key] = value

    if merged["style_profile"] == "monochrome":
        merged["group_color_mode"] = "same"

    if merged["plot_type"] == "line_scatter":
        merged["plot_type"] = "line"
        merged["show_markers"] = True

    if merged["plot_type"] == "scatter":
        merged["show_markers"] = True

    if merged["data_reduction"] == "summary" and merged["summary_group_column"] == "none":
        merged["summary_group_column"] = merged["group_column"]

    if merged["use_step_axis"]:
        if merged["step_axis_custom_labels"].strip():
            merged["step_axis_mode"] = "uniform_custom"
        if merged["step_axis_custom_positions"].strip():
            merged["step_axis_placement"] = "custom_positions"

    if merged["x_major_interval"] is not None:
        merged["x_tick_mode"] = "uniform"

    if merged["x_custom_ticks"].strip():
        merged["x_tick_mode"] = "custom"

    if merged["y_major_interval"] is not None:
        merged["y_tick_mode"] = "uniform"

    if merged["y_custom_ticks"].strip():
        merged["y_tick_mode"] = "custom"

    return merged

def _validate_config(config, column_names):
    config["x_column"] = _match_column(config.get("x_column"), column_names, allow_none=False)
    config["y_column"] = _match_column(config.get("y_column"), column_names, allow_none=True)
    config["group_column"] = _match_column(config.get("group_column"), column_names, allow_none=True)
    config["summary_group_column"] = _match_column(config.get("summary_group_column"), column_names, allow_none=True)
    config["step_axis_value_column"] = _match_column(config.get("step_axis_value_column"), column_names, allow_none=True)
    config["step_axis_group_column"] = _match_column(config.get("step_axis_group_column"), column_names, allow_none=True)
    config["x2_column"] = _match_column(config.get("x2_column"), column_names, allow_none=True)
    config["y2_column"] = _match_column(config.get("y2_column"), column_names, allow_none=True)

    if config["secondary_mode"] == "same_y_different_x" and config["x2_column"] == "none":
        config["secondary_mode"] = "none"

    if config["secondary_mode"] == "same_x_different_y" and config["y2_column"] == "none":
        config["secondary_mode"] = "none"

    if config["plot_type"] in {"histogram", "bar"} and config["y_column"] == "none":
        config["y_label"] = config["y_label"] or "Count"

    if config["plot_type"] in {"line", "line_scatter", "scatter"} and config["y_column"] == "none":
        raise ValueError("Line and scatter plots require a valid Y column.")

    if config["x_min"] is not None and config["x_max"] is not None and config["x_min"] >= config["x_max"]:
        raise ValueError("x_min must be smaller than x_max.")

    if config["y_min"] is not None and config["y_max"] is not None and config["y_min"] >= config["y_max"]:
        raise ValueError("y_min must be smaller than y_max.")

    return config

def parse_plot_request(user_request, column_names):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    column_names = _clean_columns(column_names)

    if not column_names:
        raise ValueError("No column names were provided.")

    columns_text = ", ".join(column_names)
    profiles_text = ", ".join(STYLE_PROFILES)

    response = client.responses.create(
        model=AI_MODEL,
        instructions=(
            "You are an AI assistant for a scientific plotting platform. "
            "Convert the user's natural language request into a structured plotting configuration. "
            "Only choose column fields from the available column names. "
            "Use 'none' for optional column fields that are not needed. "
            "Use null for numeric, boolean, and color fields when the user did not specify them and no style profile is clearly requested. "
            "When the user asks for a journal or output style, select the closest style_profile from the supported profiles. "
            "Use style_profile='publication' for generic publication-quality scientific figures. "
            "Use style_profile='nature', 'science', 'acs', 'rsc', 'elsevier', or 'ieee' only when the user explicitly names that target. "
            "Use style_profile='thesis', 'presentation', or 'poster' when the figure is intended for those formats. "
            "Use style_profile='monochrome' for black-and-white printing, 'colorblind' for colorblind-safe plots, and 'dark' for dark-background slides. "
            "For electrochemistry, common axis labels include E / V vs. RHE, j / mA cm^-2, I / A, t / s, and t / min, but still follow the user's request. "
            "Do not invent columns. Do not generate code. Do not execute plotting."
        ),
        input=(
            f"Supported style profiles: {profiles_text}\n\n"
            f"Available columns: {columns_text}\n\n"
            f"User request: {user_request}"
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "plot_config",
                "schema": PLOT_CONFIG_SCHEMA,
                "strict": True
            }
        }
    )

    output_text = _extract_output_text(response)
    config = json.loads(output_text)
    config = _merge_profile_defaults(config)
    config = _validate_config(config, column_names)

    return config

def get_style_profiles():
    return STYLE_PROFILES

def get_style_presets():
    return STYLE_PRESETS
