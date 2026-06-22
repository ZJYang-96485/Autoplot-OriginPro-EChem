import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AI_MODEL = os.getenv("AI_PLOT_MODEL", os.getenv("AI_MODEL", "gpt-5.4"))
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
    "grid_axis": "both",
    "grid_which": "major",
    "major_grid_linestyle": "solid",
    "minor_grid_linestyle": "dashed",
    "major_grid_alpha": 0.25,
    "minor_grid_alpha": 0.12,
    "major_grid_width": 0.7,
    "minor_grid_width": 0.5,
    "legend_location": "best",
    "legend_frame": True,
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
        "grid_axis": {"type": "string", "enum": ["x", "y", "both"]},
        "grid_which": {"type": "string", "enum": ["major", "minor", "both"]},
        "major_grid_linestyle": {"type": "string", "enum": ["solid", "dashed", "dotted", "dashdot"]},
        "minor_grid_linestyle": {"type": "string", "enum": ["solid", "dashed", "dotted", "dashdot"]},
        "major_grid_alpha": {"type": ["number", "null"]},
        "minor_grid_alpha": {"type": ["number", "null"]},
        "major_grid_width": {"type": ["number", "null"]},
        "minor_grid_width": {"type": ["number", "null"]},
        "legend_location": {"type": "string", "enum": ["auto", "best", "upper right", "upper left", "lower right", "lower left"]},
        "legend_frame": {"type": ["boolean", "null"]},
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
        "grid_axis",
        "grid_which",
        "major_grid_linestyle",
        "minor_grid_linestyle",
        "major_grid_alpha",
        "minor_grid_alpha",
        "major_grid_width",
        "minor_grid_width",
        "legend_location",
        "legend_frame",
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

def _looks_like_averaged_dataset(column_names):
    columns = {column.lower(): column for column in column_names}
    required = {"y_mean", "condition"}

    return all(name in columns for name in required)


def _find_column(column_names, candidates):
    lookup = {column.lower(): column for column in column_names}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def _apply_averaged_replicate_defaults(config, column_names, user_request):
    text = str(user_request).lower()
    intent_terms = [
        "average",
        "averaged",
        "mean curve",
        "mean curves",
        "replicate",
        "replicates",
        "condition",
        "conditions",
        "sem",
        "std"
    ]

    if not any(term in text for term in intent_terms):
        return config

    if not _looks_like_averaged_dataset(column_names):
        return config

    y_mean_col = _find_column(column_names, ["y_mean"])
    condition_col = _find_column(column_names, ["condition", "Condition", "group", "Group"])

    if y_mean_col:
        config["y_column"] = y_mean_col

    if condition_col:
        config["group_column"] = condition_col
        config["group_label"] = ""
        config["show_legend"] = True
        config["group_color_mode"] = "auto"

    config["plot_type"] = "line"
    config["data_reduction"] = "raw"
    config["summary_group_column"] = "none"
    config["fit_guide"] = "connect"
    config["show_markers"] = config.get("show_markers", True)
    config["primary_label"] = ""
    config["notes"] = "Using pre-averaged replicate data: y_mean is plotted and condition is used as the group column."

    if not config.get("y_label"):
        config["y_label"] = "Mean response"

    if "y_sem" in [column.lower() for column in column_names]:
        config["notes"] += " The dataset includes y_sem for future error-band plotting."

    return config




def _has_column(column_names, *names):
    lower = {str(column).strip().lower(): column for column in column_names}

    for name in names:
        if str(name).strip().lower() in lower:
            return lower[str(name).strip().lower()]

    return ""


def _column_contains(column_names, *terms):
    for column in column_names:
        lower = str(column).strip().lower()

        if any(str(term).lower() in lower for term in terms):
            return column

    return ""


def _is_spectroscopy_schema(column_names):
    has_wavelength = bool(_has_column(column_names, "wavelength_nm") or _column_contains(column_names, "wavelength"))
    has_absorbance = bool(_has_column(column_names, "absorbance") or _column_contains(column_names, "absorbance", "abs"))

    return has_wavelength and has_absorbance


def _is_electrochem_reference_schema(column_names):
    has_time = bool(_has_column(column_names, "global_time_min", "time_min", "time", "t"))
    has_current_density = bool(_has_column(column_names, "y_mean", "j_mA_cm2", "current_density", "current_density_mA_cm2"))
    has_potential = bool(_has_column(column_names, "E_RHE", "Vf", "potential", "voltage"))

    return has_time and has_current_density and (has_potential or bool(_has_column(column_names, "condition")))


def _apply_spectroscopy_defaults(config, column_names, user_request):
    text = str(user_request).lower()
    spectroscopy_request = any(term in text for term in [
        "uv", "uv-vis", "uvvis", "spectrum", "spectra", "spectroscopy", "wavelength", "absorbance", "abs"
    ])

    if not (_is_spectroscopy_schema(column_names) or spectroscopy_request):
        return config

    x_column = _has_column(column_names, "wavelength_nm") or _column_contains(column_names, "wavelength")
    y_column = _has_column(column_names, "absorbance") or _column_contains(column_names, "absorbance", "abs")
    group_column = _has_column(column_names, "condition", "sample", "group", "label")

    if x_column:
        config["x_column"] = x_column

    if y_column:
        config["y_column"] = y_column

    if group_column:
        config["group_column"] = group_column

    requested_nature = "nature" in text

    config.update({
        "plot_title": "",
        "plot_type": "line",
        "x_label": r"$\mathbf{Wavelength\ /\ nm}$",
        "y_label": r"$\mathbf{Absorbance}$",
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
        "style_profile": "nature" if requested_nature else config.get("style_profile", "publication"),
        "data_reduction": "raw",
        "summary_group_column": "none",
        "use_step_axis": False,
        "step_axis_mode": "auto_data",
        "step_axis_placement": "uniform",
        "step_axis_value_column": "none",
        "step_axis_group_column": "none",
        "step_axis_label": "",
        "step_axis_custom_labels": "",
        "step_axis_custom_positions": "",
        "secondary_mode": "none",
        "x2_column": "none",
        "y2_column": "none",
        "top_x_label": "",
        "right_y_label": "",
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
        "line_order": "sort_x",
        "line_width": 1.6,
        "opacity": 1.0,
        "show_markers": False,
        "show_legend": True if group_column else False,
        "group_label": "",
        "group_color_mode": "auto",
        "legend_location": "best",
        "legend_frame": False,
        "figure_width": 7.5,
        "figure_height": 5.0,
        "figure_dpi": 300,
        "axis_label_size": 18,
        "tick_label_size": 13,
        "axis_label_weight": "bold",
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
        "grid_axis": "both",
        "grid_which": "major",
        "bottom_margin": None,
        "primary_label": "",
        "bottom_annotation": ""
    })

    config["notes"] = (
        "Spectroscopy schema detected. Electrochemical reference-axis settings were disabled. "
        "Using wavelength as X, absorbance as Y, and condition/sample as group when available."
    )

    return config


def _apply_electrochem_reference_defaults(config, column_names, user_request):
    text = str(user_request).lower()

    if _is_spectroscopy_schema(column_names):
        return config

    electrochem_request = any(term in text for term in [
        "electrochem", "electrochemical", "rhe", "reference electrode", "pbs", "nano3", "no3",
        "current density", "potential", "chrono", "gamry", "dta", "orr", "her", "oer"
    ])

    reference_plot_request = any(term in text for term in [
        "reference plot", "reproduce", "bottom potential", "potential labels", "e / v", "vs. rhe"
    ])

    if not (electrochem_request and (_is_electrochem_reference_schema(column_names) or reference_plot_request)):
        return config

    columns_lower = {str(column).lower(): column for column in column_names}

    if "global_time_min" in columns_lower:
        config["x_column"] = columns_lower["global_time_min"]

    if "y_mean" in columns_lower:
        config["y_column"] = columns_lower["y_mean"]
    elif "j_ma_cm2" in columns_lower:
        config["y_column"] = columns_lower["j_ma_cm2"]

    if "condition" in columns_lower:
        config["group_column"] = columns_lower["condition"]

    config.update({
        "plot_title": "",
        "plot_type": "line",
        "data_reduction": "raw",
        "summary_group_column": "none",
        "x_label": "",
        "top_x_label": r"$\mathbf{t\ /\ min}$",
        "y_label": r"$\mathbf{j\ /\ mA\ cm^{-2}}$",
        "step_axis_label": r"$\mathbf{E\ /\ V\ vs.\ RHE}$",
        "x_min": 0,
        "x_max": 160,
        "y_min": -100,
        "y_max": 10,
        "x_tick_mode": "uniform",
        "x_major_interval": 10,
        "x_minor_interval": 5,
        "y_tick_mode": "uniform",
        "y_major_interval": 20,
        "y_minor_interval": 10,
        "use_step_axis": True,
        "step_axis_mode": "uniform_custom",
        "step_axis_placement": "custom_positions",
        "step_axis_value_column": "none",
        "step_axis_group_column": "none",
        "step_axis_custom_labels": "0.5,0.2,-0.1,-0.5,-0.9,-0.7,-0.3,0,0.4",
        "step_axis_custom_positions": "5,20,40,60,80,100,120,140,160",
        "step_axis_decimal_places": 1,
        "step_axis_label_stride": 1,
        "step_axis_label_rotation": 0,
        "step_axis_label_pad": 14,
        "bottom_margin": 0.20,
        "figure_width": 8.0,
        "figure_height": 5.5,
        "figure_dpi": 300,
        "line_order": "sort_x",
        "line_width": 2.0,
        "opacity": 1.0,
        "show_markers": False,
        "show_legend": True,
        "group_label": "",
        "group_color_mode": "auto",
        "legend_location": "lower right",
        "legend_frame": False,
        "axis_label_size": 18,
        "tick_label_size": 13,
        "axis_label_weight": "bold",
        "title_size": 18,
        "legend_font_size": 11,
        "spine_width": 1.3,
        "tick_width": 1.2,
        "tick_length": 5,
        "tick_direction": "in",
        "show_full_frame": True,
        "show_top_ticks": True,
        "show_bottom_ticks": True,
        "show_left_ticks": True,
        "show_right_ticks": True,
        "show_grid": True,
        "grid_axis": "y",
        "grid_which": "both",
        "major_grid_linestyle": "solid",
        "minor_grid_linestyle": "dashed",
        "major_grid_alpha": 0.22,
        "minor_grid_alpha": 0.14,
        "major_grid_width": 0.7,
        "minor_grid_width": 0.5,
        "marker_color": "#9A4DFF",
        "line_color": "#9A4DFF",
        "second_marker_color": "#FF3030",
        "second_line_color": "#FF3030"
    })

    config["notes"] = (
        "Electrochemical reference plot preset applied because the request and dataset schema both support it."
    )

    return config

def _choose_default_y_column(column_names):
    preferred = [
        "y_mean",
        "absorbance",
        "Abs",
        "j_mA_cm2",
        "j_A_cm2",
        "current_density",
        "current_density_mA_cm2",
        "intensity",
        "signal",
        "response",
        "y_value",
        "I_A",
        "Im_A",
        "Current",
        "current",
        "y"
    ]

    lookup = {str(column).strip().lower(): column for column in column_names}

    for name in preferred:
        if name.lower() in lookup:
            return lookup[name.lower()]

    for column in column_names:
        lower = str(column).strip().lower()

        if any(term in lower for term in ["j_", "current", "im_", "i_", "response", "signal"]):
            return column

    return "none"


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
        fallback_y = _choose_default_y_column(column_names)

        if fallback_y != "none":
            config["y_column"] = fallback_y
        else:
            available_columns = ", ".join(column_names)
            raise ValueError(
                "Line and scatter plots require a valid Y column. "
                f"No suitable default Y column was found. Available columns: {available_columns}"
            )

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
            "For line or scatter plots, always choose an existing Y column. "
            "If y_mean exists, prefer y_mean for averaged datasets. If y_mean does not exist but j_mA_cm2 exists, use j_mA_cm2. "
            "Use 'none' for optional column fields that are not needed, but never use 'none' for y_column in line or scatter plots. "
            "Apply the electrochemical PBS/PBS+NaNO3 reference-axis preset only when the user's request explicitly asks for an electrochemical/RHE/PBS/NO3 reference plot and the available columns support electrochemical data. Do not apply electrochemical defaults merely because the user requests publication or Nature style. "
            "Use null for numeric, boolean, and color fields when the user did not specify them and no style profile is clearly requested. "
            "When the user asks for a journal or output style, select the closest style_profile from the supported profiles. "
            "Use style_profile='publication' for generic publication-quality scientific figures. "
            "Use style_profile='nature', 'science', 'acs', 'rsc', 'elsevier', or 'ieee' only when the user explicitly names that target. "
            "Use style_profile='thesis', 'presentation', or 'poster' when the figure is intended for those formats. "
            "Use style_profile='monochrome' for black-and-white printing, 'colorblind' for colorblind-safe plots, and 'dark' for dark-background slides. "
            "For electrochemistry, common axis labels include E / V vs. RHE, j / mA cm^-2, I / A, t / s, and t / min, but still follow the user's request. "
            "For spectroscopy or UV-Vis datasets with wavelength and absorbance columns, use wavelength as X, absorbance as Y, condition/sample as group, disable secondary axes and step axes, clear electrochemical labels, and use automatic axis limits. "
            "Use Matplotlib mathtext for scientific labels when appropriate, such as '$j$ / mA cm$^{-2}$', '$E$ / V vs. RHE', and '$t$ / min. "
            "Never return the literal string 'none', 'null', or 'n/a' for plot_title, primary_label, bottom_annotation, or notes; use an empty string instead. "
            "If the available columns include y_mean, y_std, y_sem, n_replicates, and condition, treat this as a pre-averaged replicate dataset. "
            "For pre-averaged replicate datasets, use y_mean as the Y column, condition as the group column, data_reduction='raw', plot_type='line', show_legend=true, and group_color_mode='auto'. "
            "If the user asks to compare conditions after averaging replicates, do not use summary mode; the dataset is already averaged. "
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
    config = _apply_averaged_replicate_defaults(config, column_names, user_request)
    config = _apply_spectroscopy_defaults(config, column_names, user_request)
    config = _apply_electrochem_reference_defaults(config, column_names, user_request)
    config = _validate_config(config, column_names)

    return config

def get_style_profiles():
    return STYLE_PROFILES

def get_style_presets():
    return STYLE_PRESETS
