from pathlib import Path
from datetime import datetime
import json
import uuid
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator, FixedFormatter

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename

from data_loader import read_dataset, inspect_dataset, save_cleaned_dataset
from data_combiner import combine_file_paths
from sequential_tools import combine_sequential_file_paths
from conversion_tools import convert_dataset_variables
from replicate_tools import average_condition_replicates
from ai_plot_assistant import parse_plot_request
from ai_workflow_assistant import create_workflow_plan, describe_workflow_plan
from workflow_executor import execute_workflow_plan

app = Flask(__name__)
app.secret_key = "dev"

BASE_DIR = Path(__file__).resolve().parent
DATA_STORAGE_DIR = BASE_DIR / "data_storage"
TEST_DATA_DIR = DATA_STORAGE_DIR / "test_data"
UPLOADED_DATA_DIR = DATA_STORAGE_DIR / "uploaded_data"
PROCESSED_DATA_DIR = DATA_STORAGE_DIR / "processed_data"
METADATA_DIR = DATA_STORAGE_DIR / "metadata"
MANIFEST_PATH = METADATA_DIR / "dataset_manifest.json"
PLOT_DIR = BASE_DIR / "static" / "generated_plots"

ALLOWED_EXTENSIONS = {"csv", "dat", "dta", "txt"}
SECONDARY_MODES = {"none", "same_y_different_x", "same_x_different_y"}
STEP_AXIS_MODES = {"auto_data", "uniform_custom"}
STEP_AXIS_PLACEMENTS = {"uniform", "data_positions", "custom_positions"}
TICK_MODES = {"auto", "uniform", "custom"}


def create_dirs():
    for path in [
        DATA_STORAGE_DIR,
        TEST_DATA_DIR,
        UPLOADED_DATA_DIR,
        PROCESSED_DATA_DIR,
        METADATA_DIR,
        PLOT_DIR
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_manifest():
    create_dirs()

    if not MANIFEST_PATH.exists():
        return {"datasets": []}

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def write_manifest(manifest):
    create_dirs()

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)


def is_supported_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def sanitize_color(value, default):
    if isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
        return value

    return default


def clamp_float(value, default, minimum, maximum):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    return max(minimum, min(value, maximum))


def clamp_int(value, default, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default

    return max(minimum, min(value, maximum))


def optional_float(value):
    try:
        if value in [None, ""]:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_output_name(value, default_name):
    value = str(value).strip()

    if not value:
        return default_name

    value = secure_filename(value)

    if not value:
        return default_name

    return value


def parse_csv_text(value):
    if value in [None, ""]:
        return []

    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_csv_floats(value):
    items = parse_csv_text(value)
    numbers = []

    for item in items:
        try:
            numbers.append(float(item))
        except ValueError:
            continue

    return numbers


def format_step_value(value, decimal_places):
    rounded = round(float(value), decimal_places)

    if abs(rounded) < 10 ** (-(decimal_places + 1)):
        rounded = 0.0

    text = f"{rounded:.{decimal_places}f}"

    if "." in text:
        text = text.rstrip("0").rstrip(".")

    if text == "-0":
        text = "0"

    return text


def evenly_select_records(records, target_count):
    if not records or target_count <= 0:
        return []

    if target_count >= len(records):
        return records

    if target_count == 1:
        return [records[len(records) // 2]]

    indices = [
        round(index * (len(records) - 1) / (target_count - 1))
        for index in range(target_count)
    ]

    return [records[index] for index in indices]


def get_axis_x_range(axis_df, x_col, x_min, x_max, ax):
    data_x = pd.to_numeric(axis_df[x_col], errors="coerce").dropna()

    if x_min is not None:
        x_start = x_min
    elif not data_x.empty:
        x_start = float(data_x.min())
    else:
        x_start = ax.get_xlim()[0]

    if x_max is not None:
        x_end = x_max
    elif not data_x.empty:
        x_end = float(data_x.max())
    else:
        x_end = ax.get_xlim()[1]

    return x_start, x_end


def apply_axis_limits(axis, axis_name, minimum=None, maximum=None):
    if axis_name == "x":
        current_min, current_max = axis.get_xlim()
        axis.set_xlim(
            minimum if minimum is not None else current_min,
            maximum if maximum is not None else current_max
        )
    else:
        current_min, current_max = axis.get_ylim()
        axis.set_ylim(
            minimum if minimum is not None else current_min,
            maximum if maximum is not None else current_max
        )


def apply_axis_tick_control(
    axis,
    axis_name,
    tick_mode,
    major_interval=None,
    minor_interval=None,
    custom_ticks=None,
    custom_labels=None
):
    target_axis = axis.xaxis if axis_name == "x" else axis.yaxis
    tick_mode = tick_mode if tick_mode in TICK_MODES else "auto"

    if tick_mode == "custom":
        ticks = parse_csv_floats(custom_ticks)
        labels = parse_csv_text(custom_labels)

        if ticks:
            target_axis.set_major_locator(FixedLocator(ticks))

            if labels and len(labels) == len(ticks):
                target_axis.set_major_formatter(FixedFormatter(labels))

        return

    if tick_mode == "uniform":
        if major_interval is not None and major_interval > 0:
            target_axis.set_major_locator(MultipleLocator(major_interval))

        if minor_interval is not None and minor_interval > 0:
            target_axis.set_minor_locator(MultipleLocator(minor_interval))

        return



def choose_legend_location_from_axes(ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    if x_max == x_min or y_max == y_min:
        return "best"

    points = []

    for line in ax.lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()

        for x, y in zip(x_data, y_data):
            try:
                x_norm = (float(x) - x_min) / (x_max - x_min)
                y_norm = (float(y) - y_min) / (y_max - y_min)
            except (TypeError, ValueError):
                continue

            if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
                points.append((x_norm, y_norm))

    for collection in ax.collections:
        try:
            offsets = collection.get_offsets()
        except Exception:
            offsets = []

        for x, y in offsets:
            try:
                x_norm = (float(x) - x_min) / (x_max - x_min)
                y_norm = (float(y) - y_min) / (y_max - y_min)
            except (TypeError, ValueError):
                continue

            if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
                points.append((x_norm, y_norm))

    if not points:
        return "best"

    corner_boxes = {
        "upper left": (0.00, 0.38, 0.62, 1.00),
        "upper right": (0.62, 1.00, 0.62, 1.00),
        "lower left": (0.00, 0.38, 0.00, 0.38),
        "lower right": (0.62, 1.00, 0.00, 0.38)
    }

    scores = {}

    for location, (x0, x1, y0, y1) in corner_boxes.items():
        score = 0

        for x, y in points:
            if x0 <= x <= x1 and y0 <= y <= y1:
                score += 1

        scores[location] = score

    return min(scores, key=scores.get)


def build_file_summary_for_ai(file_path, original_name=None):
    file_path = Path(file_path)
    summary = {
        "file_name": original_name or file_path.name,
        "file_extension": file_path.suffix.lower().lstrip("."),
        "rows": 0,
        "columns": 0,
        "column_names": [],
        "numeric_columns": [],
        "categorical_columns": [],
        "dataset_type": "uploaded_file",
        "description": "",
        "preview": ""
    }

    try:
        inspected = inspect_dataset(file_path)
        summary.update({
            "rows": inspected.get("rows", 0),
            "columns": inspected.get("columns", 0),
            "column_names": inspected.get("column_names", []),
            "numeric_columns": inspected.get("numeric_columns", []),
            "categorical_columns": inspected.get("categorical_columns", [])
        })
    except Exception as error:
        summary["description"] = f"Inspection failed: {error}"

    try:
        preview_df = read_dataset(file_path).head(5)
        summary["preview"] = preview_df.to_csv(index=False)
    except Exception:
        summary["preview"] = ""

    return summary


def build_dataset_summary_for_ai(dataset):
    if not dataset:
        return {}

    file_path = Path(dataset.get("file_path", ""))
    summary = {
        "file_name": dataset.get("file_name", file_path.name),
        "file_extension": file_path.suffix.lower().lstrip("."),
        "rows": dataset.get("rows", 0),
        "columns": dataset.get("columns", 0),
        "column_names": dataset.get("column_names", []),
        "numeric_columns": dataset.get("numeric_columns", []),
        "categorical_columns": dataset.get("categorical_columns", []),
        "dataset_type": dataset.get("dataset_type", ""),
        "description": dataset.get("description", ""),
        "dataset_id": dataset.get("dataset_id", ""),
        "source": dataset.get("source", "")
    }

    try:
        if file_path.exists():
            preview_df = read_dataset(file_path).head(5)
            summary["preview"] = preview_df.to_csv(index=False)
        else:
            summary["preview"] = ""
    except Exception:
        summary["preview"] = ""

    return summary


def save_ai_workflow_uploads(files):
    create_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    file_summaries = []

    for index, file in enumerate(files, start=1):
        if file is None or not file.filename:
            continue

        if not is_supported_file(file.filename):
            raise ValueError(f"Unsupported file type: {file.filename}")

        safe_name = secure_filename(file.filename)
        raw_path = UPLOADED_DATA_DIR / f"{timestamp}_ai_workflow_{index:03d}_{safe_name}"
        file.save(raw_path)

        saved_files.append({
            "original_name": file.filename,
            "saved_path": str(raw_path),
            "saved_name": raw_path.name
        })

        file_summaries.append(
            build_file_summary_for_ai(
                file_path=raw_path,
                original_name=file.filename
            )
        )

    return saved_files, file_summaries


def parse_dataset_ids_from_request(payload):
    dataset_ids = []

    if isinstance(payload, dict):
        raw_ids = payload.get("dataset_ids", [])

        if isinstance(raw_ids, str):
            dataset_ids.extend([item.strip() for item in raw_ids.split(",") if item.strip()])
        elif isinstance(raw_ids, list):
            dataset_ids.extend([str(item).strip() for item in raw_ids if str(item).strip()])

    form_ids = request.form.getlist("workflow_dataset_ids")

    for item in form_ids:
        if item:
            dataset_ids.extend([value.strip() for value in str(item).split(",") if value.strip()])

    return list(dict.fromkeys(dataset_ids))


def build_selected_dataset_summaries(dataset_ids):
    summaries = []

    for dataset_id in dataset_ids:
        dataset = get_dataset(dataset_id)

        if dataset is not None:
            summaries.append(build_dataset_summary_for_ai(dataset))

    return summaries


def extract_workflow_mapping_preview(plan):
    preview = []

    if not isinstance(plan, dict):
        return preview

    for step in plan.get("steps", []):
        params = step.get("parameters") or {}
        file_condition_map = params.get("file_condition_map") or []

        for item in file_condition_map:
            preview.append({
                "step_id": step.get("step_id"),
                "action": step.get("action", ""),
                "file_name": item.get("file_name", ""),
                "condition": item.get("condition", ""),
                "dataset_label": item.get("dataset_label", "")
            })

    return preview


def register_dataset(file_path, dataset_type, source, uploaded_by="user", description=""):
    manifest = read_manifest()
    data_info = inspect_dataset(file_path)

    dataset_id = f"{dataset_type}_{Path(file_path).stem}_{uuid.uuid4().hex[:8]}"

    dataset_info = {
        "dataset_id": dataset_id,
        "file_name": Path(file_path).name,
        "file_path": str(file_path),
        "source": source,
        "dataset_type": dataset_type,
        "uploaded_by": uploaded_by,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": description,
        **data_info
    }

    manifest["datasets"].append(dataset_info)
    write_manifest(manifest)

    return dataset_info


def register_test_data_if_needed():
    manifest = read_manifest()
    existing_paths = set()

    for item in manifest["datasets"]:
        file_path = Path(item.get("file_path", ""))

        if file_path.exists():
            existing_paths.add(file_path.resolve())

    for csv_path in TEST_DATA_DIR.glob("*.csv"):
        if csv_path.resolve() not in existing_paths:
            register_dataset(
                file_path=csv_path,
                dataset_type="test_data",
                source="Built-in test dataset",
                uploaded_by="system",
                description="Built-in CSV dataset for testing automatic plotting."
            )


def get_datasets():
    register_test_data_if_needed()
    manifest = read_manifest()

    datasets = []

    for item in manifest["datasets"]:
        file_path = Path(item.get("file_path", ""))

        if file_path.exists():
            datasets.append(item)

    return datasets


def get_dataset(dataset_id):
    datasets = get_datasets()

    for dataset in datasets:
        if dataset["dataset_id"] == dataset_id:
            return dataset

    return None


def get_column_types(dataset):
    numeric_columns = dataset.get("numeric_columns", [])
    categorical_columns = dataset.get("categorical_columns", [])

    column_types = {}

    for column in numeric_columns:
        column_types[column] = "numeric"

    for column in categorical_columns:
        column_types[column] = "categorical"

    return column_types


def suggest_plot_types(x_type, y_type):
    if x_type == "numeric" and y_type == "numeric":
        return ["scatter", "line"]

    if x_type == "categorical" and y_type == "numeric":
        return ["bar", "box"]

    if x_type == "categorical" and y_type in ["none", None, ""]:
        return ["count"]

    if x_type == "categorical" and y_type == "categorical":
        return ["count"]

    if x_type == "numeric" and y_type in ["none", None, ""]:
        return ["histogram"]

    return ["scatter"]


def summary_value(series, method, tail_fraction):
    series = series.dropna()

    if series.empty:
        return pd.NA

    if method == "mean":
        return series.mean()

    if method == "median":
        return series.median()

    if method == "first":
        return series.iloc[0]

    if method == "last":
        return series.iloc[-1]

    if method == "min":
        return series.min()

    if method == "max":
        return series.max()

    if method == "mean_tail":
        n_tail = max(1, int(len(series) * tail_fraction))
        return series.tail(n_tail).mean()

    return series.mean()


def summarize_by_group(df, x_col, y_col, group_col, x_method, y_method, tail_fraction):
    rows = []

    keep_columns = [
        "source_file",
        "source_index",
        "sequence_index",
        "condition",
        "dataset_label",
        "replicate",
        "step_variable_name",
        "step_variable_value",
        "step_label",
        "global_time_min",
        "E_RHE"
    ]

    for group_name, group in df.groupby(group_col, sort=False):
        group = group.dropna(subset=[x_col, y_col])

        if group.empty:
            continue

        row = {
            group_col: group_name,
            x_col: summary_value(pd.to_numeric(group[x_col], errors="coerce"), x_method, tail_fraction),
            y_col: summary_value(pd.to_numeric(group[y_col], errors="coerce"), y_method, tail_fraction)
        }

        for column in keep_columns:
            if column in group.columns and column not in row:
                if column in ["step_variable_value", "E_RHE"]:
                    row[column] = pd.to_numeric(group[column], errors="coerce").mean()
                else:
                    row[column] = group[column].iloc[0]

        rows.append(row)

    return pd.DataFrame(rows)


def smooth_summary_data(df, x_col, y_col, window):
    data = df[[x_col, y_col]].dropna().sort_values(by=x_col)

    if len(data) < 3:
        return data

    window = min(window, len(data))

    if window % 2 == 0:
        window -= 1

    if window < 3:
        return data

    smoothed = data.copy()
    smoothed[y_col] = smoothed[y_col].rolling(
        window=window,
        center=True,
        min_periods=1
    ).mean()

    return smoothed


def order_data(data, x_col, line_order):
    if line_order == "sort_x":
        return data.sort_values(by=x_col)

    if "sequence_index" in data.columns:
        return data.sort_values(by="sequence_index")

    if "source_index" in data.columns:
        return data.sort_values(by="source_index")

    return data


def apply_plot_style(
    ax,
    axis_label_size,
    tick_label_size,
    axis_label_weight,
    spine_width,
    tick_width,
    tick_length,
    show_full_frame,
    tick_direction,
    show_top_ticks,
    show_bottom_ticks,
    show_left_ticks,
    show_right_ticks,
    show_grid,
    title_size,
    grid_axis="both",
    grid_which="major",
    major_grid_linestyle="solid",
    minor_grid_linestyle="dashed",
    major_grid_alpha=0.25,
    minor_grid_alpha=0.12,
    major_grid_width=0.7,
    minor_grid_width=0.5
):
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    if show_full_frame:
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        which="both",
        direction=tick_direction,
        top=show_top_ticks,
        bottom=show_bottom_ticks,
        left=show_left_ticks,
        right=show_right_ticks,
        width=tick_width,
        length=tick_length,
        labelsize=tick_label_size
    )

    ax.xaxis.label.set_size(axis_label_size)
    ax.yaxis.label.set_size(axis_label_size)
    ax.xaxis.label.set_weight(axis_label_weight)
    ax.yaxis.label.set_weight(axis_label_weight)
    ax.title.set_size(title_size)
    ax.title.set_weight(axis_label_weight)

    ax.grid(False)

    if show_grid:
        if grid_axis not in {"x", "y", "both"}:
            grid_axis = "both"

        if grid_which not in {"major", "minor", "both"}:
            grid_which = "major"

        if grid_which in {"minor", "both"}:
            ax.minorticks_on()

        if grid_which in {"major", "both"}:
            ax.grid(
                True,
                which="major",
                axis=grid_axis,
                linestyle=major_grid_linestyle,
                linewidth=major_grid_width,
                alpha=major_grid_alpha
            )

        if grid_which in {"minor", "both"}:
            ax.grid(
                True,
                which="minor",
                axis=grid_axis,
                linestyle=minor_grid_linestyle,
                linewidth=minor_grid_width,
                alpha=minor_grid_alpha
            )


def style_secondary_axis(axis, axis_label_size, tick_label_size, axis_label_weight, spine_width, tick_width, tick_length, tick_direction):
    for spine in axis.spines.values():
        spine.set_linewidth(spine_width)

    axis.tick_params(
        axis="both",
        which="both",
        direction=tick_direction,
        width=tick_width,
        length=tick_length,
        labelsize=tick_label_size
    )

    axis.xaxis.label.set_size(axis_label_size)
    axis.yaxis.label.set_size(axis_label_size)
    axis.xaxis.label.set_weight(axis_label_weight)
    axis.yaxis.label.set_weight(axis_label_weight)


def plot_single_curve(
    ax,
    df,
    x_col,
    y_col,
    plot_type,
    marker_color,
    line_color,
    label,
    line_order,
    show_markers,
    marker_size,
    line_width,
    opacity
):
    if plot_type in ["scatter", "line", "bar", "box"] and y_col in [None, "", "none"]:
        raise ValueError("This plot type requires a Y variable.")

    if plot_type == "scatter":
        data = df[[x_col, y_col]].dropna()
        ax.scatter(
            data[x_col],
            data[y_col],
            facecolor=marker_color,
            edgecolor=line_color,
            alpha=opacity,
            s=marker_size,
            label=label if label else "_nolegend_"
        )

    elif plot_type == "line":
        data = df[[x_col, y_col]].dropna()
        data = order_data(data, x_col, line_order)
        marker = "o" if show_markers else None

        ax.plot(
            data[x_col],
            data[y_col],
            marker=marker,
            color=line_color,
            markerfacecolor=marker_color,
            markeredgecolor=line_color,
            linewidth=line_width,
            markersize=max(1, marker_size ** 0.5),
            alpha=opacity,
            label=label if label else "_nolegend_"
        )

    elif plot_type == "bar":
        data = df[[x_col, y_col]].dropna()
        grouped = data.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        ax.bar(
            grouped.index.astype(str),
            grouped.values,
            color=marker_color,
            edgecolor=line_color,
            alpha=opacity,
            label=label if label else "_nolegend_"
        )

    elif plot_type == "box":
        data = df[[x_col, y_col]].dropna()
        grouped = data.groupby(x_col)
        groups = [group[y_col].values for _, group in grouped]
        labels = [str(name) for name, _ in grouped]
        box = ax.boxplot(groups, labels=labels, patch_artist=True)

        for patch in box["boxes"]:
            patch.set_facecolor(marker_color)
            patch.set_edgecolor(line_color)
            patch.set_alpha(opacity)

        for item in box["whiskers"] + box["caps"] + box["medians"]:
            item.set_color(line_color)

    elif plot_type == "count":
        data = df[[x_col]].dropna()
        counts = data[x_col].astype(str).value_counts()
        ax.bar(
            counts.index,
            counts.values,
            color=marker_color,
            edgecolor=line_color,
            alpha=opacity,
            label=label if label else "_nolegend_"
        )

    elif plot_type == "histogram":
        data = df[[x_col]].dropna()
        ax.hist(
            data[x_col],
            bins=20,
            facecolor=marker_color,
            edgecolor=line_color,
            alpha=opacity,
            label=label if label else "_nolegend_"
        )

    else:
        raise ValueError("Unsupported plot type.")


def plot_grouped_curves(
    ax,
    df,
    x_col,
    y_col,
    group_col,
    plot_type,
    line_order,
    show_markers,
    marker_size,
    line_width,
    opacity,
    marker_color,
    line_color,
    group_color_mode
):
    if plot_type not in ["line", "scatter"]:
        raise ValueError("Group-by plotting is only supported for line or scatter plots.")

    if y_col in [None, "", "none"]:
        raise ValueError("Group-by plotting requires a Y variable.")

    columns = [x_col, y_col, group_col]

    for optional_col in ["sequence_index", "source_index"]:
        if optional_col in df.columns and optional_col not in columns:
            columns.append(optional_col)

    data = df[columns].dropna(subset=[x_col, y_col, group_col])

    if data.empty:
        raise ValueError("No valid data points were found for the selected group-by plot.")

    for group_name, group in data.groupby(group_col, sort=False):
        label = normalize_matplotlib_text(str(group_name))

        if plot_type == "scatter":
            if group_color_mode == "same":
                ax.scatter(
                    group[x_col],
                    group[y_col],
                    facecolor=marker_color,
                    edgecolor=line_color,
                    alpha=opacity,
                    s=marker_size,
                    label=label
                )
            else:
                ax.scatter(
                    group[x_col],
                    group[y_col],
                    alpha=opacity,
                    s=marker_size,
                    label=label
                )
        else:
            group = order_data(group, x_col, line_order)
            marker = "o" if show_markers else None

            if group_color_mode == "same":
                ax.plot(
                    group[x_col],
                    group[y_col],
                    marker=marker,
                    color=line_color,
                    markerfacecolor=marker_color,
                    markeredgecolor=line_color,
                    linewidth=line_width,
                    markersize=max(1, marker_size ** 0.5),
                    alpha=opacity,
                    label=label
                )
            else:
                ax.plot(
                    group[x_col],
                    group[y_col],
                    marker=marker,
                    linewidth=line_width,
                    markersize=max(1, marker_size ** 0.5),
                    alpha=opacity,
                    label=label
                )


def plot_summary_curve(
    ax,
    summary_df,
    x_col,
    y_col,
    fit_guide,
    line_order,
    show_markers,
    marker_color,
    line_color,
    marker_size,
    line_width,
    opacity,
    smooth_window,
    label
):
    columns = [x_col, y_col]

    if "sequence_index" in summary_df.columns:
        columns.append("sequence_index")

    data = summary_df[columns].dropna(subset=[x_col, y_col])

    if data.empty:
        raise ValueError("No summarized data points are available for plotting.")

    data = order_data(data, x_col, line_order)

    if fit_guide == "none":
        ax.scatter(
            data[x_col],
            data[y_col],
            facecolor=marker_color,
            edgecolor=line_color,
            s=marker_size,
            alpha=opacity,
            label=label if label else "_nolegend_"
        )
        return

    if fit_guide == "connect":
        marker = "o" if show_markers else None
        ax.plot(
            data[x_col],
            data[y_col],
            marker=marker,
            color=line_color,
            markerfacecolor=marker_color,
            markeredgecolor=line_color,
            linewidth=line_width,
            markersize=max(1, marker_size ** 0.5),
            alpha=opacity,
            label=label if label else "_nolegend_"
        )
        return

    if fit_guide == "smooth":
        ax.scatter(
            data[x_col],
            data[y_col],
            facecolor=marker_color,
            edgecolor=line_color,
            s=marker_size,
            alpha=max(0.3, opacity),
            label=label if label else "_nolegend_"
        )

        smoothed = smooth_summary_data(data, x_col, y_col, smooth_window)

        ax.plot(
            smoothed[x_col],
            smoothed[y_col],
            color=line_color,
            linewidth=line_width,
            alpha=opacity,
            label="Smooth guide" if label else "_nolegend_"
        )
        return

    raise ValueError("Unsupported fit guide option.")


def plot_secondary_line_or_scatter(
    ax,
    df,
    x_col,
    y_col,
    plot_type,
    marker_color,
    line_color,
    label,
    line_order,
    show_markers,
    marker_size,
    line_width,
    opacity
):
    data = df[[x_col, y_col]].dropna()
    data = order_data(data, x_col, line_order)

    if plot_type == "scatter":
        ax.scatter(
            data[x_col],
            data[y_col],
            facecolor=marker_color,
            edgecolor=line_color,
            alpha=opacity,
            s=marker_size,
            label=label if label else "_nolegend_"
        )

    else:
        marker = "o" if show_markers else None

        ax.plot(
            data[x_col],
            data[y_col],
            marker=marker,
            color=line_color,
            markerfacecolor=marker_color,
            markeredgecolor=line_color,
            linewidth=line_width,
            markersize=max(1, marker_size ** 0.5),
            alpha=opacity,
            label=label if label else "_nolegend_"
        )


def build_auto_step_records(axis_df, x_col, step_value_col, step_group_col, decimal_places):
    if step_value_col in [None, "", "none"]:
        return []

    if step_group_col in [None, "", "none"]:
        return []

    if step_value_col not in axis_df.columns or step_group_col not in axis_df.columns or x_col not in axis_df.columns:
        return []

    data = axis_df[[x_col, step_value_col, step_group_col]].dropna()

    if data.empty:
        return []

    records = []

    for group_value, group in data.groupby(step_group_col, sort=True):
        x_value = pd.to_numeric(group[x_col], errors="coerce").mean()
        step_value = pd.to_numeric(group[step_value_col], errors="coerce").mean()

        if pd.notna(x_value) and pd.notna(step_value):
            records.append({
                "x": float(x_value),
                "label": format_step_value(step_value, decimal_places),
                "group": group_value
            })

    return records


def build_step_labels_and_records(records, step_axis_mode, custom_labels, max_ticks, label_stride):
    labels_from_user = parse_csv_text(custom_labels)

    if step_axis_mode == "uniform_custom" and labels_from_user:
        return labels_from_user, []

    if not records:
        return [], []

    max_ticks = max(2, max_ticks)
    label_stride = max(1, label_stride)

    if len(records[::label_stride]) > max_ticks:
        label_stride = max(label_stride, int((len(records) + max_ticks - 1) / max_ticks))

    selected_records = records[::label_stride]
    labels = [record["label"] for record in selected_records]

    return labels, selected_records


def setup_step_axis(
    ax,
    axis_df,
    x_col,
    x_label,
    step_value_col,
    step_group_col,
    step_axis_label,
    step_axis_mode,
    step_axis_placement,
    max_ticks,
    decimal_places,
    label_stride,
    custom_labels,
    custom_positions,
    label_rotation,
    label_pad,
    x_min,
    x_max
):
    top_axis = ax.twiny()
    top_axis.patch.set_alpha(0)
    top_axis.set_xlabel(x_label if x_label else format_column_label(x_col))
    top_axis.xaxis.tick_top()
    top_axis.xaxis.set_label_position("top")
    top_axis.spines["bottom"].set_visible(False)
    top_axis.grid(False)

    step_axis_mode = step_axis_mode if step_axis_mode in STEP_AXIS_MODES else "auto_data"
    step_axis_placement = step_axis_placement if step_axis_placement in STEP_AXIS_PLACEMENTS else "uniform"

    records = build_auto_step_records(
        axis_df=axis_df,
        x_col=x_col,
        step_value_col=step_value_col,
        step_group_col=step_group_col,
        decimal_places=decimal_places
    )

    labels, selected_records = build_step_labels_and_records(
        records=records,
        step_axis_mode=step_axis_mode,
        custom_labels=custom_labels,
        max_ticks=max_ticks,
        label_stride=label_stride
    )

    if not labels:
        return top_axis

    x_start, x_end = get_axis_x_range(axis_df, x_col, x_min, x_max, ax)
    positions_from_user = parse_csv_floats(custom_positions)

    if step_axis_placement == "custom_positions" and positions_from_user:
        ticks = positions_from_user[:len(labels)]
        labels = labels[:len(ticks)]

    elif step_axis_placement == "data_positions":
        if step_axis_mode == "auto_data" and selected_records:
            ticks = [record["x"] for record in selected_records]
            labels = labels[:len(ticks)]
        elif records:
            position_records = evenly_select_records(records, len(labels))
            ticks = [record["x"] for record in position_records]
            labels = labels[:len(ticks)]
        else:
            if len(labels) == 1:
                ticks = [(x_start + x_end) / 2]
            else:
                ticks = [
                    x_start + index * (x_end - x_start) / (len(labels) - 1)
                    for index in range(len(labels))
                ]

    else:
        if len(labels) == 1:
            ticks = [(x_start + x_end) / 2]
        else:
            ticks = [
                x_start + index * (x_end - x_start) / (len(labels) - 1)
                for index in range(len(labels))
            ]

    if not ticks:
        return top_axis

    label_alignment = "center" if float(label_rotation) == 0 else "right"

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=label_rotation, ha=label_alignment)
    ax.set_xlabel(step_axis_label if step_axis_label else format_column_label(step_value_col), labelpad=label_pad)

    return top_axis

def normalize_matplotlib_text(value):
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"none", "null", "n/a", "na"}:
        return ""

    symbol_replacements = {
        "j /": r"$j$ /",
        "i /": r"$i$ /",
        "I /": r"$I$ /",
        "E /": r"$E$ /",
        "V /": r"$V$ /",
        "t /": r"$t$ /",
        "T /": r"$T$ /",
        "C /": r"$C$ /",
        "R /": r"$R$ /",
        "Z /": r"$Z$ /",
        "eta /": r"$\eta$ /",
        "η /": r"$\eta$ /",
        "alpha /": r"$\alpha$ /",
        "α /": r"$\alpha$ /",
        "beta /": r"$\beta$ /",
        "β /": r"$\beta$ /",
        "lambda /": r"$\lambda$ /",
        "λ /": r"$\lambda$ /",
        "theta /": r"$\theta$ /",
        "θ /": r"$\theta$ /"
    }

    for old, new in symbol_replacements.items():
        text = text.replace(old, new)

    unit_replacements = [
        ("mA cm^-2", r"mA cm$^{-2}$"),
        ("A cm^-2", r"A cm$^{-2}$"),
        ("uA cm^-2", r"$\mu$A cm$^{-2}$"),
        ("µA cm^-2", r"$\mu$A cm$^{-2}$"),
        ("μA cm^-2", r"$\mu$A cm$^{-2}$"),
        ("mA/cm^2", r"mA cm$^{-2}$"),
        ("A/cm^2", r"A cm$^{-2}$"),
        ("uA/cm^2", r"$\mu$A cm$^{-2}$"),
        ("µA/cm^2", r"$\mu$A cm$^{-2}$"),
        ("μA/cm^2", r"$\mu$A cm$^{-2}$"),
        ("mA cm-2", r"mA cm$^{-2}$"),
        ("A cm-2", r"A cm$^{-2}$"),
        ("mol cm^-2", r"mol cm$^{-2}$"),
        ("mol/cm^2", r"mol cm$^{-2}$"),
        ("F cm^-2", r"F cm$^{-2}$"),
        ("F/cm^2", r"F cm$^{-2}$"),
        ("S cm^-1", r"S cm$^{-1}$"),
        ("S/cm", r"S cm$^{-1}$"),
        ("ohm cm", r"$\Omega$ cm"),
        ("Ohm cm", r"$\Omega$ cm"),
        ("Ω cm", r"$\Omega$ cm"),
        ("ohm", r"$\Omega$"),
        ("Ohm", r"$\Omega$"),
        ("degC", r"$^\circ$C"),
        ("°C", r"$^\circ$C"),
        ("cm^-1", r"cm$^{-1}$"),
        ("cm^-2", r"cm$^{-2}$"),
        ("cm^-3", r"cm$^{-3}$"),
        ("m^-1", r"m$^{-1}$"),
        ("m^-2", r"m$^{-2}$"),
        ("m^-3", r"m$^{-3}$"),
        ("s^-1", r"s$^{-1}$"),
        ("min^-1", r"min$^{-1}$"),
        ("h^-1", r"h$^{-1}$"),
        ("g^-1", r"g$^{-1}$"),
        ("kg^-1", r"kg$^{-1}$"),
        ("mAh g^-1", r"mAh g$^{-1}$"),
        ("mAh/g", r"mAh g$^{-1}$"),
        ("Wh kg^-1", r"Wh kg$^{-1}$"),
        ("Wh/kg", r"Wh kg$^{-1}$"),
        ("mV dec^-1", r"mV dec$^{-1}$"),
        ("mV/dec", r"mV dec$^{-1}$")
    ]

    for old, new in unit_replacements:
        text = text.replace(old, new)

    text = re.sub(r"\^(-?\d+)", r"$^{\1}$", text)
    text = re.sub(r"(?<!\$)\bcm-([123])\b", r"cm$^{-\1}$", text)
    text = re.sub(r"(?<!\$)\bm-([123])\b", r"m$^{-\1}$", text)
    text = text.replace("micro", r"$\mu$")
    text = text.replace("NaNO3", r"NaNO$_3$")
    text = text.replace("NO3", r"NO$_3$")

    return text


def format_column_label(column_name):
    if column_name in [None, "", "none"]:
        return ""

    text = str(column_name).strip()

    known_labels = {
        "global_time_min": r"$t$ / min",
        "global_time_s": r"$t$ / s",
        "T_s": r"$t$ / s",
        "Potential_V": r"$E$ / V",
        "E_RHE": r"$E$ / V vs. RHE",
        "Vf_V_vs_Ref": r"$E$ / V vs. Ref",
        "Current_mA": r"$I$ / mA",
        "Im_A": r"$I$ / A",
        "j_mA_cm2": r"$j$ / mA cm$^{-2}$",
        "j_A_cm2": r"$j$ / A cm$^{-2}$",
        "Z_ohm": r"$Z$ / $\Omega$",
        "phase_deg": r"Phase / degree",
        "frequency_Hz": r"Frequency / Hz"
    }

    if text in known_labels:
        return known_labels[text]

    text = text.replace("_per_", "/")
    text = text.replace("_", " ")
    text = text.replace(" cm2", " cm^-2")
    text = text.replace(" cm3", " cm^-3")

    return normalize_matplotlib_text(text)


def create_plot(
    dataset_id,
    x_col,
    y_col,
    x_label,
    y_label,
    plot_type,
    plot_title,
    bottom_annotation,
    marker_color,
    line_color,
    primary_label,
    group_col,
    group_label,
    group_color_mode,
    data_reduction,
    summary_group_col,
    x_summary_method,
    y_summary_method,
    tail_fraction,
    fit_guide,
    smooth_window,
    use_step_axis,
    step_axis_mode,
    step_axis_placement,
    step_axis_value_col,
    step_axis_group_col,
    step_axis_label,
    step_axis_max_ticks,
    secondary_mode,
    x2_col,
    y2_col,
    top_x_label,
    right_y_label,
    secondary_plot_type,
    second_marker_color,
    second_line_color,
    second_label,
    line_order,
    show_markers,
    show_legend,
    marker_size,
    line_width,
    opacity,
    axis_label_size,
    tick_label_size,
    axis_label_weight,
    spine_width,
    tick_width,
    tick_length,
    show_full_frame,
    tick_direction,
    show_top_ticks,
    show_bottom_ticks,
    show_left_ticks,
    show_right_ticks,
    show_grid,
    title_size,
    legend_font_size,
    legend_location,
    legend_frame,
    figure_width,
    figure_height,
    figure_dpi,
    x_min,
    x_max,
    x_tick_mode,
    x_major_interval,
    x_minor_interval,
    x_custom_ticks,
    x_custom_tick_labels,
    y_min,
    y_max,
    y_tick_mode,
    y_major_interval,
    y_minor_interval,
    y_custom_ticks,
    y_custom_tick_labels,
    step_axis_decimal_places,
    step_axis_label_stride,
    step_axis_custom_labels,
    step_axis_custom_positions,
    step_axis_label_rotation,
    step_axis_label_pad,
    bottom_margin,
    grid_axis="both",
    grid_which="major",
    major_grid_linestyle="solid",
    minor_grid_linestyle="dashed",
    major_grid_alpha=0.25,
    minor_grid_alpha=0.12,
    major_grid_width=0.7,
    minor_grid_width=0.5
):
    dataset = get_dataset(dataset_id)

    if dataset is None:
        raise ValueError("Dataset not found.")

    data_path = Path(dataset["file_path"])
    df = read_dataset(data_path)

    x_label = normalize_matplotlib_text(x_label)
    y_label = normalize_matplotlib_text(y_label)
    plot_title = normalize_matplotlib_text(plot_title)
    bottom_annotation = normalize_matplotlib_text(bottom_annotation)
    primary_label = normalize_matplotlib_text(primary_label)
    group_label = normalize_matplotlib_text(group_label)
    step_axis_label = normalize_matplotlib_text(step_axis_label)
    top_x_label = normalize_matplotlib_text(top_x_label)
    right_y_label = normalize_matplotlib_text(right_y_label)
    second_label = normalize_matplotlib_text(second_label)

    group_col = None if group_col in [None, "", "none"] else group_col
    summary_group_col = None if summary_group_col in [None, "", "none"] else summary_group_col

    if secondary_mode not in SECONDARY_MODES:
        raise ValueError("Invalid secondary mode.")

    if step_axis_mode not in STEP_AXIS_MODES:
        step_axis_mode = "auto_data"

    if step_axis_placement not in STEP_AXIS_PLACEMENTS:
        step_axis_placement = "uniform"

    if x_tick_mode not in TICK_MODES:
        x_tick_mode = "auto"

    if y_tick_mode not in TICK_MODES:
        y_tick_mode = "auto"

    if x_col not in df.columns:
        raise ValueError("Primary X column not found.")

    if y_col and y_col != "none" and y_col not in df.columns:
        raise ValueError("Primary Y column not found.")

    if group_col is not None and group_col not in df.columns:
        raise ValueError("Group-by column not found.")

    if data_reduction == "summary":
        secondary_mode = "none"
        group_col = None

        if summary_group_col is None:
            raise ValueError("Summary mode requires a summary group column.")

        if summary_group_col not in df.columns:
            raise ValueError("Summary group column not found.")

    if use_step_axis:
        secondary_mode = "none"

    if group_col is not None and secondary_mode != "none":
        raise ValueError("Group-by plotting cannot be used together with secondary-axis mode.")

    if secondary_mode != "none" and plot_type not in ["line", "scatter"]:
        raise ValueError("Secondary curve mode is only supported for line or scatter plots.")

    if secondary_mode != "none" and y_col in [None, "", "none"]:
        raise ValueError("Secondary curve mode requires a primary Y variable.")

    if secondary_mode == "same_y_different_x":
        if x2_col in [None, "", "none"]:
            raise ValueError("Second X variable is required for Y(X1) and Y(X2) mode.")

        if x2_col not in df.columns:
            raise ValueError("Second X column not found.")

        y2_col = y_col

    if secondary_mode == "same_x_different_y":
        if y2_col in [None, "", "none"]:
            raise ValueError("Second Y variable is required for Y1(X) and Y2(X) mode.")

        if y2_col not in df.columns:
            raise ValueError("Second Y column not found.")

        x2_col = x_col

    marker_color = sanitize_color(marker_color, "#FF5F05")
    line_color = sanitize_color(line_color, "#13294B")
    second_marker_color = sanitize_color(second_marker_color, "#9A4DFF")
    second_line_color = sanitize_color(second_line_color, "#9A4DFF")

    line_order = "sort_x" if line_order == "sort_x" else "original"
    marker_size = clamp_float(marker_size, 18, 1, 200)
    line_width = clamp_float(line_width, 2.2, 0.1, 10)
    opacity = clamp_float(opacity, 1.0, 0.05, 1.0)
    tail_fraction = clamp_float(tail_fraction, 0.2, 0.01, 1.0)
    smooth_window = clamp_int(smooth_window, 5, 3, 99)
    step_axis_max_ticks = clamp_int(step_axis_max_ticks, 12, 2, 64)
    group_color_mode = "auto" if group_color_mode == "auto" else "same"

    axis_label_size = clamp_float(axis_label_size, 18, 6, 80)
    tick_label_size = clamp_float(tick_label_size, 13, 6, 60)
    spine_width = clamp_float(spine_width, 1.2, 0.1, 10)
    tick_width = clamp_float(tick_width, 1.2, 0.1, 10)
    tick_length = clamp_float(tick_length, 5, 1, 30)
    title_size = clamp_float(title_size, 18, 6, 80)
    legend_font_size = clamp_float(legend_font_size, 11, 6, 60)
    allowed_legend_locations = {"auto", "best", "upper right", "upper left", "lower right", "lower left"}
    legend_location = legend_location if legend_location in allowed_legend_locations else "best"
    legend_frame = bool(legend_frame)

    grid_axis = grid_axis if grid_axis in {"x", "y", "both"} else "both"
    grid_which = grid_which if grid_which in {"major", "minor", "both"} else "major"

    allowed_grid_linestyles = {
        "solid": "solid",
        "-": "solid",
        "dashed": "dashed",
        "--": "dashed",
        "dotted": "dotted",
        ":": "dotted",
        "dashdot": "dashdot",
        "-.": "dashdot"
    }
    major_grid_linestyle = allowed_grid_linestyles.get(str(major_grid_linestyle).strip().lower(), "solid")
    minor_grid_linestyle = allowed_grid_linestyles.get(str(minor_grid_linestyle).strip().lower(), "dashed")
    major_grid_alpha = clamp_float(major_grid_alpha, 0.25, 0.0, 1.0)
    minor_grid_alpha = clamp_float(minor_grid_alpha, 0.12, 0.0, 1.0)
    major_grid_width = clamp_float(major_grid_width, 0.7, 0.05, 5.0)
    minor_grid_width = clamp_float(minor_grid_width, 0.5, 0.05, 5.0)

    axis_label_weight = "normal" if axis_label_weight == "normal" else "bold"
    tick_direction = tick_direction if tick_direction in ["in", "out", "inout"] else "in"

    figure_width = clamp_float(figure_width, 8, 3, 30)
    figure_height = clamp_float(figure_height, 5, 2, 20)
    figure_dpi = clamp_int(figure_dpi, 150, 72, 600)

    x_min = optional_float(x_min)
    x_max = optional_float(x_max)
    x_major_interval = optional_float(x_major_interval)
    x_minor_interval = optional_float(x_minor_interval)

    y_min = optional_float(y_min)
    y_max = optional_float(y_max)
    y_major_interval = optional_float(y_major_interval)
    y_minor_interval = optional_float(y_minor_interval)

    if x_tick_mode == "auto" and (x_major_interval is not None or x_minor_interval is not None):
        x_tick_mode = "uniform"

    if y_tick_mode == "auto" and (y_major_interval is not None or y_minor_interval is not None):
        y_tick_mode = "uniform"

    step_axis_decimal_places = clamp_int(step_axis_decimal_places, 1, 0, 6)
    step_axis_label_stride = clamp_int(step_axis_label_stride, 2, 1, 100)
    step_axis_label_rotation = clamp_float(step_axis_label_rotation, 0, 0, 90)
    step_axis_label_pad = clamp_float(step_axis_label_pad, 14, 0, 80)
    bottom_margin = optional_float(bottom_margin)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=figure_dpi)
    axis_df = df

    if data_reduction == "summary":
        summary_df = summarize_by_group(
            df=df,
            x_col=x_col,
            y_col=y_col,
            group_col=summary_group_col,
            x_method=x_summary_method,
            y_method=y_summary_method,
            tail_fraction=tail_fraction
        )

        axis_df = summary_df

        plot_summary_curve(
            ax=ax,
            summary_df=summary_df,
            x_col=x_col,
            y_col=y_col,
            fit_guide=fit_guide,
            line_order=line_order,
            show_markers=show_markers,
            marker_color=marker_color,
            line_color=line_color,
            marker_size=marker_size,
            line_width=line_width,
            opacity=opacity,
            smooth_window=smooth_window,
            label=primary_label
        )

    elif group_col is not None:
        plot_grouped_curves(
            ax=ax,
            df=df,
            x_col=x_col,
            y_col=y_col,
            group_col=group_col,
            plot_type=plot_type,
            line_order=line_order,
            show_markers=show_markers,
            marker_size=marker_size,
            line_width=line_width,
            opacity=opacity,
            marker_color=marker_color,
            line_color=line_color,
            group_color_mode=group_color_mode
        )

    else:
        plot_single_curve(
            ax=ax,
            df=df,
            x_col=x_col,
            y_col=y_col,
            plot_type=plot_type,
            marker_color=marker_color,
            line_color=line_color,
            label=primary_label,
            line_order=line_order,
            show_markers=show_markers,
            marker_size=marker_size,
            line_width=line_width,
            opacity=opacity
        )

    final_x_label = x_label if x_label else format_column_label(x_col)

    if plot_type in ["count", "histogram"]:
        final_y_label = y_label if y_label else "Count"
    else:
        final_y_label = y_label if y_label else format_column_label(y_col)

    ax.set_xlabel(final_x_label)
    ax.set_ylabel(final_y_label)

    secondary_axis = None

    if secondary_mode == "same_y_different_x":
        secondary_axis = ax.twiny()
        secondary_axis.patch.set_alpha(0)
        secondary_axis.xaxis.tick_top()
        secondary_axis.xaxis.set_label_position("top")
        secondary_axis.spines["bottom"].set_visible(False)
        secondary_axis.grid(False)

        plot_secondary_line_or_scatter(
            ax=secondary_axis,
            df=df,
            x_col=x2_col,
            y_col=y_col,
            plot_type=secondary_plot_type,
            marker_color=second_marker_color,
            line_color=second_line_color,
            label=second_label,
            line_order=line_order,
            show_markers=show_markers,
            marker_size=marker_size,
            line_width=line_width,
            opacity=opacity
        )

        secondary_axis.set_xlabel(top_x_label if top_x_label else format_column_label(x2_col))

    elif secondary_mode == "same_x_different_y":
        secondary_axis = ax.twinx()
        secondary_axis.patch.set_alpha(0)
        secondary_axis.yaxis.tick_right()
        secondary_axis.yaxis.set_label_position("right")
        secondary_axis.spines["left"].set_visible(False)
        secondary_axis.grid(False)

        plot_secondary_line_or_scatter(
            ax=secondary_axis,
            df=df,
            x_col=x_col,
            y_col=y2_col,
            plot_type=secondary_plot_type,
            marker_color=second_marker_color,
            line_color=second_line_color,
            label=second_label,
            line_order=line_order,
            show_markers=show_markers,
            marker_size=marker_size,
            line_width=line_width,
            opacity=opacity
        )

        secondary_axis.set_ylabel(right_y_label if right_y_label else format_column_label(y2_col))

    if plot_title:
        ax.set_title(plot_title)

    apply_plot_style(
        ax=ax,
        axis_label_size=axis_label_size,
        tick_label_size=tick_label_size,
        axis_label_weight=axis_label_weight,
        spine_width=spine_width,
        tick_width=tick_width,
        tick_length=tick_length,
        show_full_frame=show_full_frame,
        tick_direction=tick_direction,
        show_top_ticks=show_top_ticks,
        show_bottom_ticks=show_bottom_ticks,
        show_left_ticks=show_left_ticks,
        show_right_ticks=show_right_ticks,
        show_grid=show_grid,
        title_size=title_size,
        grid_axis=grid_axis,
        grid_which=grid_which,
        major_grid_linestyle=major_grid_linestyle,
        minor_grid_linestyle=minor_grid_linestyle,
        major_grid_alpha=major_grid_alpha,
        minor_grid_alpha=minor_grid_alpha,
        major_grid_width=major_grid_width,
        minor_grid_width=minor_grid_width
    )

    step_top_axis = None

    if use_step_axis:
        step_top_axis = setup_step_axis(
            ax=ax,
            axis_df=axis_df,
            x_col=x_col,
            x_label=final_x_label,
            step_value_col=step_axis_value_col,
            step_group_col=step_axis_group_col,
            step_axis_label=step_axis_label,
            step_axis_mode=step_axis_mode,
            step_axis_placement=step_axis_placement,
            max_ticks=step_axis_max_ticks,
            decimal_places=step_axis_decimal_places,
            label_stride=step_axis_label_stride,
            custom_labels=step_axis_custom_labels,
            custom_positions=step_axis_custom_positions,
            label_rotation=step_axis_label_rotation,
            label_pad=step_axis_label_pad,
            x_min=x_min,
            x_max=x_max
        )

        ax.tick_params(axis="x", top=False, labeltop=False, rotation=step_axis_label_rotation)
    else:
        ax.tick_params(axis="x", rotation=30)

    if secondary_axis is not None:
        style_secondary_axis(
            axis=secondary_axis,
            axis_label_size=axis_label_size,
            tick_label_size=tick_label_size,
            axis_label_weight=axis_label_weight,
            spine_width=spine_width,
            tick_width=tick_width,
            tick_length=tick_length,
            tick_direction=tick_direction
        )

    if step_top_axis is not None:
        style_secondary_axis(
            axis=step_top_axis,
            axis_label_size=axis_label_size,
            tick_label_size=tick_label_size,
            axis_label_weight=axis_label_weight,
            spine_width=spine_width,
            tick_width=tick_width,
            tick_length=tick_length,
            tick_direction=tick_direction
        )

    apply_axis_limits(ax, "x", x_min, x_max)
    apply_axis_limits(ax, "y", y_min, y_max)

    if use_step_axis and step_top_axis is not None:
        step_top_axis.set_xlim(ax.get_xlim())

    if use_step_axis and step_top_axis is not None:
        apply_axis_tick_control(
            axis=step_top_axis,
            axis_name="x",
            tick_mode=x_tick_mode,
            major_interval=x_major_interval,
            minor_interval=x_minor_interval,
            custom_ticks=x_custom_ticks,
            custom_labels=x_custom_tick_labels
        )
    else:
        apply_axis_tick_control(
            axis=ax,
            axis_name="x",
            tick_mode=x_tick_mode,
            major_interval=x_major_interval,
            minor_interval=x_minor_interval,
            custom_ticks=x_custom_ticks,
            custom_labels=x_custom_tick_labels
        )

    apply_axis_tick_control(
        axis=ax,
        axis_name="y",
        tick_mode=y_tick_mode,
        major_interval=y_major_interval,
        minor_interval=y_minor_interval,
        custom_ticks=y_custom_ticks,
        custom_labels=y_custom_tick_labels
    )

    handles_primary, labels_primary = ax.get_legend_handles_labels()
    handles = handles_primary
    labels = labels_primary

    if secondary_axis is not None:
        handles_secondary, labels_secondary = secondary_axis.get_legend_handles_labels()
        handles += handles_secondary
        labels += labels_secondary

    if show_legend and handles and any(labels):
        legend_title = group_label if group_col and group_label else None
        resolved_legend_location = choose_legend_location_from_axes(ax) if legend_location == "auto" else legend_location
        legend_kwargs = {
            "loc": resolved_legend_location,
            "fontsize": legend_font_size,
            "frameon": legend_frame
        }

        if legend_title:
            legend_kwargs["title"] = legend_title

        legend = ax.legend(handles, labels, **legend_kwargs)

        if legend is not None and legend.get_title() is not None:
            legend.get_title().set_fontsize(legend_font_size)

    if bottom_annotation:
        fig.text(
            0.5,
            0.03,
            bottom_annotation,
            ha="center",
            va="bottom",
            fontsize=legend_font_size
        )
        plt.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        plt.tight_layout()

    if bottom_margin is not None:
        fig.subplots_adjust(bottom=bottom_margin)

    output_name = f"plot_{uuid.uuid4().hex[:12]}.png"
    output_path = PLOT_DIR / output_name

    fig.savefig(output_path)
    plt.close(fig)

    return url_for("static", filename=f"generated_plots/{output_name}")



@app.route("/api/ai/workflow-plan", methods=["POST"])
def ai_workflow_plan():
    try:
        if request.is_json:
            payload = request.get_json() or {}
            user_request = (payload.get("workflow_prompt") or payload.get("message") or "").strip()
            saved_files = []
            file_summaries = []
            dataset_ids = parse_dataset_ids_from_request(payload)
        else:
            payload = {}
            user_request = (request.form.get("workflow_prompt") or request.form.get("message") or "").strip()
            files = request.files.getlist("workflow_files") or request.files.getlist("files")
            saved_files, file_summaries = save_ai_workflow_uploads(files)
            dataset_ids = parse_dataset_ids_from_request(payload)

        if not user_request:
            return jsonify({
                "ok": False,
                "error": "Missing workflow prompt."
            }), 400

        selected_dataset_summaries = build_selected_dataset_summaries(dataset_ids)

        if not selected_dataset_summaries:
            selected_dataset_summaries = [
                build_dataset_summary_for_ai(dataset)
                for dataset in get_datasets()
            ]

        plan = create_workflow_plan(
            user_request=user_request,
            file_summaries=file_summaries,
            current_datasets=selected_dataset_summaries
        )

        plan_text = describe_workflow_plan(plan)
        mapping_preview = extract_workflow_mapping_preview(plan)

        return jsonify({
            "ok": True,
            "plan": plan,
            "plan_text": plan_text,
            "mapping_preview": mapping_preview,
            "uploaded_files": saved_files
        })

    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 500


@app.route("/api/ai/workflow-execute", methods=["POST"])
def ai_workflow_execute():
    try:
        payload = request.get_json() or {}
        plan = payload.get("plan")
        uploaded_files = payload.get("uploaded_files", [])
        approved = payload.get("approved", False)

        if not approved:
            return jsonify({
                "ok": False,
                "error": "Workflow execution requires user approval."
            }), 400

        if not isinstance(plan, dict):
            return jsonify({
                "ok": False,
                "error": "Missing workflow plan."
            }), 400

        result = execute_workflow_plan(
            plan=plan,
            context={
                "uploaded_files": uploaded_files,
                "processed_data_dir": PROCESSED_DATA_DIR,
                "get_dataset": get_dataset,
                "register_dataset": register_dataset,
                "create_plot": create_plot
            }
        )

        return jsonify({
            "ok": True,
            **result
        })

    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 500

@app.route("/api/ai/plot-config", methods=["POST"])
def ai_plot_config():
    payload = request.get_json() or {}

    user_request = payload.get("message", "").strip()
    column_names = payload.get("columns", [])

    if not user_request:
        return jsonify({
            "ok": False,
            "error": "Missing plot request."
        }), 400

    if not column_names:
        return jsonify({
            "ok": False,
            "error": "No column names provided."
        }), 400

    try:
        config = parse_plot_request(user_request, column_names)
        return jsonify({
            "ok": True,
            "config": config
        })
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 500

@app.route("/", methods=["GET"])
def index():
    datasets = get_datasets()

    return render_template(
        "index.html",
        datasets=datasets,
        plot_url=None,
        selected_dataset_id=None,
        form_state={}
    )


@app.route("/upload", methods=["POST"])
def upload_dataset():
    file = request.files.get("dataset_file")

    if file is None or file.filename == "":
        flash("Please select a data file.")
        return redirect(url_for("index"))

    if not is_supported_file(file.filename):
        flash("Only .csv, .dat, .dta, and .txt files are supported for now.")
        return redirect(url_for("index"))

    safe_name = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = UPLOADED_DATA_DIR / f"{timestamp}_{safe_name}"

    file.save(raw_path)

    cleaned_name = f"{timestamp}_{Path(safe_name).stem}_cleaned.csv"
    cleaned_path = PROCESSED_DATA_DIR / cleaned_name

    try:
        save_cleaned_dataset(raw_path, cleaned_path)
    except Exception as error:
        flash(f"File uploaded, but cleaning failed: {error}")
        return redirect(url_for("index"))

    dataset_name = request.form.get("dataset_name", "").strip()
    description = dataset_name if dataset_name else "User-uploaded and cleaned dataset."

    register_dataset(
        file_path=cleaned_path,
        dataset_type="processed_data",
        source=f"Cleaned from {safe_name}",
        uploaded_by="user",
        description=description
    )

    flash("Dataset uploaded and cleaned successfully.")
    return redirect(url_for("index"))


@app.route("/upload_batch", methods=["POST"])
def upload_batch_dataset():
    files = request.files.getlist("dataset_files")
    files = [file for file in files if file and file.filename]

    if not files:
        flash("Please select at least one data file.")
        return redirect(url_for("index"))

    for file in files:
        if not is_supported_file(file.filename):
            flash("Only .csv, .dat, .dta, and .txt files are supported for batch upload.")
            return redirect(url_for("index"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_paths = []

    for index, file in enumerate(files, start=1):
        safe_name = secure_filename(file.filename)
        raw_path = UPLOADED_DATA_DIR / f"{timestamp}_{index:03d}_{safe_name}"
        file.save(raw_path)
        raw_paths.append(raw_path)

    batch_name = request.form.get("batch_name", "").strip()
    condition_label = request.form.get("batch_condition", "").strip()

    output_stem = safe_output_name(batch_name, f"batch_combined_{timestamp}")
    output_path = PROCESSED_DATA_DIR / f"{timestamp}_{output_stem}_combined.csv"

    condition_labels = None

    if condition_label:
        condition_labels = [condition_label] * len(raw_paths)

    try:
        combine_file_paths(
            file_paths=raw_paths,
            output_path=output_path,
            condition_labels=condition_labels
        )
    except Exception as error:
        flash(f"Batch upload saved raw files, but combining failed: {error}")
        return redirect(url_for("index"))

    description = batch_name if batch_name else "Batch uploaded and combined dataset."

    register_dataset(
        file_path=output_path,
        dataset_type="combined_data",
        source=f"Batch combined upload: {len(raw_paths)} files",
        uploaded_by="user",
        description=description
    )

    flash("Batch files uploaded, cleaned, and combined successfully.")
    return redirect(url_for("index"))


@app.route("/upload_sequence", methods=["POST"])
def upload_sequence_dataset():
    files = request.files.getlist("sequence_files")
    files = [file for file in files if file and file.filename]

    if not files:
        flash("Please select at least one sequential data file.")
        return redirect(url_for("index"))

    for file in files:
        if not is_supported_file(file.filename):
            flash("Only .csv, .dat, .dta, and .txt files are supported for sequential upload.")
            return redirect(url_for("index"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_paths = []

    for index, file in enumerate(files, start=1):
        safe_name = secure_filename(file.filename)
        raw_path = UPLOADED_DATA_DIR / f"{timestamp}_sequence_{index:03d}_{safe_name}"
        file.save(raw_path)
        raw_paths.append(raw_path)

    sequence_name = request.form.get("sequence_name", "").strip()
    sequence_condition = request.form.get("sequence_condition", "").strip()
    sequence_time_col = request.form.get("sequence_time_col", "T_s").strip()
    sequence_step_variable_col = request.form.get("sequence_step_variable_col", "Vf_V_vs_Ref").strip()
    sequence_duration_s = request.form.get("sequence_duration_s", "300").strip()
    sequence_regex = request.form.get("sequence_regex", r"#\s*(\d+)").strip()

    if not sequence_condition:
        sequence_condition = sequence_name if sequence_name else "Sequential condition"

    output_stem = safe_output_name(sequence_name, f"sequential_combined_{timestamp}")
    output_path = PROCESSED_DATA_DIR / f"{timestamp}_{output_stem}_sequential.csv"

    try:
        combine_sequential_file_paths(
            file_paths=raw_paths,
            output_path=output_path,
            condition_label=sequence_condition,
            dataset_label=sequence_name if sequence_name else sequence_condition,
            time_col=sequence_time_col,
            step_variable_col=sequence_step_variable_col,
            step_duration_s=sequence_duration_s,
            sequence_regex=sequence_regex
        )
    except Exception as error:
        flash(f"Sequential files were uploaded, but stitching failed: {error}")
        return redirect(url_for("index"))

    description = sequence_name if sequence_name else "Sequential files stitched into one dataset."

    register_dataset(
        file_path=output_path,
        dataset_type="sequential_data",
        source=f"Sequential stitched upload: {len(raw_paths)} files",
        uploaded_by="user",
        description=description
    )

    flash("Sequential files uploaded and stitched successfully.")
    return redirect(url_for("index"))


@app.route("/convert_variables", methods=["POST"])
def convert_variables_dataset():
    dataset_id = request.form.get("conversion_dataset_id")
    dataset = get_dataset(dataset_id)

    if dataset is None:
        flash("Please select a valid dataset for conversion.")
        return redirect(url_for("index"))

    convert_potential = request.form.get("convert_potential") == "on"
    convert_current = request.form.get("convert_current") == "on"

    if not convert_potential and not convert_current:
        flash("Please select at least one conversion.")
        return redirect(url_for("index"))

    converted_name = request.form.get("converted_name", "").strip()

    potential_col = request.form.get("potential_col", "step_variable_value").strip()
    reference_offset_v = request.form.get("reference_offset_v", "0").strip()
    ph_value = request.form.get("ph_value", "0").strip()
    potential_output_col = request.form.get("potential_output_col", "E_RHE").strip()

    current_col = request.form.get("current_col", "Im_A").strip()
    electrode_area_cm2 = request.form.get("electrode_area_cm2", "1").strip()
    current_density_output_col = request.form.get("current_density_output_col", "j_mA_cm2").strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_stem = safe_output_name(converted_name, f"converted_{timestamp}")
    output_path = PROCESSED_DATA_DIR / f"{timestamp}_{output_stem}_converted.csv"

    try:
        convert_dataset_variables(
            input_path=Path(dataset["file_path"]),
            output_path=output_path,
            convert_potential=convert_potential,
            potential_col=potential_col,
            reference_offset_v=reference_offset_v,
            ph_value=ph_value,
            potential_output_col=potential_output_col,
            convert_current=convert_current,
            current_col=current_col,
            electrode_area_cm2=electrode_area_cm2,
            current_density_output_col=current_density_output_col
        )
    except Exception as error:
        flash(f"Variable conversion failed: {error}")
        return redirect(url_for("index"))

    description = converted_name if converted_name else "Dataset with converted electrochemical variables."

    register_dataset(
        file_path=output_path,
        dataset_type="converted_data",
        source=f"Converted from {dataset.get('file_name', 'dataset')}",
        uploaded_by="user",
        description=description
    )

    flash("Variable conversion completed successfully.")
    return redirect(url_for("index"))


@app.route("/combine_existing", methods=["POST"])
def combine_existing_datasets():
    dataset_ids = request.form.getlist("combine_dataset_ids")

    if len(dataset_ids) < 2:
        flash("Please select at least two datasets to combine.")
        return redirect(url_for("index"))

    selected_datasets = []
    file_paths = []
    condition_labels = []
    dataset_labels = []

    for dataset_id in dataset_ids:
        dataset = get_dataset(dataset_id)

        if dataset is None:
            continue

        condition = request.form.get(f"condition_{dataset_id}", "").strip()
        dataset_label = request.form.get(f"dataset_label_{dataset_id}", "").strip()

        if not condition:
            condition = dataset.get("description") or Path(dataset["file_path"]).stem

        if not dataset_label:
            dataset_label = dataset.get("file_name") or Path(dataset["file_path"]).stem

        selected_datasets.append(dataset)
        file_paths.append(Path(dataset["file_path"]))
        condition_labels.append(condition)
        dataset_labels.append(dataset_label)

    if len(file_paths) < 2:
        flash("At least two valid datasets are required for combining.")
        return redirect(url_for("index"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_name = request.form.get("combined_existing_name", "").strip()
    output_stem = safe_output_name(combined_name, f"selected_combined_{timestamp}")
    output_path = PROCESSED_DATA_DIR / f"{timestamp}_{output_stem}_combined.csv"

    try:
        combine_file_paths(
            file_paths=file_paths,
            output_path=output_path,
            condition_labels=condition_labels,
            dataset_labels=dataset_labels
        )
    except Exception as error:
        flash(f"Combining selected datasets failed: {error}")
        return redirect(url_for("index"))

    description = combined_name if combined_name else "Combined from selected existing datasets."

    register_dataset(
        file_path=output_path,
        dataset_type="combined_data",
        source=f"Combined from {len(selected_datasets)} existing datasets",
        uploaded_by="user",
        description=description
    )

    flash("Selected datasets combined successfully.")
    return redirect(url_for("index"))


@app.route("/average_replicates", methods=["POST"], endpoint="average_replicates")
@app.route("/average_replicates_dataset", methods=["POST"])
def average_replicates_dataset():
    dataset_id = request.form.get("average_dataset_id")
    dataset = get_dataset(dataset_id)

    if dataset is None:
        flash("Please select a valid combined dataset for replicate averaging.")
        return redirect(url_for("index"))

    averaged_name = request.form.get("averaged_name", "").strip()
    x_col = request.form.get("average_x_col", "").strip()
    y_col = request.form.get("average_y_col", "").strip()
    condition_col = request.form.get("average_condition_col", "condition").strip() or "condition"
    replicate_col = request.form.get("average_replicate_col", "dataset_label").strip() or "dataset_label"
    method = request.form.get("average_method", "interpolate").strip() or "interpolate"
    x_grid_method = request.form.get("average_x_grid_method", "overlap").strip() or "overlap"
    grid_points = request.form.get("average_grid_points", "500").strip() or "500"
    x_round_decimals = request.form.get("average_x_round_decimals", "6").strip() or "6"
    min_replicates = request.form.get("average_min_replicates", "2").strip() or "2"

    if not x_col or x_col == "none":
        flash("Please select an X column for replicate averaging.")
        return redirect(url_for("index"))

    if not y_col or y_col == "none":
        flash("Please select a Y column for replicate averaging.")
        return redirect(url_for("index"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_stem = safe_output_name(averaged_name, f"averaged_replicates_{timestamp}")
    output_path = PROCESSED_DATA_DIR / f"{timestamp}_{output_stem}_averaged.csv"

    try:
        result = average_condition_replicates(
            input_path=Path(dataset["file_path"]),
            output_path=output_path,
            x_col=x_col,
            y_col=y_col,
            condition_col=condition_col,
            replicate_col=replicate_col,
            method=method,
            x_grid_method=x_grid_method,
            grid_points=grid_points,
            x_round_decimals=x_round_decimals,
            min_replicates=min_replicates
        )
    except Exception as error:
        flash(f"Replicate averaging failed: {error}")
        return redirect(url_for("index"))

    description = averaged_name if averaged_name else f"Averaged replicate curves from {dataset.get('file_name', 'dataset')}."

    register_dataset(
        file_path=output_path,
        dataset_type="averaged_replicates",
        source=f"Averaged replicates from {dataset.get('file_name', 'dataset')}",
        uploaded_by="user",
        description=description
    )

    condition_count = len(result.get("conditions", []))
    row_count = result.get("rows", 0)
    flash(f"Replicate averaging completed: {condition_count} condition(s), {row_count} averaged rows. Plot y_mean grouped by {condition_col}.")
    return redirect(url_for("index"))


@app.route("/dataset/<dataset_id>/columns", methods=["GET"])
def dataset_columns(dataset_id):
    dataset = get_dataset(dataset_id)

    if dataset is None:
        return jsonify({"error": "Dataset not found."}), 404

    column_types = get_column_types(dataset)

    return jsonify({
        "dataset_id": dataset["dataset_id"],
        "file_name": dataset["file_name"],
        "rows": dataset["rows"],
        "columns": dataset["columns"],
        "column_names": dataset["column_names"],
        "numeric_columns": dataset["numeric_columns"],
        "categorical_columns": dataset["categorical_columns"],
        "column_types": column_types,
        "description": dataset.get("description", ""),
        "source": dataset.get("source", ""),
        "dataset_type": dataset.get("dataset_type", "")
    })


@app.route("/suggest_plot_types", methods=["POST"])
def suggest_plot_types_route():
    data = request.get_json()
    dataset_id = data.get("dataset_id")
    x_col = data.get("x_col")
    y_col = data.get("y_col")

    dataset = get_dataset(dataset_id)

    if dataset is None:
        return jsonify({"error": "Dataset not found."}), 404

    column_types = get_column_types(dataset)

    x_type = column_types.get(x_col)
    y_type = "none" if y_col in [None, "", "none"] else column_types.get(y_col)

    plot_types = suggest_plot_types(x_type, y_type)

    return jsonify({
        "x_type": x_type,
        "y_type": y_type,
        "plot_types": plot_types
    })


@app.route("/plot", methods=["POST"])
def plot():
    datasets = get_datasets()

    dataset_id = request.form.get("dataset_id")
    x_col = request.form.get("x_col")
    y_col = request.form.get("y_col")
    x_label = request.form.get("x_label", "").strip()
    y_label = request.form.get("y_label", "").strip()
    plot_type = request.form.get("plot_type")
    plot_title = request.form.get("plot_title", "").strip()
    bottom_annotation = request.form.get("bottom_annotation", "").strip()
    marker_color = request.form.get("marker_color", "#FF5F05")
    line_color = request.form.get("line_color", "#13294B")
    primary_label = request.form.get("primary_label", "").strip()

    group_col = request.form.get("group_col")
    group_label = request.form.get("group_label", "").strip()
    group_color_mode = request.form.get("group_color_mode", "same")

    data_reduction = request.form.get("data_reduction", "raw")
    summary_group_col = request.form.get("summary_group_col")
    x_summary_method = request.form.get("x_summary_method", "mean")
    y_summary_method = request.form.get("y_summary_method", "mean_tail")
    tail_fraction = request.form.get("tail_fraction", 0.2)
    fit_guide = request.form.get("fit_guide", "connect")
    smooth_window = request.form.get("smooth_window", 5)

    use_step_axis = request.form.get("use_step_axis") == "on"
    step_axis_mode = request.form.get("step_axis_mode", "auto_data")
    step_axis_placement = request.form.get("step_axis_placement", "uniform")
    step_axis_value_col = request.form.get("step_axis_value_col")
    step_axis_group_col = request.form.get("step_axis_group_col")
    step_axis_label = request.form.get("step_axis_label", "").strip()
    step_axis_max_ticks = request.form.get("step_axis_max_ticks", 12)
    step_axis_decimal_places = request.form.get("step_axis_decimal_places", 1)
    step_axis_label_stride = request.form.get("step_axis_label_stride", 2)
    step_axis_custom_labels = request.form.get("step_axis_custom_labels", "").strip()
    step_axis_custom_positions = request.form.get("step_axis_custom_positions", "").strip()
    step_axis_label_rotation = request.form.get("step_axis_label_rotation", 0)
    step_axis_label_pad = request.form.get("step_axis_label_pad", 14)
    bottom_margin = request.form.get("bottom_margin", "")

    if use_step_axis and step_axis_custom_labels:
        step_axis_mode = "uniform_custom"

    if use_step_axis and step_axis_custom_positions:
        step_axis_placement = "custom_positions"

    if step_axis_group_col in [None, "", "none"]:
        step_axis_group_col = request.form.get("step_axis_sequence_col", "")

    line_order = request.form.get("line_order", "original")
    show_markers = request.form.get("show_markers") == "on"
    show_legend = request.form.get("show_legend") == "on"
    marker_size = request.form.get("marker_size", 18)
    line_width = request.form.get("line_width", 2.2)
    opacity = request.form.get("opacity", 1.0)

    axis_label_size = request.form.get("axis_label_size", 18)
    tick_label_size = request.form.get("tick_label_size", 13)
    axis_label_weight = request.form.get("axis_label_weight", "bold")
    spine_width = request.form.get("spine_width", 1.2)
    tick_width = request.form.get("tick_width", 1.2)
    tick_length = request.form.get("tick_length", 5)
    show_full_frame = request.form.get("show_full_frame") == "on"
    tick_direction = request.form.get("tick_direction", "in")
    show_top_ticks = request.form.get("show_top_ticks") == "on"
    show_bottom_ticks = request.form.get("show_bottom_ticks") == "on"
    show_left_ticks = request.form.get("show_left_ticks") == "on"
    show_right_ticks = request.form.get("show_right_ticks") == "on"
    show_grid = request.form.get("show_grid") == "on"
    grid_axis = request.form.get("grid_axis", "both")
    grid_which = request.form.get("grid_which", "major")
    major_grid_linestyle = request.form.get("major_grid_linestyle", "solid")
    minor_grid_linestyle = request.form.get("minor_grid_linestyle", "dashed")
    major_grid_alpha = request.form.get("major_grid_alpha", 0.25)
    minor_grid_alpha = request.form.get("minor_grid_alpha", 0.12)
    major_grid_width = request.form.get("major_grid_width", 0.7)
    minor_grid_width = request.form.get("minor_grid_width", 0.5)
    title_size = request.form.get("title_size", 18)
    legend_font_size = request.form.get("legend_font_size", 11)
    legend_location = request.form.get("legend_location", "best")
    legend_frame = request.form.get("legend_frame") == "on"

    figure_width = request.form.get("figure_width", 8)
    figure_height = request.form.get("figure_height", 5)
    figure_dpi = request.form.get("figure_dpi", 150)

    x_min = request.form.get("x_min", "")
    x_max = request.form.get("x_max", "")
    x_tick_mode = request.form.get("x_tick_mode", "auto")
    x_major_interval = request.form.get("x_major_interval", "")
    x_minor_interval = request.form.get("x_minor_interval", "")
    x_custom_ticks = request.form.get("x_custom_ticks", "").strip()
    x_custom_tick_labels = request.form.get("x_custom_tick_labels", "").strip()

    y_min = request.form.get("y_min", "")
    y_max = request.form.get("y_max", "")
    y_tick_mode = request.form.get("y_tick_mode", "auto")
    y_major_interval = request.form.get("y_major_interval", "")
    y_minor_interval = request.form.get("y_minor_interval", "")
    y_custom_ticks = request.form.get("y_custom_ticks", "").strip()
    y_custom_tick_labels = request.form.get("y_custom_tick_labels", "").strip()

    secondary_mode = request.form.get("secondary_mode", "none")
    x2_col = request.form.get("x2_col")
    y2_col = request.form.get("y2_col")
    top_x_label = request.form.get("top_x_label", "").strip()
    right_y_label = request.form.get("right_y_label", "").strip()
    secondary_plot_type = request.form.get("secondary_plot_type", "line")
    second_marker_color = request.form.get("second_marker_color", "#9A4DFF")
    second_line_color = request.form.get("second_line_color", "#9A4DFF")
    second_label = request.form.get("second_label", "").strip()

    x_label = normalize_matplotlib_text(x_label)
    y_label = normalize_matplotlib_text(y_label)
    plot_title = normalize_matplotlib_text(plot_title)
    bottom_annotation = normalize_matplotlib_text(bottom_annotation)
    primary_label = normalize_matplotlib_text(primary_label)
    group_label = normalize_matplotlib_text(group_label)
    step_axis_label = normalize_matplotlib_text(step_axis_label)
    top_x_label = normalize_matplotlib_text(top_x_label)
    right_y_label = normalize_matplotlib_text(right_y_label)
    second_label = normalize_matplotlib_text(second_label)

    if group_col not in [None, "", "none"]:
        secondary_mode = "none"

    if use_step_axis:
        secondary_mode = "none"

    if data_reduction == "summary":
        secondary_mode = "none"
        group_col = "none"

    form_state = {
        "dataset_id": dataset_id,
        "x_col": x_col,
        "y_col": y_col,
        "x_label": x_label,
        "y_label": y_label,
        "plot_type": plot_type,
        "plot_title": plot_title,
        "bottom_annotation": bottom_annotation,
        "marker_color": marker_color,
        "line_color": line_color,
        "primary_label": primary_label,
        "group_col": group_col,
        "group_label": group_label,
        "group_color_mode": group_color_mode,
        "data_reduction": data_reduction,
        "summary_group_col": summary_group_col,
        "x_summary_method": x_summary_method,
        "y_summary_method": y_summary_method,
        "tail_fraction": tail_fraction,
        "fit_guide": fit_guide,
        "smooth_window": smooth_window,
        "use_step_axis": use_step_axis,
        "step_axis_mode": step_axis_mode,
        "step_axis_placement": step_axis_placement,
        "step_axis_value_col": step_axis_value_col,
        "step_axis_group_col": step_axis_group_col,
        "step_axis_sequence_col": step_axis_group_col,
        "step_axis_label": step_axis_label,
        "step_axis_max_ticks": step_axis_max_ticks,
        "step_axis_decimal_places": step_axis_decimal_places,
        "step_axis_label_stride": step_axis_label_stride,
        "step_axis_custom_labels": step_axis_custom_labels,
        "step_axis_custom_positions": step_axis_custom_positions,
        "step_axis_label_rotation": step_axis_label_rotation,
        "step_axis_label_pad": step_axis_label_pad,
        "bottom_margin": bottom_margin,
        "line_order": line_order,
        "show_markers": show_markers,
        "show_legend": show_legend,
        "marker_size": marker_size,
        "line_width": line_width,
        "opacity": opacity,
        "axis_label_size": axis_label_size,
        "tick_label_size": tick_label_size,
        "axis_label_weight": axis_label_weight,
        "spine_width": spine_width,
        "tick_width": tick_width,
        "tick_length": tick_length,
        "show_full_frame": show_full_frame,
        "tick_direction": tick_direction,
        "show_top_ticks": show_top_ticks,
        "show_bottom_ticks": show_bottom_ticks,
        "show_left_ticks": show_left_ticks,
        "show_right_ticks": show_right_ticks,
        "show_grid": show_grid,
        "grid_axis": grid_axis,
        "grid_which": grid_which,
        "major_grid_linestyle": major_grid_linestyle,
        "minor_grid_linestyle": minor_grid_linestyle,
        "major_grid_alpha": major_grid_alpha,
        "minor_grid_alpha": minor_grid_alpha,
        "major_grid_width": major_grid_width,
        "minor_grid_width": minor_grid_width,
        "title_size": title_size,
        "legend_font_size": legend_font_size,
        "legend_location": legend_location,
        "legend_frame": legend_frame,
        "figure_width": figure_width,
        "figure_height": figure_height,
        "figure_dpi": figure_dpi,
        "x_min": x_min,
        "x_max": x_max,
        "x_tick_mode": x_tick_mode,
        "x_major_interval": x_major_interval,
        "x_minor_interval": x_minor_interval,
        "x_custom_ticks": x_custom_ticks,
        "x_custom_tick_labels": x_custom_tick_labels,
        "y_min": y_min,
        "y_max": y_max,
        "y_tick_mode": y_tick_mode,
        "y_major_interval": y_major_interval,
        "y_minor_interval": y_minor_interval,
        "y_custom_ticks": y_custom_ticks,
        "y_custom_tick_labels": y_custom_tick_labels,
        "secondary_mode": secondary_mode,
        "x2_col": x2_col,
        "y2_col": y2_col,
        "top_x_label": top_x_label,
        "right_y_label": right_y_label,
        "secondary_plot_type": secondary_plot_type,
        "second_marker_color": second_marker_color,
        "second_line_color": second_line_color,
        "second_label": second_label
    }

    try:
        plot_url = create_plot(
            dataset_id=dataset_id,
            x_col=x_col,
            y_col=y_col,
            x_label=x_label,
            y_label=y_label,
            plot_type=plot_type,
            plot_title=plot_title,
            bottom_annotation=bottom_annotation,
            marker_color=marker_color,
            line_color=line_color,
            primary_label=primary_label,
            group_col=group_col,
            group_label=group_label,
            group_color_mode=group_color_mode,
            data_reduction=data_reduction,
            summary_group_col=summary_group_col,
            x_summary_method=x_summary_method,
            y_summary_method=y_summary_method,
            tail_fraction=tail_fraction,
            fit_guide=fit_guide,
            smooth_window=smooth_window,
            use_step_axis=use_step_axis,
            step_axis_mode=step_axis_mode,
            step_axis_placement=step_axis_placement,
            step_axis_value_col=step_axis_value_col,
            step_axis_group_col=step_axis_group_col,
            step_axis_label=step_axis_label,
            step_axis_max_ticks=step_axis_max_ticks,
            secondary_mode=secondary_mode,
            x2_col=x2_col,
            y2_col=y2_col,
            top_x_label=top_x_label,
            right_y_label=right_y_label,
            secondary_plot_type=secondary_plot_type,
            second_marker_color=second_marker_color,
            second_line_color=second_line_color,
            second_label=second_label,
            line_order=line_order,
            show_markers=show_markers,
            show_legend=show_legend,
            marker_size=marker_size,
            line_width=line_width,
            opacity=opacity,
            axis_label_size=axis_label_size,
            tick_label_size=tick_label_size,
            axis_label_weight=axis_label_weight,
            spine_width=spine_width,
            tick_width=tick_width,
            tick_length=tick_length,
            show_full_frame=show_full_frame,
            tick_direction=tick_direction,
            show_top_ticks=show_top_ticks,
            show_bottom_ticks=show_bottom_ticks,
            show_left_ticks=show_left_ticks,
            show_right_ticks=show_right_ticks,
            show_grid=show_grid,
            grid_axis=grid_axis,
            grid_which=grid_which,
            major_grid_linestyle=major_grid_linestyle,
            minor_grid_linestyle=minor_grid_linestyle,
            major_grid_alpha=major_grid_alpha,
            minor_grid_alpha=minor_grid_alpha,
            major_grid_width=major_grid_width,
            minor_grid_width=minor_grid_width,
            title_size=title_size,
            legend_font_size=legend_font_size,
            legend_location=legend_location,
            legend_frame=legend_frame,
            figure_width=figure_width,
            figure_height=figure_height,
            figure_dpi=figure_dpi,
            x_min=x_min,
            x_max=x_max,
            x_tick_mode=x_tick_mode,
            x_major_interval=x_major_interval,
            x_minor_interval=x_minor_interval,
            x_custom_ticks=x_custom_ticks,
            x_custom_tick_labels=x_custom_tick_labels,
            y_min=y_min,
            y_max=y_max,
            y_tick_mode=y_tick_mode,
            y_major_interval=y_major_interval,
            y_minor_interval=y_minor_interval,
            y_custom_ticks=y_custom_ticks,
            y_custom_tick_labels=y_custom_tick_labels,
            step_axis_decimal_places=step_axis_decimal_places,
            step_axis_label_stride=step_axis_label_stride,
            step_axis_custom_labels=step_axis_custom_labels,
            step_axis_custom_positions=step_axis_custom_positions,
            step_axis_label_rotation=step_axis_label_rotation,
            step_axis_label_pad=step_axis_label_pad,
            bottom_margin=bottom_margin
        )
    except Exception as error:
        flash(str(error))
        plot_url = None

    return render_template(
        "index.html",
        datasets=datasets,
        plot_url=plot_url,
        selected_dataset_id=dataset_id,
        form_state=form_state
    )


if __name__ == "__main__":
    create_dirs()
    register_test_data_if_needed()
    app.run(debug=True)