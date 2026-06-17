from pathlib import Path
from datetime import datetime
import json
import uuid
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename

from data_loader import read_dataset, inspect_dataset, save_cleaned_dataset
from data_combiner import combine_file_paths
from sequential_tools import combine_sequential_file_paths


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


def safe_output_name(value, default_name):
    value = str(value).strip()

    if not value:
        return default_name

    value = secure_filename(value)

    if not value:
        return default_name

    return value


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
        "global_time_min"
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
                if column == "step_variable_value":
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
            label=label if label else "Primary"
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
            label=label if label else "Primary"
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
            label=label if label else "Primary"
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
            label=label if label else "Primary"
        )

    elif plot_type == "histogram":
        data = df[[x_col]].dropna()
        ax.hist(
            data[x_col],
            bins=20,
            facecolor=marker_color,
            edgecolor=line_color,
            alpha=opacity,
            label=label if label else "Primary"
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

    data = df[[x_col, y_col, group_col]].dropna()

    if "sequence_index" in df.columns:
        data = df[[x_col, y_col, group_col, "sequence_index"]].dropna(subset=[x_col, y_col, group_col])

    if "source_index" in df.columns and "source_index" not in data.columns:
        data = data.join(df["source_index"])

    if data.empty:
        raise ValueError("No valid data points were found for the selected group-by plot.")

    for group_name, group in data.groupby(group_col, sort=False):
        label = str(group_name)

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
    data = summary_df[[x_col, y_col]].dropna()

    if data.empty:
        raise ValueError("No summarized data points are available for plotting.")

    if "sequence_index" in summary_df.columns:
        data = summary_df[[x_col, y_col, "sequence_index"]].dropna(subset=[x_col, y_col])

    data = order_data(data, x_col, line_order)

    if fit_guide == "none":
        ax.scatter(
            data[x_col],
            data[y_col],
            facecolor=marker_color,
            edgecolor=line_color,
            s=marker_size,
            alpha=opacity,
            label=label if label else "Summary points"
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
            label=label if label else "Summary"
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
            label=label if label else "Summary points"
        )

        smoothed = smooth_summary_data(data, x_col, y_col, smooth_window)

        ax.plot(
            smoothed[x_col],
            smoothed[y_col],
            color=line_color,
            linewidth=line_width,
            alpha=opacity,
            label="Smooth guide"
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
            label=label if label else "Secondary"
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
            label=label if label else "Secondary"
        )


def setup_step_axis(ax, axis_df, x_col, x_label, step_value_col, sequence_col, step_axis_label, max_ticks):
    if step_value_col in [None, "", "none"]:
        return

    if sequence_col in [None, "", "none"]:
        return

    if step_value_col not in axis_df.columns or sequence_col not in axis_df.columns or x_col not in axis_df.columns:
        return

    data = axis_df[[x_col, step_value_col, sequence_col]].dropna()

    if data.empty:
        return

    ticks = []
    labels = []

    for _, group in data.groupby(sequence_col, sort=True):
        x_value = pd.to_numeric(group[x_col], errors="coerce").mean()
        step_value = pd.to_numeric(group[step_value_col], errors="coerce").mean()

        if pd.notna(x_value) and pd.notna(step_value):
            ticks.append(x_value)
            labels.append(f"{step_value:.3g}")

    if not ticks:
        return

    max_ticks = max(2, max_ticks)
    stride = max(1, int(len(ticks) / max_ticks))

    ticks = ticks[::stride]
    labels = labels[::stride]

    top_axis = ax.secondary_xaxis("top")
    top_axis.set_xlabel(x_label if x_label else x_col)

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel(step_axis_label if step_axis_label else step_value_col)


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
    step_axis_value_col,
    step_axis_sequence_col,
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
    opacity
):
    dataset = get_dataset(dataset_id)

    if dataset is None:
        raise ValueError("Dataset not found.")

    data_path = Path(dataset["file_path"])
    df = read_dataset(data_path)

    group_col = None if group_col in [None, "", "none"] else group_col
    summary_group_col = None if summary_group_col in [None, "", "none"] else summary_group_col

    if secondary_mode not in SECONDARY_MODES:
        raise ValueError("Invalid secondary mode.")

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
    line_width = clamp_float(line_width, 1.2, 0.1, 10)
    opacity = clamp_float(opacity, 0.35, 0.05, 1.0)
    tail_fraction = clamp_float(tail_fraction, 0.2, 0.01, 1.0)
    smooth_window = clamp_int(smooth_window, 5, 3, 99)
    step_axis_max_ticks = clamp_int(step_axis_max_ticks, 12, 2, 64)
    group_color_mode = "auto" if group_color_mode == "auto" else "same"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
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

    final_x_label = x_label if x_label else x_col

    if plot_type in ["count", "histogram"]:
        final_y_label = y_label if y_label else "Count"
    else:
        final_y_label = y_label if y_label else y_col

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

        secondary_axis.set_xlabel(top_x_label if top_x_label else x2_col)

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

        secondary_axis.set_ylabel(right_y_label if right_y_label else y2_col)

    if use_step_axis:
        setup_step_axis(
            ax=ax,
            axis_df=axis_df,
            x_col=x_col,
            x_label=final_x_label,
            step_value_col=step_axis_value_col,
            sequence_col=step_axis_sequence_col,
            step_axis_label=step_axis_label,
            max_ticks=step_axis_max_ticks
        )

    if plot_title:
        ax.set_title(plot_title)

    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", rotation=30)

    handles_primary, labels_primary = ax.get_legend_handles_labels()
    handles = handles_primary
    labels = labels_primary

    if secondary_axis is not None:
        handles_secondary, labels_secondary = secondary_axis.get_legend_handles_labels()
        handles += handles_secondary
        labels += labels_secondary

    if show_legend and handles and any(labels):
        legend_title = group_label if group_col and group_label else group_col
        ax.legend(handles, labels, loc="best", title=legend_title)

    if bottom_annotation:
        fig.text(
            0.5,
            0.03,
            bottom_annotation,
            ha="center",
            va="bottom",
            fontsize=9
        )
        plt.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        plt.tight_layout()

    output_name = f"plot_{uuid.uuid4().hex[:12]}.png"
    output_path = PLOT_DIR / output_name

    fig.savefig(output_path)
    plt.close(fig)

    return url_for("static", filename=f"generated_plots/{output_name}")


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
    step_axis_value_col = request.form.get("step_axis_value_col")
    step_axis_sequence_col = request.form.get("step_axis_sequence_col", "sequence_index")
    step_axis_label = request.form.get("step_axis_label", "").strip()
    step_axis_max_ticks = request.form.get("step_axis_max_ticks", 12)

    line_order = request.form.get("line_order", "original")
    show_markers = request.form.get("show_markers") == "on"
    show_legend = request.form.get("show_legend") == "on"
    marker_size = request.form.get("marker_size", 18)
    line_width = request.form.get("line_width", 1.2)
    opacity = request.form.get("opacity", 0.35)

    secondary_mode = request.form.get("secondary_mode", "none")
    x2_col = request.form.get("x2_col")
    y2_col = request.form.get("y2_col")
    top_x_label = request.form.get("top_x_label", "").strip()
    right_y_label = request.form.get("right_y_label", "").strip()
    secondary_plot_type = request.form.get("secondary_plot_type", "line")
    second_marker_color = request.form.get("second_marker_color", "#9A4DFF")
    second_line_color = request.form.get("second_line_color", "#9A4DFF")
    second_label = request.form.get("second_label", "").strip()

    if group_col not in [None, "", "none"]:
        secondary_mode = "none"

    if use_step_axis:
        secondary_mode = "none"

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
        "step_axis_value_col": step_axis_value_col,
        "step_axis_sequence_col": step_axis_sequence_col,
        "step_axis_label": step_axis_label,
        "step_axis_max_ticks": step_axis_max_ticks,
        "line_order": line_order,
        "show_markers": show_markers,
        "show_legend": show_legend,
        "marker_size": marker_size,
        "line_width": line_width,
        "opacity": opacity,
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
            step_axis_value_col=step_axis_value_col,
            step_axis_sequence_col=step_axis_sequence_col,
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
            opacity=opacity
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