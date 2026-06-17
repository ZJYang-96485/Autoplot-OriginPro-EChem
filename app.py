from pathlib import Path
from datetime import datetime
import json
import uuid

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename

from data_loader import read_dataset, inspect_dataset, save_cleaned_dataset


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
    show_top_x_axis,
    show_right_y_axis,
    top_x_label,
    right_y_label
):
    dataset = get_dataset(dataset_id)

    if dataset is None:
        raise ValueError("Dataset not found.")

    data_path = Path(dataset["file_path"])
    df = read_dataset(data_path)

    if x_col not in df.columns:
        raise ValueError("X column not found.")

    if y_col and y_col != "none" and y_col not in df.columns:
        raise ValueError("Y column not found.")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    if plot_type == "scatter":
        data = df[[x_col, y_col]].dropna()
        ax.scatter(
            data[x_col],
            data[y_col],
            facecolor=marker_color,
            edgecolor=line_color,
            alpha=0.75
        )

    elif plot_type == "line":
        data = df[[x_col, y_col]].dropna().sort_values(by=x_col)
        ax.plot(
            data[x_col],
            data[y_col],
            marker="o",
            color=line_color,
            markerfacecolor=marker_color,
            markeredgecolor=line_color
        )

    elif plot_type == "bar":
        data = df[[x_col, y_col]].dropna()
        grouped = data.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        ax.bar(
            grouped.index.astype(str),
            grouped.values,
            color=marker_color,
            edgecolor=line_color
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

        for item in box["whiskers"] + box["caps"] + box["medians"]:
            item.set_color(line_color)

    elif plot_type == "count":
        data = df[[x_col]].dropna()
        counts = data[x_col].astype(str).value_counts()
        ax.bar(
            counts.index,
            counts.values,
            color=marker_color,
            edgecolor=line_color
        )
        y_label = y_label if y_label else "Count"

    elif plot_type == "histogram":
        data = df[[x_col]].dropna()
        ax.hist(
            data[x_col],
            bins=20,
            facecolor=marker_color,
            edgecolor=line_color
        )
        y_label = y_label if y_label else "Count"

    else:
        raise ValueError("Unsupported plot type.")

    final_x_label = x_label if x_label else x_col

    if plot_type in ["count", "histogram"]:
        final_y_label = y_label if y_label else "Count"
    else:
        final_y_label = y_label if y_label else y_col

    ax.set_xlabel(final_x_label)
    ax.set_ylabel(final_y_label)

    if plot_title:
        ax.set_title(plot_title)

    if show_top_x_axis:
        top_axis = ax.secondary_xaxis("top")
        top_axis.set_xlabel(top_x_label if top_x_label else final_x_label)

    if show_right_y_axis:
        right_axis = ax.secondary_yaxis("right")
        right_axis.set_ylabel(right_y_label if right_y_label else final_y_label)

    ax.grid(True, alpha=0.25)

    plt.xticks(rotation=30, ha="right")

    if bottom_annotation:
        fig.text(
            0.5,
            0.02,
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
    show_top_x_axis = request.form.get("show_top_x_axis") == "on"
    show_right_y_axis = request.form.get("show_right_y_axis") == "on"
    top_x_label = request.form.get("top_x_label", "").strip()
    right_y_label = request.form.get("right_y_label", "").strip()

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
        "show_top_x_axis": show_top_x_axis,
        "show_right_y_axis": show_right_y_axis,
        "top_x_label": top_x_label,
        "right_y_label": right_y_label
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
            show_top_x_axis=show_top_x_axis,
            show_right_y_axis=show_right_y_axis,
            top_x_label=top_x_label,
            right_y_label=right_y_label
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