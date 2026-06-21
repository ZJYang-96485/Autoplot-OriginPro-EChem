from pathlib import Path
from datetime import datetime
import re
import pandas as pd

from data_loader import save_cleaned_dataset
from data_combiner import combine_file_paths
from sequential_tools import combine_sequential_file_paths
from conversion_tools import convert_dataset_variables
from replicate_tools import average_condition_replicates


def _text(value):
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in {"none", "null", "nan", "na", "n/a"}:
        return ""

    return value


def _bool(value, default=False):
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    value = str(value).strip().lower()

    if value in {"true", "yes", "y", "1", "on"}:
        return True

    if value in {"false", "no", "n", "0", "off"}:
        return False

    return default


def _int(value, default):
    try:
        if value in [None, ""]:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value, default):
    try:
        if value in [None, ""]:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_name(value, default):
    value = _text(value) or default
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or default


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _find_time_column(columns, requested_time_col=""):
    requested_time_col = _text(requested_time_col)

    if requested_time_col and requested_time_col in columns:
        return requested_time_col

    candidates = [
        "global_time_min",
        "T_s",
        "Time_s",
        "time_s",
        "t_s",
        "T",
        "Time",
        "time",
        "Elapsed Time",
        "Elapsed_Time"
    ]

    for candidate in candidates:
        if candidate in columns:
            return candidate

    lower_lookup = {str(col).strip().lower(): col for col in columns}

    for candidate in candidates:
        key = candidate.lower()

        if key in lower_lookup:
            return lower_lookup[key]

    for col in columns:
        lower_col = str(col).strip().lower()

        if "time" in lower_col or lower_col in {"t", "ts"}:
            return col

    return ""


def _ensure_global_time_min(csv_path, requested_time_col=""):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return False

    df = pd.read_csv(csv_path)

    if "global_time_min" in df.columns:
        return True

    time_col = _find_time_column(df.columns, requested_time_col)

    if not time_col:
        return False

    time_values = pd.to_numeric(df[time_col], errors="coerce")
    lower_time_col = str(time_col).strip().lower()

    if "min" in lower_time_col:
        df["global_time_min"] = time_values
    else:
        df["global_time_min"] = time_values / 60.0

    df.to_csv(csv_path, index=False)

    return True



def _find_numeric_column(columns, requested_col, candidates, contains_terms):
    requested_col = _text(requested_col)

    if requested_col and requested_col in columns:
        return requested_col

    lower_lookup = {str(col).strip().lower(): col for col in columns}

    if requested_col and requested_col.lower() in lower_lookup:
        return lower_lookup[requested_col.lower()]

    for candidate in candidates:
        if candidate in columns:
            return candidate

        lower_candidate = candidate.lower()

        if lower_candidate in lower_lookup:
            return lower_lookup[lower_candidate]

    for col in columns:
        lower_col = str(col).strip().lower()

        if any(term in lower_col for term in contains_terms):
            return col

    return ""


def _find_current_column(columns, requested_col=""):
    return _find_numeric_column(
        columns,
        requested_col,
        [
            "Im_A",
            "Im",
            "I_A",
            "I",
            "Current_A",
            "Current",
            "current",
            "i",
            "I/mA",
            "Current_mA"
        ],
        ["current", "im", "i/a", "i_ma", "ma"]
    )


def _find_potential_column(columns, requested_col=""):
    return _find_numeric_column(
        columns,
        requested_col,
        [
            "Vf_V_vs_Ref",
            "Vf",
            "Ewe/V",
            "Ewe",
            "Potential_V",
            "Potential",
            "E_V",
            "E",
            "Voltage",
            "V"
        ],
        ["potential", "vf", "ewe", "voltage", "e/v"]
    )


def _read_csv_relaxed(path):
    path = Path(path)

    try:
        return pd.read_csv(path)
    except Exception:
        pass

    for sep in ["\t", ";", r"\s+"]:
        try:
            return pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            continue

    raise ValueError(f"Could not read dataset file for conversion: {path}")


def _create_converted_dataset(
    input_path,
    output_path,
    convert_potential,
    potential_col,
    potential_output_col,
    reference_offset_v,
    ph_value,
    convert_current,
    current_col,
    current_density_output_col,
    electrode_area_cm2,
    time_col
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _read_csv_relaxed(input_path)

    if df.empty:
        raise ValueError(f"Input dataset is empty: {input_path}")

    columns = list(df.columns)

    resolved_time_col = _find_time_column(columns, time_col)

    if not resolved_time_col:
        raise ValueError(
            "Time conversion failed because no time column was found. "
            f"Available columns: {', '.join(map(str, columns))}"
        )

    time_values = pd.to_numeric(df[resolved_time_col], errors="coerce")
    lower_time_col = str(resolved_time_col).strip().lower()

    if "min" in lower_time_col:
        df["global_time_min"] = time_values
    else:
        df["global_time_min"] = time_values / 60.0

    if convert_potential:
        resolved_potential_col = _find_potential_column(columns, potential_col)

        if not resolved_potential_col:
            raise ValueError(
                "Potential conversion failed because no potential column was found. "
                f"Available columns: {', '.join(map(str, columns))}"
            )

        potential_values = pd.to_numeric(df[resolved_potential_col], errors="coerce")
        df[potential_output_col or "E_RHE"] = potential_values + reference_offset_v + 0.0591 * ph_value

    if convert_current:
        resolved_current_col = _find_current_column(columns, current_col)

        if not resolved_current_col:
            raise ValueError(
                "Current conversion failed because no current column was found. "
                f"Available columns: {', '.join(map(str, columns))}"
            )

        if electrode_area_cm2 <= 0:
            raise ValueError("Current-density conversion requires electrode_area_cm2 > 0.")

        current_values = pd.to_numeric(df[resolved_current_col], errors="coerce")
        lower_current_col = str(resolved_current_col).strip().lower()

        if "ma" in lower_current_col and "/a" not in lower_current_col:
            df[current_density_output_col or "j_mA_cm2"] = current_values / electrode_area_cm2
        else:
            df[current_density_output_col or "j_mA_cm2"] = current_values * 1000.0 / electrode_area_cm2

    df.to_csv(output_path, index=False)

    if not output_path.exists():
        raise ValueError(f"Conversion did not create output file: {output_path}")

    return {
        "time_col": resolved_time_col,
        "potential_col": _find_potential_column(columns, potential_col) if convert_potential else "",
        "current_col": _find_current_column(columns, current_col) if convert_current else "",
        "output_path": str(output_path)
    }


def _params(step):
    params = step.get("parameters") or {}

    if not isinstance(params, dict):
        return {}

    return params


def _output_path(processed_dir, output_name, suffix):
    stem = _safe_name(output_name, suffix)
    return Path(processed_dir) / f"{_timestamp()}_{stem}_{suffix}.csv"


def _index_uploaded_files(uploaded_files):
    index = {}

    for item in uploaded_files or []:
        original_name = _text(item.get("original_name"))
        saved_name = _text(item.get("saved_name"))
        saved_path = _text(item.get("saved_path"))

        if not saved_path:
            continue

        path = Path(saved_path)

        for key in [original_name, saved_name, path.name, str(path)]:
            key = _text(key)

            if key:
                index[key] = path

    return index


def _resolve_file_paths(step, state):
    index = state["uploaded_file_index"]
    input_files = step.get("input_files") or []
    paths = []

    for name in input_files:
        name = _text(name)

        if not name:
            continue

        if name in index:
            paths.append(index[name])
            continue

        candidate = Path(name)

        if candidate.exists():
            paths.append(candidate)

    if not paths:
        paths = list(index.values())

    unique = []
    seen = set()

    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)

        if key not in seen:
            unique.append(path)
            seen.add(key)

    return unique


def _add_dataset_to_state(state, dataset):
    dataset_id = dataset.get("dataset_id")

    if not dataset_id:
        return

    state["dataset_by_id"][dataset_id] = dataset
    state["created_dataset_ids"].append(dataset_id)
    state["last_dataset_ids"] = [dataset_id]

    for key in [
        dataset.get("file_name"),
        dataset.get("description"),
        dataset.get("source"),
        Path(dataset.get("file_path", "")).name
    ]:
        key = _text(key)

        if key:
            state["dataset_lookup"][key] = dataset_id


def _resolve_dataset_ids(step, state):
    ids = []

    for value in step.get("input_dataset_ids") or []:
        value = _text(value)

        if not value:
            continue

        if value in state["dataset_by_id"]:
            ids.append(value)
        elif value in state["dataset_lookup"]:
            ids.append(state["dataset_lookup"][value])
        elif state["get_dataset"](value) is not None:
            ids.append(value)

    if not ids:
        ids = list(state["last_dataset_ids"])

    unique = []
    seen = set()

    for dataset_id in ids:
        if dataset_id not in seen:
            unique.append(dataset_id)
            seen.add(dataset_id)

    return unique


def _get_dataset(state, dataset_id):
    dataset = state["dataset_by_id"].get(dataset_id)

    if dataset is not None:
        return dataset

    dataset = state["get_dataset"](dataset_id)

    if dataset is not None:
        state["dataset_by_id"][dataset_id] = dataset

    return dataset


def _map_condition_for_dataset(dataset, condition_map, default_condition):
    keys = [
        dataset.get("file_name"),
        dataset.get("description"),
        dataset.get("source"),
        Path(dataset.get("file_path", "")).name,
        Path(dataset.get("file_path", "")).stem
    ]

    for key in keys:
        key = _text(key)

        if key in condition_map:
            return condition_map[key]

    return default_condition or dataset.get("description") or Path(dataset["file_path"]).stem


def _map_label_for_dataset(dataset, label_map, default_label):
    keys = [
        dataset.get("file_name"),
        dataset.get("description"),
        dataset.get("source"),
        Path(dataset.get("file_path", "")).name,
        Path(dataset.get("file_path", "")).stem
    ]

    for key in keys:
        key = _text(key)

        if key in label_map:
            return label_map[key]

    return default_label or dataset.get("file_name") or Path(dataset["file_path"]).stem


def _plot_defaults(params, dataset_id):
    return {
        "dataset_id": dataset_id,
        "x_col": _text(params.get("x_col")) or "global_time_min",
        "y_col": _text(params.get("y_col")) or "y_mean",
        "x_label": _text(params.get("x_label")) or "$t$ / min",
        "y_label": _text(params.get("y_label")) or "$j$ / mA cm$^{-2}$",
        "plot_type": _text(params.get("plot_type")) or "line",
        "plot_title": _text(params.get("plot_title")),
        "bottom_annotation": "",
        "marker_color": "#FF5F05",
        "line_color": "#13294B",
        "primary_label": "",
        "group_col": _text(params.get("group_col")) or _text(params.get("condition_col")) or "condition",
        "group_label": _text(params.get("legend_title")),
        "group_color_mode": "auto",
        "data_reduction": "raw",
        "summary_group_col": "none",
        "x_summary_method": "mean",
        "y_summary_method": "mean_tail",
        "tail_fraction": 0.2,
        "fit_guide": "connect",
        "smooth_window": 5,
        "use_step_axis": _bool(params.get("use_step_axis"), False),
        "step_axis_mode": "uniform_custom" if _text(params.get("step_axis_custom_labels")) else "auto_data",
        "step_axis_placement": "custom_positions" if _text(params.get("step_axis_custom_positions")) else "uniform",
        "step_axis_value_col": "",
        "step_axis_group_col": "",
        "step_axis_label": _text(params.get("step_axis_label")),
        "step_axis_max_ticks": 12,
        "secondary_mode": "none",
        "x2_col": "none",
        "y2_col": "none",
        "top_x_label": "",
        "right_y_label": "",
        "secondary_plot_type": "line",
        "second_marker_color": "#9A4DFF",
        "second_line_color": "#9A4DFF",
        "second_label": "",
        "line_order": "original",
        "show_markers": False,
        "show_legend": _bool(params.get("show_legend"), True),
        "marker_size": 8,
        "line_width": 2.0,
        "opacity": 1.0,
        "axis_label_size": 13,
        "tick_label_size": 11,
        "axis_label_weight": "normal",
        "spine_width": 1.2,
        "tick_width": 1.2,
        "tick_length": 5,
        "show_full_frame": True,
        "tick_direction": "in",
        "show_top_ticks": True,
        "show_bottom_ticks": True,
        "show_left_ticks": True,
        "show_right_ticks": True,
        "show_grid": True,
        "title_size": 13,
        "legend_font_size": 10,
        "legend_location": _text(params.get("legend_location")) or "auto",
        "legend_frame": _bool(params.get("legend_frame"), False),
        "figure_width": 8,
        "figure_height": 5,
        "figure_dpi": 150,
        "x_min": "",
        "x_max": "",
        "x_tick_mode": "auto",
        "x_major_interval": "",
        "x_minor_interval": "",
        "x_custom_ticks": "",
        "x_custom_tick_labels": "",
        "y_min": "",
        "y_max": "",
        "y_tick_mode": "auto",
        "y_major_interval": "",
        "y_minor_interval": "",
        "y_custom_ticks": "",
        "y_custom_tick_labels": "",
        "step_axis_decimal_places": 1,
        "step_axis_label_stride": 2,
        "step_axis_custom_labels": _text(params.get("step_axis_custom_labels")),
        "step_axis_custom_positions": _text(params.get("step_axis_custom_positions")),
        "step_axis_label_rotation": 0,
        "step_axis_label_pad": 14,
        "bottom_margin": ""
    }


def execute_workflow_plan(plan, context):
    if not isinstance(plan, dict):
        raise ValueError("Workflow plan must be a dictionary.")

    processed_dir = Path(context["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "uploaded_file_index": _index_uploaded_files(context.get("uploaded_files", [])),
        "get_dataset": context["get_dataset"],
        "register_dataset": context["register_dataset"],
        "create_plot": context["create_plot"],
        "dataset_by_id": {},
        "dataset_lookup": {},
        "created_dataset_ids": [],
        "last_dataset_ids": [],
        "plot_urls": [],
        "step_results": []
    }

    for step in plan.get("steps", []):
        action = _text(step.get("action"))
        params = _params(step)
        title = _text(step.get("title")) or action
        output_name = _text(step.get("output_name")) or _text(params.get("dataset_name")) or title

        if action == "clean_files":
            file_paths = _resolve_file_paths(step, state)

            if not file_paths:
                raise ValueError("No uploaded files are available for cleaning.")

            created_ids = []

            for raw_path in file_paths:
                output_path = _output_path(
                    processed_dir,
                    f"{output_name}_{raw_path.stem}",
                    "cleaned"
                )
                save_cleaned_dataset(raw_path, output_path)
                dataset = state["register_dataset"](
                    file_path=output_path,
                    dataset_type="processed_data",
                    source=f"AI workflow cleaned from {raw_path.name}",
                    uploaded_by="ai_workflow",
                    description=output_name
                )
                _add_dataset_to_state(state, dataset)
                created_ids.append(dataset["dataset_id"])

            state["last_dataset_ids"] = created_ids
            state["step_results"].append({
                "step_id": step.get("step_id"),
                "action": action,
                "created_dataset_ids": created_ids
            })

        elif action == "stitch_sequence":
            file_paths = _resolve_file_paths(step, state)

            if not file_paths:
                raise ValueError("No uploaded files are available for sequence stitching.")

            output_path = _output_path(processed_dir, output_name, "sequential")
            combine_sequential_file_paths(
                file_paths=file_paths,
                output_path=output_path,
                condition_label=_text(params.get("condition_label")) or output_name,
                dataset_label=_text(params.get("dataset_name")) or output_name,
                time_col=_text(params.get("time_col")) or "T_s",
                step_variable_col=_text(params.get("step_variable_col")) or "Vf_V_vs_Ref",
                step_duration_s=_float(params.get("sequence_duration_s"), 300),
                sequence_regex=_text(params.get("sequence_regex")) or r"#\s*(\d+)"
            )
            dataset = state["register_dataset"](
                file_path=output_path,
                dataset_type="sequential_data",
                source=f"AI workflow stitched sequence from {len(file_paths)} files",
                uploaded_by="ai_workflow",
                description=output_name
            )
            _add_dataset_to_state(state, dataset)
            state["step_results"].append({
                "step_id": step.get("step_id"),
                "action": action,
                "created_dataset_ids": [dataset["dataset_id"]]
            })

        elif action == "convert_variables":
            dataset_ids = _resolve_dataset_ids(step, state)

            if not dataset_ids:
                raise ValueError("No dataset is available for variable conversion.")

            created_ids = []

            for dataset_id in dataset_ids:
                dataset = _get_dataset(state, dataset_id)

                if dataset is None:
                    continue

                convert_potential = _bool(params.get("convert_potential"), bool(_text(params.get("potential_col"))))
                convert_current = _bool(params.get("convert_current"), bool(_text(params.get("current_col"))))

                if not convert_potential and not convert_current:
                    raise ValueError("Variable conversion requires at least one enabled conversion.")

                electrode_area_cm2 = _float(params.get("electrode_area_cm2"), 0)

                if convert_current and electrode_area_cm2 <= 0:
                    raise ValueError(
                        "Current-density conversion needs electrode_area_cm2. "
                        "Add electrode area to the AI workflow prompt and create the plan again."
                    )

                potential_col = _text(params.get("potential_col")) or "step_variable_value"
                current_col = _text(params.get("current_col")) or "Im_A"
                time_col = _text(params.get("time_col"))

                output_path = _output_path(
                    processed_dir,
                    f"{output_name}_{Path(dataset['file_path']).stem}",
                    "converted"
                )
                conversion_summary = _create_converted_dataset(
                    input_path=Path(dataset["file_path"]),
                    output_path=output_path,
                    convert_potential=convert_potential,
                    potential_col=potential_col,
                    potential_output_col=_text(params.get("potential_output_col")) or "E_RHE",
                    reference_offset_v=_float(params.get("reference_offset_v"), 0),
                    ph_value=_float(params.get("ph_value"), 0),
                    convert_current=convert_current,
                    current_col=current_col,
                    current_density_output_col=_text(params.get("current_density_output_col")) or "j_mA_cm2",
                    electrode_area_cm2=electrode_area_cm2,
                    time_col=time_col
                )
                registered = state["register_dataset"](
                    file_path=output_path,
                    dataset_type="converted_data",
                    source=f"AI workflow converted from {dataset.get('file_name', 'dataset')}",
                    uploaded_by="ai_workflow",
                    description=output_name
                )
                _add_dataset_to_state(state, registered)
                created_ids.append(registered["dataset_id"])

            state["last_dataset_ids"] = created_ids
            state["step_results"].append({
                "step_id": step.get("step_id"),
                "action": action,
                "created_dataset_ids": created_ids
            })

        elif action == "combine_datasets":
            dataset_ids = _resolve_dataset_ids(step, state)

            if len(dataset_ids) < 2:
                dataset_ids = list(state["created_dataset_ids"])

            datasets = [_get_dataset(state, dataset_id) for dataset_id in dataset_ids]
            datasets = [dataset for dataset in datasets if dataset is not None]

            if len(datasets) < 2:
                raise ValueError("At least two datasets are required for combining.")

            condition_map = {}
            label_map = {}

            for item in params.get("file_condition_map", []) or []:
                file_name = _text(item.get("file_name"))

                if file_name:
                    condition_map[file_name] = _text(item.get("condition"))
                    label_map[file_name] = _text(item.get("dataset_label"))

            file_paths = [Path(dataset["file_path"]) for dataset in datasets]
            condition_labels = [
                _map_condition_for_dataset(
                    dataset,
                    condition_map,
                    _text(params.get("condition_label"))
                )
                for dataset in datasets
            ]
            dataset_labels = [
                _map_label_for_dataset(
                    dataset,
                    label_map,
                    ""
                )
                for dataset in datasets
            ]

            output_path = _output_path(processed_dir, output_name, "combined")
            combine_file_paths(
                file_paths=file_paths,
                output_path=output_path,
                condition_labels=condition_labels,
                dataset_labels=dataset_labels
            )
            registered = state["register_dataset"](
                file_path=output_path,
                dataset_type="combined_data",
                source=f"AI workflow combined {len(datasets)} datasets",
                uploaded_by="ai_workflow",
                description=output_name
            )
            _add_dataset_to_state(state, registered)
            state["step_results"].append({
                "step_id": step.get("step_id"),
                "action": action,
                "created_dataset_ids": [registered["dataset_id"]]
            })

        elif action == "average_replicates":
            dataset_ids = _resolve_dataset_ids(step, state)

            if not dataset_ids:
                raise ValueError("No combined dataset is available for replicate averaging.")

            dataset = _get_dataset(state, dataset_ids[0])

            if dataset is None:
                raise ValueError("Selected averaging dataset was not found.")

            output_path = _output_path(processed_dir, output_name, "averaged")
            result = average_condition_replicates(
                input_path=Path(dataset["file_path"]),
                output_path=output_path,
                x_col=_text(params.get("x_col")) or "global_time_min",
                y_col=_text(params.get("y_col")) or "j_mA_cm2",
                condition_col=_text(params.get("condition_col")) or "condition",
                replicate_col=_text(params.get("replicate_col")) or "dataset_label",
                method=_text(params.get("averaging_method")) or "interpolate",
                x_grid_method=_text(params.get("x_grid_method")) or "overlap",
                grid_points=_int(params.get("grid_points"), 500),
                x_round_decimals=_int(params.get("x_round_decimals"), 6),
                min_replicates=_int(params.get("min_replicates"), 1)
            )
            registered = state["register_dataset"](
                file_path=output_path,
                dataset_type="averaged_replicates",
                source=f"AI workflow averaged replicates from {dataset.get('file_name', 'dataset')}",
                uploaded_by="ai_workflow",
                description=output_name
            )
            _add_dataset_to_state(state, registered)
            state["step_results"].append({
                "step_id": step.get("step_id"),
                "action": action,
                "created_dataset_ids": [registered["dataset_id"]],
                "summary": result
            })

        elif action == "plot":
            dataset_ids = _resolve_dataset_ids(step, state)

            if not dataset_ids:
                raise ValueError("No dataset is available for plotting.")

            dataset_id = dataset_ids[0]
            plot_kwargs = _plot_defaults(params, dataset_id)
            plot_url = state["create_plot"](**plot_kwargs)
            state["plot_urls"].append(plot_url)
            state["step_results"].append({
                "step_id": step.get("step_id"),
                "action": action,
                "plot_url": plot_url,
                "dataset_id": dataset_id
            })

        else:
            raise ValueError(f"Unsupported workflow action: {action}")

    return {
        "created_dataset_ids": state["created_dataset_ids"],
        "last_dataset_ids": state["last_dataset_ids"],
        "plot_urls": state["plot_urls"],
        "step_results": state["step_results"]
    }
