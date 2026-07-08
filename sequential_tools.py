from pathlib import Path

import pandas as pd

from data_loader import read_dataset
from processing_utils import extract_sequence_index as _extract_sequence_index
from processing_utils import resolve_column
from processing_utils import to_numeric_series


DEFAULT_SEQUENCE_REGEX = r"#\s*(\d+)"
TIME_COLUMN_ALIASES = ["T_s", "T", "Time", "time_s", "time_sec", "seconds", "elapsed_time_s"]
STEP_VARIABLE_ALIASES = ["Vf_V_vs_Ref", "Vf_V", "Vf", "E", "Potential", "Potential_V", "voltage", "step_variable_value"]


def extract_sequence_index(file_path, fallback_index, sequence_regex=DEFAULT_SEQUENCE_REGEX):
    return _extract_sequence_index(
        file_name=Path(file_path).name,
        fallback_index=fallback_index,
        sequence_regex=sequence_regex
    )


def protect_existing_column(df, column_name):
    if column_name in df.columns:
        new_name = f"original_{column_name}"
        counter = 1

        while new_name in df.columns:
            counter += 1
            new_name = f"original_{column_name}_{counter}"

        df = df.rename(columns={column_name: new_name})

    return df


def parse_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_sequential_columns(
    df,
    file_path,
    fallback_index,
    condition_label,
    dataset_label,
    time_col,
    step_variable_col,
    step_duration_s,
    sequence_regex
):
    df = df.copy()

    protected_columns = [
        "source_file",
        "source_path",
        "source_index",
        "sequence_index",
        "condition",
        "dataset_label",
        "replicate",
        "global_time_s",
        "global_time_min",
        "step_duration_s",
        "step_variable_name",
        "step_variable_value",
        "step_label"
    ]

    for column in protected_columns:
        df = protect_existing_column(df, column)

    sequence_index = extract_sequence_index(
        file_path=file_path,
        fallback_index=fallback_index,
        sequence_regex=sequence_regex
    )

    df.insert(0, "source_file", Path(file_path).name)
    df.insert(1, "source_path", str(file_path))
    df.insert(2, "source_index", fallback_index)
    df.insert(3, "sequence_index", sequence_index)
    df.insert(4, "condition", condition_label)
    df.insert(5, "dataset_label", dataset_label)
    df.insert(6, "replicate", sequence_index)

    duration = parse_float_or_none(step_duration_s)

    resolved_time_col = resolve_column(
        df.columns,
        preferred=time_col,
        aliases=TIME_COLUMN_ALIASES,
        df=df,
        min_numeric=2
    )
    resolved_step_variable_col = resolve_column(
        df.columns,
        preferred=step_variable_col,
        aliases=STEP_VARIABLE_ALIASES,
        df=df,
        min_numeric=1
    )

    if resolved_time_col in df.columns:
        local_time = to_numeric_series(df[resolved_time_col])

        if duration is None:
            valid_time = local_time.dropna()

            if len(valid_time) > 1:
                duration = float(valid_time.max() - valid_time.min())
            else:
                duration = 300.0

        df["step_duration_s"] = duration
        df["global_time_s"] = local_time + (sequence_index - 1) * duration
        df["global_time_min"] = df["global_time_s"] / 60.0
    else:
        if duration is None:
            duration = 300.0

        df["step_duration_s"] = duration
        df["global_time_s"] = pd.NA
        df["global_time_min"] = pd.NA

    if resolved_step_variable_col in df.columns:
        step_values = to_numeric_series(df[resolved_step_variable_col])
        step_value = step_values.mean()

        df["step_variable_name"] = resolved_step_variable_col
        df["step_variable_value"] = step_value

        if pd.notna(step_value):
            df["step_label"] = f"{step_value:.4g}"
        else:
            df["step_label"] = ""
    else:
        df["step_variable_name"] = step_variable_col
        df["step_variable_value"] = pd.NA
        df["step_label"] = ""

    return df


def combine_sequential_file_paths(
    file_paths,
    output_path,
    condition_label,
    dataset_label,
    time_col="T_s",
    step_variable_col="Vf_V_vs_Ref",
    step_duration_s=300,
    sequence_regex=DEFAULT_SEQUENCE_REGEX
):
    indexed_paths = []

    for fallback_index, file_path in enumerate(file_paths, start=1):
        sequence_index = extract_sequence_index(
            file_path=file_path,
            fallback_index=fallback_index,
            sequence_regex=sequence_regex
        )

        indexed_paths.append((sequence_index, fallback_index, Path(file_path)))

    indexed_paths = sorted(indexed_paths, key=lambda item: item[0])

    frames = []

    for _, fallback_index, file_path in indexed_paths:
        df = read_dataset(file_path)

        df = add_sequential_columns(
            df=df,
            file_path=file_path,
            fallback_index=fallback_index,
            condition_label=condition_label,
            dataset_label=dataset_label,
            time_col=time_col,
            step_variable_col=step_variable_col,
            step_duration_s=step_duration_s,
            sequence_regex=sequence_regex
        )

        frames.append(df)

    if not frames:
        raise ValueError("No sequential files were available to combine.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["sequence_index", "global_time_s"], na_position="last")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    return output_path
