
from pathlib import Path
from datetime import datetime
import re
import pandas as pd
import numpy as np


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _text(value):
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in {"none", "null", "nan", "na", "n/a"}:
        return ""

    return value


def _to_number(value):
    if isinstance(value, pd.Series):
        cleaned = (
            value.astype(str)
            .str.strip()
            .str.replace("\u2212", "-", regex=False)
            .str.replace(",", ".", regex=False)
        )
        cleaned = cleaned.mask(cleaned.str.lower().isin(["", "none", "nan", "na", "n/a"]))
        return pd.to_numeric(cleaned, errors="coerce")

    return pd.to_numeric(str(value).strip().replace("\u2212", "-").replace(",", "."), errors="coerce")


def _safe_file_stem(value):
    value = Path(str(value)).stem
    value = re.sub(r"\(\d+\)$", "", value)
    value = re.sub(r"[^A-Za-z0-9_.#()+-]+", "_", value)
    return value.strip("_") or "dataset"


def _infer_condition(file_name):
    lower = str(file_name).lower()

    if "no3" in lower or "nano3" in lower or "pbsno3" in lower or "pbs+no3" in lower:
        return "PBS+NaNO3"

    if "pbs" in lower:
        return "PBS"

    return "Unknown"


def _sequence_number(file_name):
    stem = Path(str(file_name)).stem
    match = re.search(r"#\s*(\d+)", stem)

    if match:
        return int(match.group(1))

    numbers = re.findall(r"\d+", stem)

    if numbers:
        return int(numbers[-1])

    return 0


def _is_b_sequence(file_name):
    lower = str(file_name).lower()
    return "chronoa_b" in lower or "_b_" in lower


def _sequence_prefix(file_name):
    stem = Path(str(file_name)).stem
    stem = re.sub(r"\(\d+\)$", "", stem)
    stem = re.sub(r"#\s*\d+.*$", "", stem)
    stem = stem.strip("_- ")
    return stem or Path(str(file_name)).stem


def _normalized_ab_prefix(file_name):
    prefix = _sequence_prefix(file_name)
    prefix = re.sub(r"CHRONOA[_-]*B$", "CHRONOA", prefix, flags=re.IGNORECASE)
    prefix = re.sub(r"CHRONOA[_-]*B[_-]*$", "CHRONOA", prefix, flags=re.IGNORECASE)
    prefix = prefix.replace("CHRONOA_B", "CHRONOA")
    prefix = prefix.replace("chronoa_b", "chronoa")
    prefix = prefix.strip("_- ")
    return prefix


def _dataset_label(file_name):
    condition = _infer_condition(file_name)

    if condition == "PBS":
        return "PBS_stitched_sequence"

    if condition == "PBS+NaNO3":
        prefix = _normalized_ab_prefix(file_name)
        prefix = re.sub(r"[^A-Za-z0-9]+", "_", prefix).strip("_")
        return prefix or "PBS_NaNO3_stitched_sequence"

    return _normalized_ab_prefix(file_name)


def _read_dta_curve(path):
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()

    curve_index = None

    for index, line in enumerate(lines):
        if line.startswith("CURVE"):
            curve_index = index
            break

    if curve_index is None or curve_index + 2 >= len(lines):
        raise ValueError(f"No CURVE table found in {path.name}")

    header_parts = lines[curve_index + 1].split("\t")
    headers = [item.strip() for item in header_parts[1:] if item.strip()]

    rows = []

    for line in lines[curve_index + 3:]:
        if not line.strip():
            continue

        parts = line.split("\t")

        if len(parts) < len(headers) + 1:
            continue

        values = parts[1:1 + len(headers)]
        rows.append(values)

    if not rows:
        raise ValueError(f"No curve rows found in {path.name}")

    df = pd.DataFrame(rows, columns=headers)

    for column in df.columns:
        df[column] = _to_number(df[column])

    return df


def _select_column(columns, candidates):
    lower_map = {str(column).strip().lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for column in columns:
        lower = str(column).strip().lower()

        for candidate in candidates:
            if candidate.lower() in lower:
                return column

    return ""


def _load_segment(path, original_name, electrode_area_cm2, reference_offset_v, ph_value):
    raw = _read_dta_curve(path)
    columns = list(raw.columns)

    time_col = _select_column(columns, ["T", "Time", "time_s", "T_s"])
    potential_col = _select_column(columns, ["Vf", "E", "Potential", "V"])
    current_col = _select_column(columns, ["Im", "I", "Current", "current_A"])

    if not time_col:
        raise ValueError(f"No time column found in {original_name}. Available columns: {columns}")

    if not current_col:
        raise ValueError(f"No current column found in {original_name}. Available columns: {columns}")

    if not potential_col:
        raise ValueError(f"No potential column found in {original_name}. Available columns: {columns}")

    time_values = _to_number(raw[time_col])
    current_values = _to_number(raw[current_col])
    potential_values = _to_number(raw[potential_col])

    if "min" in str(time_col).lower():
        local_time_min = time_values
    else:
        local_time_min = time_values / 60.0

    if "ma" in str(current_col).lower() and "/a" not in str(current_col).lower():
        j_values = current_values / electrode_area_cm2
    else:
        j_values = current_values * 1000.0 / electrode_area_cm2

    condition = _infer_condition(original_name)

    out = pd.DataFrame({
        "local_time_min": local_time_min,
        "E_RHE": potential_values + reference_offset_v + 0.0591 * ph_value,
        "j_mA_cm2": j_values,
        "condition": condition,
        "dataset_label": _dataset_label(original_name),
        "source_file": original_name,
        "sequence_number": _sequence_number(original_name),
        "is_b_sequence": _is_b_sequence(original_name),
        "sequence_prefix": _sequence_prefix(original_name)
    })

    out = out.dropna(subset=["local_time_min", "j_mA_cm2", "condition", "dataset_label"])
    return out


def _uploaded_file_entries(context):
    entries = []

    for item in context.get("uploaded_files", []) or []:
        if not isinstance(item, dict):
            continue

        original_name = _text(item.get("original_name")) or Path(_text(item.get("saved_path"))).name
        saved_path = _text(item.get("saved_path"))

        if not saved_path:
            continue

        path = Path(saved_path)

        if path.exists():
            entries.append({
                "original_name": original_name,
                "path": path
            })

    return entries


def _deduplicate_entries(entries):
    selected = {}

    for entry in entries:
        name = entry["original_name"]
        condition = _infer_condition(name)
        label = _dataset_label(name)
        key = (condition, label, _is_b_sequence(name), _sequence_number(name))

        if key not in selected:
            selected[key] = entry
            continue

        existing = selected[key]["original_name"]

        if "(1)" in name and "(1)" not in existing:
            selected[key] = entry

    return list(selected.values())


def _sort_segment_key(frame):
    row = frame.iloc[0]
    condition = str(row["condition"])
    label = str(row["dataset_label"])
    is_b = int(row["is_b_sequence"])
    number = int(row["sequence_number"])

    return (condition, label, is_b, number)


def _stitch_segments(segments):
    stitched = []

    for (condition, dataset_label), group in pd.concat(segments, ignore_index=True).groupby(["condition", "dataset_label"], sort=False):
        pieces = []

        for source_file, piece in group.groupby("source_file", sort=False):
            pieces.append(piece.copy())

        pieces = sorted(pieces, key=_sort_segment_key)

        offset = 0.0

        for piece in pieces:
            local = _to_number(piece["local_time_min"])
            local_min = float(local.min())
            local_max = float(local.max())
            span = max(local_max - local_min, 0.0)

            piece["global_time_min"] = local - local_min + offset
            stitched.append(piece)

            if span > 0:
                offset += span

    if not stitched:
        raise ValueError("No valid segments remained after stitching.")

    out = pd.concat(stitched, ignore_index=True)
    out = out.drop(columns=["local_time_min"], errors="ignore")
    return out


def _average_condition_sequences(combined):
    averaged_parts = []

    for condition, condition_data in combined.groupby("condition", sort=False):
        sequences = []

        for dataset_label, sequence_data in condition_data.groupby("dataset_label", sort=False):
            data = sequence_data[["global_time_min", "j_mA_cm2"]].dropna().sort_values("global_time_min")
            data = data.groupby("global_time_min", as_index=False)["j_mA_cm2"].mean()

            if len(data) >= 2:
                sequences.append((str(dataset_label), data))

        if not sequences:
            continue

        if len(sequences) == 1:
            label, data = sequences[0]
            part = pd.DataFrame({
                "condition": condition,
                "global_time_min": data["global_time_min"],
                "y_mean": data["j_mA_cm2"],
                "y_std": 0.0,
                "y_sem": 0.0,
                "y_min": data["j_mA_cm2"],
                "y_max": data["j_mA_cm2"],
                "n_replicates": 1,
                "replicate_names": label,
                "source_x_col": "global_time_min",
                "source_y_col": "j_mA_cm2",
                "averaging_method": "single_sequence",
                "x_grid_method": "observed"
            })
            averaged_parts.append(part)
            continue

        x_min = max(float(data["global_time_min"].min()) for _, data in sequences)
        x_max = min(float(data["global_time_min"].max()) for _, data in sequences)

        if x_max <= x_min:
            x_min = min(float(data["global_time_min"].min()) for _, data in sequences)
            x_max = max(float(data["global_time_min"].max()) for _, data in sequences)

        grid = np.linspace(x_min, x_max, 1000)
        values = []
        names = []

        for label, data in sequences:
            x = data["global_time_min"].to_numpy(dtype=float)
            y = data["j_mA_cm2"].to_numpy(dtype=float)

            if len(x) < 2:
                continue

            interp = np.interp(grid, x, y, left=np.nan, right=np.nan)
            values.append(interp)
            names.append(label)

        if not values:
            continue

        matrix = np.vstack(values)
        valid = ~np.all(np.isnan(matrix), axis=0)
        matrix = matrix[:, valid]
        valid_grid = grid[valid]

        n = np.sum(~np.isnan(matrix), axis=0)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0, ddof=1)
        std = np.where(n > 1, std, 0.0)
        sem = np.where(n > 1, std / np.sqrt(n), 0.0)

        part = pd.DataFrame({
            "condition": condition,
            "global_time_min": valid_grid,
            "y_mean": mean,
            "y_std": std,
            "y_sem": sem,
            "y_min": np.nanmin(matrix, axis=0),
            "y_max": np.nanmax(matrix, axis=0),
            "n_replicates": n,
            "replicate_names": ", ".join(names),
            "source_x_col": "global_time_min",
            "source_y_col": "j_mA_cm2",
            "averaging_method": "interpolate",
            "x_grid_method": "overlap" if x_max > x_min else "union"
        })
        averaged_parts.append(part)

    if not averaged_parts:
        raise ValueError("No averaged data was created. Check DTA parsing, condition mapping, and current conversion.")

    return pd.concat(averaged_parts, ignore_index=True)


def _summarize_condition_ranges(df, x_col="global_time_min", y_col="j_mA_cm2"):
    summaries = []

    for condition, group in df.groupby("condition", sort=False):
        x = _to_number(group[x_col]).dropna()
        y = _to_number(group[y_col]).dropna()

        if "dataset_label" in group.columns:
            labels = sorted(group["dataset_label"].dropna().astype(str).unique().tolist())
        elif "replicate_names" in group.columns:
            labels = sorted(group["replicate_names"].dropna().astype(str).unique().tolist())
        else:
            labels = []

        summaries.append({
            "condition": str(condition),
            "rows": int(len(group)),
            "x_min": float(x.min()) if not x.empty else None,
            "x_max": float(x.max()) if not x.empty else None,
            "y_min": float(y.min()) if not y.empty else None,
            "y_max": float(y.max()) if not y.empty else None,
            "dataset_labels": labels
        })

    return summaries



def _summarize_sequence_order(df):
    summaries = []

    columns = {"condition", "dataset_label", "source_file", "sequence_number", "is_b_sequence", "global_time_min", "j_mA_cm2"}

    if not columns.issubset(set(df.columns)):
        return summaries

    for (condition, label), group in df.groupby(["condition", "dataset_label"], sort=False):
        files = (
            group[["source_file", "sequence_number", "is_b_sequence"]]
            .drop_duplicates()
            .sort_values(["is_b_sequence", "sequence_number", "source_file"])
        )

        x = _to_number(group["global_time_min"]).dropna()
        y = _to_number(group["j_mA_cm2"]).dropna()

        summaries.append({
            "condition": str(condition),
            "dataset_label": str(label),
            "n_sequence_files": int(len(files)),
            "first_files": files["source_file"].astype(str).head(8).tolist(),
            "last_files": files["source_file"].astype(str).tail(8).tolist(),
            "x_min": float(x.min()) if not x.empty else None,
            "x_max": float(x.max()) if not x.empty else None,
            "y_min": float(y.min()) if not y.empty else None,
            "y_max": float(y.max()) if not y.empty else None
        })

    return summaries


def execute_workflow_plan(plan, context):
    processed_dir = Path(context["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    electrode_area_cm2 = 0.283
    reference_offset_v = -1.0
    ph_value = 0.0

    entries = _uploaded_file_entries(context)
    entries = _deduplicate_entries(entries)

    if not entries:
        raise ValueError("No uploaded DTA files are available for the direct DTA workflow.")

    segments = []
    errors = []

    for entry in entries:
        try:
            segments.append(
                _load_segment(
                    path=entry["path"],
                    original_name=entry["original_name"],
                    electrode_area_cm2=electrode_area_cm2,
                    reference_offset_v=reference_offset_v,
                    ph_value=ph_value
                )
            )
        except Exception as error:
            errors.append(f"{entry['original_name']}: {error}")

    if not segments:
        raise ValueError("No DTA segments could be parsed. " + "; ".join(errors[:5]))

    combined = _stitch_segments(segments)
    combined = combined.sort_values(["condition", "dataset_label", "global_time_min"]).reset_index(drop=True)
    averaged = _average_condition_sequences(combined)

    timestamp = _timestamp()
    combined_path = processed_dir / f"{timestamp}_direct_dta_combined.csv"
    averaged_path = processed_dir / f"{timestamp}_direct_dta_averaged.csv"

    combined.to_csv(combined_path, index=False)
    averaged.to_csv(averaged_path, index=False)

    register_dataset = context["register_dataset"]

    combined_dataset = register_dataset(
        file_path=combined_path,
        dataset_type="combined_data",
        source="Direct deterministic DTA workflow combined and stitched dataset",
        uploaded_by="ai_workflow",
        description="direct_dta_combined"
    )

    averaged_dataset = register_dataset(
        file_path=averaged_path,
        dataset_type="averaged_replicates",
        source="Direct deterministic DTA workflow averaged dataset",
        uploaded_by="ai_workflow",
        description="direct_dta_averaged"
    )

    condition_counts = combined.groupby("condition")["source_file"].nunique().to_dict()

    return {
        "created_dataset_ids": [
            combined_dataset["dataset_id"],
            averaged_dataset["dataset_id"]
        ],
        "last_dataset_ids": [averaged_dataset["dataset_id"]],
        "plot_urls": [],
        "step_results": [
            {
                "step_id": 1,
                "action": "direct_dta_parse_stitch_average",
                "created_dataset_ids": [
                    combined_dataset["dataset_id"],
                    averaged_dataset["dataset_id"]
                ],
                "condition_counts": condition_counts,
                "combined_summary": _summarize_condition_ranges(combined, "global_time_min", "j_mA_cm2"),
                "sequence_order_summary": _summarize_sequence_order(combined),
                "averaged_summary": _summarize_condition_ranges(averaged.rename(columns={"y_mean": "j_mA_cm2"}), "global_time_min", "j_mA_cm2"),
                "errors": errors[:10]
            }
        ]
    }
