
from pathlib import Path
from datetime import datetime
import re
import hashlib
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
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]

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


def _uploaded_entries(context):
    entries = []

    for item in context.get("uploaded_files", []) or []:
        if not isinstance(item, dict):
            continue

        path = Path(_text(item.get("saved_path")))

        if path.exists():
            entries.append({
                "original_name": _text(item.get("original_name")) or path.name,
                "path": path,
                "extension": path.suffix.lower().lstrip(".")
            })

    return entries


def _read_dta_curve(path):
    lines = Path(path).read_text(errors="ignore").splitlines()
    curve_index = None

    for index, line in enumerate(lines):
        if line.startswith("CURVE"):
            curve_index = index
            break

    if curve_index is None or curve_index + 2 >= len(lines):
        raise ValueError(f"No CURVE table found in {Path(path).name}")

    headers = [item.strip() for item in lines[curve_index + 1].split("\t")[1:] if item.strip()]
    rows = []

    for line in lines[curve_index + 3:]:
        if not line.strip():
            continue

        parts = line.split("\t")

        if len(parts) < len(headers) + 1:
            continue

        rows.append(parts[1:1 + len(headers)])

    if not rows:
        raise ValueError(f"No curve rows found in {Path(path).name}")

    df = pd.DataFrame(rows, columns=headers)

    for column in df.columns:
        df[column] = _to_number(df[column])

    return df


def _read_table(path):
    path = Path(path)
    extension = path.suffix.lower().lstrip(".")

    if extension in {"xlsx", "xls"}:
        return pd.read_excel(path)

    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"}
    ]
    last_error = None

    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)

            if df.shape[1] > 1:
                return df

        except Exception as error:
            last_error = error

    try:
        return pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception as error:
        last_error = error

    raise ValueError(f"Could not read tabular file {path.name}: {last_error}")


def _pick_column(columns, candidates, df=None, min_numeric=2):
    direct = {str(column).strip().lower(): column for column in columns}

    for candidate in candidates:
        key = str(candidate).lower()

        if key in direct:
            column = direct[key]

            if df is None or _to_number(df[column]).notna().sum() >= min_numeric:
                return column

    for column in columns:
        lower = str(column).strip().lower()

        for candidate in candidates:
            if str(candidate).lower() in lower:
                if df is None or _to_number(df[column]).notna().sum() >= min_numeric:
                    return column

    return ""


def _infer_condition(file_name):
    lower = str(file_name).lower()

    if "no3" in lower or "nano3" in lower or "pbsno3" in lower or "pbs+no3" in lower:
        return "PBS+NaNO3"

    if "pbs" in lower:
        return "PBS"

    stem = Path(str(file_name)).stem
    stem = re.sub(r"#\s*\d+.*$", "", stem)
    stem = re.sub(r"\(\d+\)$", "", stem)

    return stem.strip("_- ") or Path(str(file_name)).stem


def _sequence_number(file_name):
    stem = Path(str(file_name)).stem
    match = re.search(r"#\s*(\d+)", stem)

    if match:
        return int(match.group(1))

    numbers = re.findall(r"\d+", stem)

    return int(numbers[-1]) if numbers else 0


def _is_b_sequence(file_name):
    lower = str(file_name).lower()
    return "chronoa_b" in lower or "_b_" in lower


def _copy_index(file_name):
    stem = Path(str(file_name)).stem
    match = re.search(r"\((\d+)\)$", stem)

    if match:
        return int(match.group(1))

    return 0


def _content_hash(path):
    hasher = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def _sequence_prefix(file_name):
    stem = Path(str(file_name)).stem
    stem = re.sub(r"\(\d+\)$", "", stem)
    stem = re.sub(r"#\s*\d+.*$", "", stem)
    stem = re.sub(r"CHRONOA[_-]*B$", "CHRONOA", stem, flags=re.IGNORECASE)
    stem = stem.strip("_- ")
    return stem or Path(str(file_name)).stem


def _dataset_label(file_name):
    condition = _infer_condition(file_name)

    if condition == "PBS":
        return "PBS_stitched_sequence"

    prefix = re.sub(r"[^A-Za-z0-9]+", "_", _sequence_prefix(file_name)).strip("_")

    return prefix or condition


def _deduplicate(entries):
    kept = []
    seen_exact = {}
    duplicates = []

    for entry in entries:
        name = entry["original_name"]
        exact_key = (
            _infer_condition(name),
            _dataset_label(name),
            _is_b_sequence(name),
            _copy_index(name),
            _sequence_number(name),
            entry["extension"],
            _content_hash(entry["path"])
        )

        if exact_key in seen_exact:
            duplicates.append({
                "kept": seen_exact[exact_key]["original_name"],
                "discarded": name,
                "reason": "exact_same_content"
            })
            continue

        seen_exact[exact_key] = entry
        kept.append(entry)

    return kept, duplicates


def _repeated_sequence_report(entries):
    records = []

    for entry in entries:
        name = entry["original_name"]
        records.append({
            "file_name": name,
            "condition": _infer_condition(name),
            "dataset_label": _dataset_label(name),
            "is_b_sequence": bool(_is_b_sequence(name)),
            "copy_index": int(_copy_index(name)),
            "sequence_number": int(_sequence_number(name))
        })

    if not records:
        return []

    df = pd.DataFrame(records)
    repeated = []

    for keys, group in df.groupby(["condition", "dataset_label", "is_b_sequence", "sequence_number"], sort=False):
        if len(group) > 1:
            repeated.append({
                "condition": str(keys[0]),
                "dataset_label": str(keys[1]),
                "is_b_sequence": bool(keys[2]),
                "sequence_number": int(keys[3]),
                "copy_indices": sorted(group["copy_index"].astype(int).unique().tolist()),
                "files": group["file_name"].astype(str).tolist(),
                "interpretation": "non-identical repeated sequence files are kept; copy-index groups are treated as continuation blocks"
            })

    return repeated


def _inspect_entry(entry):
    result = {
        "file_name": entry["original_name"],
        "extension": entry["extension"],
        "readable": False,
        "rows": 0,
        "columns": 0,
        "column_names": [],
        "candidate_x": "",
        "candidate_y": "",
        "candidate_condition": "",
        "candidate_replicate": "",
        "error": ""
    }

    try:
        if entry["extension"] == "dta":
            df = _read_dta_curve(entry["path"])
        else:
            df = _read_table(entry["path"])

        result["readable"] = True
        result["rows"] = int(len(df))
        result["columns"] = int(len(df.columns))
        result["column_names"] = [str(column) for column in df.columns]
        result["candidate_x"] = _pick_column(df.columns, _x_candidates(), df)
        result["candidate_y"] = _pick_column(df.columns, _y_candidates(), df)
        result["candidate_condition"] = _pick_column(df.columns, _condition_candidates())
        result["candidate_replicate"] = _pick_column(df.columns, _replicate_candidates())
    except Exception as error:
        result["error"] = str(error)

    return result


def _x_candidates():
    return [
        "global_time_min",
        "x_value",
        "time_min",
        "time",
        "t",
        "T",
        "E_RHE",
        "potential",
        "voltage",
        "Vf",
        "E",
        "frequency",
        "freq",
        "wavelength",
        "x"
    ]


def _y_candidates():
    return [
        "y_mean",
        "y_value",
        "j_mA_cm2",
        "current_density",
        "current",
        "Im",
        "I",
        "signal",
        "response",
        "intensity",
        "absorbance",
        "y"
    ]


def _condition_candidates():
    return [
        "condition",
        "group",
        "sample",
        "electrolyte",
        "treatment",
        "catalyst",
        "label"
    ]


def _replicate_candidates():
    return [
        "dataset_label",
        "replicate",
        "sample_id",
        "file",
        "run",
        "trial"
    ]


def _execute_dta(entries, context):
    entries, duplicates = _deduplicate(entries)
    repeated_sequences = _repeated_sequence_report(entries)
    segments = []
    errors = []

    for entry in entries:
        try:
            raw = _read_dta_curve(entry["path"])
            time_col = _pick_column(raw.columns, ["T", "Time", "time"], raw)
            current_col = _pick_column(raw.columns, ["Im", "I", "Current"], raw)
            potential_col = _pick_column(raw.columns, ["Vf", "E", "Potential", "V"], raw)

            if not time_col or not current_col:
                raise ValueError(f"Missing time/current columns. Available columns: {list(raw.columns)}")

            time = _to_number(raw[time_col])
            current = _to_number(raw[current_col])
            potential = _to_number(raw[potential_col]) if potential_col else np.nan
            local_time = time if "min" in str(time_col).lower() else time / 60.0
            name = entry["original_name"]

            part = pd.DataFrame({
                "local_time_min": local_time,
                "E_RHE": potential - 1.0,
                "j_mA_cm2": current * 1000.0 / 0.283,
                "condition": _infer_condition(name),
                "dataset_label": _dataset_label(name),
                "source_file": name,
                "sequence_number": _sequence_number(name),
                "copy_index": _copy_index(name),
                "is_b_sequence": _is_b_sequence(name)
            }).dropna(subset=["local_time_min", "j_mA_cm2", "condition", "dataset_label"])

            if not part.empty:
                segments.append(part)

        except Exception as error:
            errors.append(f"{entry['original_name']}: {error}")

    if not segments:
        raise ValueError("No DTA segments could be parsed. " + "; ".join(errors[:5]))

    combined_raw = pd.concat(segments, ignore_index=True)
    stitched = []

    for (condition, label), group in combined_raw.groupby(["condition", "dataset_label"], sort=False):
        pieces = [piece.copy() for _, piece in group.groupby("source_file", sort=False)]
        pieces = sorted(
            pieces,
            key=lambda frame: (
                int(frame["is_b_sequence"].iloc[0]),
                int(_copy_index(str(frame["source_file"].iloc[0]))),
                int(frame["sequence_number"].iloc[0]),
                str(frame["source_file"].iloc[0])
            )
        )
        offset = 0.0

        for piece in pieces:
            local = _to_number(piece["local_time_min"])
            span = float(local.max() - local.min())
            piece["global_time_min"] = local - float(local.min()) + offset
            stitched.append(piece)

            if span > 0:
                offset += span

    combined = pd.concat(stitched, ignore_index=True).drop(columns=["local_time_min"], errors="ignore")
    averaged = _average_sequences(combined, "global_time_min", "j_mA_cm2", "condition", "dataset_label", "global_time_min")

    result = _save_register(combined, averaged, context, "dta_auto", errors, duplicates, "global_time_min", "j_mA_cm2")
    result["step_results"][0]["repeated_sequence_report"] = repeated_sequences
    return result


def _normalize_table_file(entry):
    df = _read_table(entry["path"])
    df.columns = [str(column).strip() for column in df.columns]
    x_col = _pick_column(df.columns, _x_candidates(), df)
    y_col = _pick_column(df.columns, _y_candidates(), df)
    condition_col = _pick_column(df.columns, _condition_candidates())
    replicate_col = _pick_column(df.columns, _replicate_candidates())

    if not x_col or not y_col:
        raise ValueError(f"Could not infer X/Y columns in {entry['original_name']}. Available columns: {list(df.columns)}")

    out = pd.DataFrame({
        "x_value": _to_number(df[x_col]),
        "y_value": _to_number(df[y_col]),
        "condition": df[condition_col].astype(str) if condition_col else _infer_condition(entry["original_name"]),
        "dataset_label": df[replicate_col].astype(str) if replicate_col else Path(entry["original_name"]).stem,
        "source_file": entry["original_name"],
        "source_x_col": x_col,
        "source_y_col": y_col
    })

    return out.dropna(subset=["x_value", "y_value", "condition", "dataset_label"])


def _execute_tabular(entries, context):
    parts = []
    errors = []

    for entry in entries:
        try:
            part = _normalize_table_file(entry)

            if not part.empty:
                parts.append(part)

        except Exception as error:
            errors.append(f"{entry['original_name']}: {error}")

    if not parts:
        raise ValueError("No tabular files could be converted into X/Y data. " + "; ".join(errors[:5]))

    combined = pd.concat(parts, ignore_index=True)
    averaged = _average_sequences(combined, "x_value", "y_value", "condition", "dataset_label", "x_value")

    return _save_register(combined, averaged, context, "tabular_auto", errors, [], "x_value", "y_value")


def _average_sequences(combined, x_col, y_col, condition_col, replicate_col, output_x_col):
    averaged_parts = []

    for condition, condition_data in combined.groupby(condition_col, sort=False):
        sequences = []

        for label, sequence_data in condition_data.groupby(replicate_col, sort=False):
            data = sequence_data[[x_col, y_col]].dropna().sort_values(x_col)
            data = data.groupby(x_col, as_index=False)[y_col].mean()

            if len(data) >= 2:
                sequences.append((str(label), data))

        if not sequences:
            continue

        if len(sequences) == 1:
            label, data = sequences[0]
            x = data[x_col].to_numpy(dtype=float)
            y = data[y_col].to_numpy(dtype=float)
            part = pd.DataFrame({
                "condition": condition,
                output_x_col: x,
                "y_mean": y,
                "y_std": 0.0,
                "y_sem": 0.0,
                "y_min": y,
                "y_max": y,
                "n_replicates": 1,
                "replicate_names": label,
                "source_x_col": x_col,
                "source_y_col": y_col,
                "averaging_method": "single_sequence",
                "x_grid_method": "observed"
            })
            averaged_parts.append(part)
            continue

        x_min = max(float(data[x_col].min()) for _, data in sequences)
        x_max = min(float(data[x_col].max()) for _, data in sequences)

        if x_max <= x_min:
            x_min = min(float(data[x_col].min()) for _, data in sequences)
            x_max = max(float(data[x_col].max()) for _, data in sequences)

        grid = np.linspace(x_min, x_max, 500)
        values = []
        labels = []

        for label, data in sequences:
            x = data[x_col].to_numpy(dtype=float)
            y = data[y_col].to_numpy(dtype=float)
            values.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
            labels.append(label)

        matrix = np.vstack(values)
        valid = ~np.all(np.isnan(matrix), axis=0)
        matrix = matrix[:, valid]
        grid = grid[valid]
        n = np.sum(~np.isnan(matrix), axis=0)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0, ddof=1)
        std = np.where(n > 1, std, 0.0)
        sem = np.where(n > 1, std / np.sqrt(n), 0.0)

        averaged_parts.append(pd.DataFrame({
            "condition": condition,
            output_x_col: grid,
            "y_mean": mean,
            "y_std": std,
            "y_sem": sem,
            "y_min": np.nanmin(matrix, axis=0),
            "y_max": np.nanmax(matrix, axis=0),
            "n_replicates": n,
            "replicate_names": ", ".join(labels),
            "source_x_col": x_col,
            "source_y_col": y_col,
            "averaging_method": "interpolate",
            "x_grid_method": "overlap"
        }))

    if not averaged_parts:
        raise ValueError("No averaged data was created after cleaning X/Y/condition/replicate columns.")

    return pd.concat(averaged_parts, ignore_index=True)


def _ranges(df, x_col, y_col):
    out = []

    for condition, group in df.groupby("condition", sort=False):
        x = _to_number(group[x_col]).dropna()
        y = _to_number(group[y_col]).dropna()

        out.append({
            "condition": str(condition),
            "rows": int(len(group)),
            "x_min": float(x.min()) if not x.empty else None,
            "x_max": float(x.max()) if not x.empty else None,
            "y_min": float(y.min()) if not y.empty else None,
            "y_max": float(y.max()) if not y.empty else None
        })

    return out



def _protocol_coverage_summary(df, x_col="global_time_min", condition_col="condition", expected_min=0.0, expected_max=160.0):
    if x_col not in df.columns or condition_col not in df.columns:
        return []

    expected_span = float(expected_max) - float(expected_min)
    summaries = []

    for condition, group in df.groupby(condition_col, sort=False):
        x = _to_number(group[x_col]).dropna()

        if x.empty:
            coverage = 0.0
            x_min = None
            x_max = None
        else:
            x_min = float(x.min())
            x_max = float(x.max())
            coverage = (x_max - x_min) / expected_span if expected_span > 0 else 0.0

        summaries.append({
            "condition": str(condition),
            "x_min": x_min,
            "x_max": x_max,
            "expected_min": float(expected_min),
            "expected_max": float(expected_max),
            "coverage_fraction": float(coverage),
            "appears_complete_for_reference_protocol": bool(coverage >= 0.85)
        })

    return summaries


def _save_register(combined, averaged, context, prefix, errors, duplicates, x_col, y_col):
    processed_dir = Path(context["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp()
    combined_path = processed_dir / f"{timestamp}_{prefix}_combined.csv"
    averaged_path = processed_dir / f"{timestamp}_{prefix}_averaged.csv"
    combined.to_csv(combined_path, index=False)
    averaged.to_csv(averaged_path, index=False)
    register_dataset = context["register_dataset"]
    combined_dataset = register_dataset(
        file_path=combined_path,
        dataset_type="combined_data",
        source=f"{prefix} combined dataset",
        uploaded_by="ai_workflow",
        description=f"{prefix}_combined"
    )
    averaged_dataset = register_dataset(
        file_path=averaged_path,
        dataset_type="averaged_replicates",
        source=f"{prefix} averaged dataset",
        uploaded_by="ai_workflow",
        description=f"{prefix}_averaged"
    )
    avg_x_col = x_col if x_col in averaged.columns else "global_time_min" if "global_time_min" in averaged.columns else "x_value"

    return {
        "created_dataset_ids": [combined_dataset["dataset_id"], averaged_dataset["dataset_id"]],
        "last_dataset_ids": [averaged_dataset["dataset_id"]],
        "plot_urls": [],
        "step_results": [
            {
                "step_id": 1,
                "action": f"{prefix}_auto_process",
                "created_dataset_ids": [combined_dataset["dataset_id"], averaged_dataset["dataset_id"]],
                "combined_summary": _ranges(combined, x_col, y_col),
                "averaged_summary": _ranges(averaged.rename(columns={"y_mean": "y_value"}), avg_x_col, "y_value"),
                "duplicate_file_report": duplicates,
                "errors": errors[:20],
                "dataset_columns": {
                    "combined": list(combined.columns),
                    "averaged": list(averaged.columns)
                },
                "protocol_coverage_summary": _protocol_coverage_summary(averaged, x_col if x_col in averaged.columns else "global_time_min")
            }
        ]
    }


def execute_workflow_plan(plan, context):
    entries = _uploaded_entries(context)

    if not entries:
        raise ValueError("No uploaded files are available for auto workflow.")

    inspections = [_inspect_entry(entry) for entry in entries]
    extensions = {entry["extension"] for entry in entries}

    try:
        if extensions.issubset({"dta"}):
            result = _execute_dta(entries, context)
        elif extensions & {"csv", "txt", "dat", "tsv", "xlsx", "xls"}:
            result = _execute_tabular(entries, context)
        else:
            raise ValueError(f"Unsupported file extensions for auto workflow: {sorted(extensions)}")

        result["file_inspection"] = inspections

        return result

    except Exception as error:
        return {
            "created_dataset_ids": [],
            "last_dataset_ids": [],
            "plot_urls": [],
            "file_inspection": inspections,
            "step_results": [
                {
                    "step_id": 1,
                    "action": "inspect_only_failed",
                    "error": str(error),
                    "file_inspection": inspections
                }
            ]
        }
