
from pathlib import Path
from datetime import datetime
import re
import hashlib
import pandas as pd
import numpy as np


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))

    if isinstance(value, pd.Series):
        return _json_safe(value.tolist())

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value

    if isinstance(value, int):
        return value

    if value is pd.NA:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


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

        saved_path = _text(item.get("saved_path"))

        if not saved_path:
            continue

        path = Path(saved_path)

        if not path.exists():
            continue

        entries.append({
            "original_name": _text(item.get("original_name")) or path.name,
            "path": path,
            "extension": path.suffix.lower().lstrip("."),
            "content_hash": _content_hash(path)
        })

    return entries


def _plan_steps(plan):
    if not isinstance(plan, dict):
        return []

    steps = plan.get("steps", [])

    return steps if isinstance(steps, list) else []


def _plan_parameters(plan):
    parameters = {}

    for step in _plan_steps(plan):
        if not isinstance(step, dict):
            continue

        step_parameters = step.get("parameters") or {}

        if isinstance(step_parameters, dict):
            parameters.update(step_parameters)

    return parameters


def _plan_dataset_ids(plan):
    dataset_ids = []

    for step in _plan_steps(plan):
        if not isinstance(step, dict):
            continue

        values = step.get("input_dataset_ids") or []

        if isinstance(values, str):
            values = [values]

        for value in values:
            value = _text(value)

            if value and value not in dataset_ids:
                dataset_ids.append(value)

    return dataset_ids


def _selected_dataset_entries(context, plan):
    entries = []
    get_dataset = context.get("get_dataset")

    if not callable(get_dataset):
        return entries

    for dataset_id in _plan_dataset_ids(plan):
        dataset = get_dataset(dataset_id)

        if not isinstance(dataset, dict):
            continue

        path = Path(_text(dataset.get("file_path")))

        if not path.exists():
            continue

        entries.append({
            "original_name": _text(dataset.get("file_name")) or path.name,
            "path": path,
            "extension": path.suffix.lower().lstrip("."),
            "content_hash": _content_hash(path),
            "dataset_id": dataset_id
        })

    return entries


def _content_hash(path):
    hasher = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def _read_table_headerless(path, nrows=None):
    path = Path(path)
    extension = path.suffix.lower().lstrip(".")

    if extension in {"xlsx", "xls"}:
        return pd.read_excel(path, header=None, nrows=nrows)

    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
        {"sep": r"\s+", "engine": "python"}
    ]
    last_error = None

    for kwargs in attempts:
        try:
            df = pd.read_csv(path, header=None, nrows=nrows, **kwargs)

            if df.shape[1] >= 2:
                return df

        except Exception as error:
            last_error = error

    raise ValueError(f"Could not read table {path.name}: {last_error}")


def _read_table_with_header(path):
    path = Path(path)
    extension = path.suffix.lower().lstrip(".")

    if extension in {"xlsx", "xls"}:
        return pd.read_excel(path)

    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
        {"sep": r"\s+", "engine": "python"}
    ]
    last_error = None

    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)

            if df.shape[1] >= 2:
                return df

        except Exception as error:
            last_error = error

    raise ValueError(f"Could not read table {path.name}: {last_error}")


def _has_dta_curve(path):
    try:
        with open(path, "r", errors="ignore") as handle:
            for line in handle:
                if line.startswith("CURVE"):
                    return True
    except Exception:
        return False

    return False


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
        raise ValueError(f"No curve rows found in {path.name}")

    df = pd.DataFrame(rows, columns=headers)

    for column in df.columns:
        df[column] = _to_number(df[column])

    return df


def _token(value):
    return str(value).strip().lower()


def _safe_column_name(value, fallback):
    name = re.sub(r"[^\w]+", "_", str(value).strip())
    name = re.sub(r"_+", "_", name).strip("_")

    return name or fallback


def _numeric_columns(df, min_numeric=2):
    columns = []

    for column in df.columns:
        if _to_number(df[column]).notna().sum() >= min_numeric:
            columns.append(column)

    return columns


def _categorical_columns(df, max_unique=30):
    columns = []

    for column in df.columns:
        if column in _numeric_columns(df):
            continue

        series = df[column].dropna()

        if series.empty:
            continue

        unique_count = int(series.astype(str).nunique())

        if 1 < unique_count <= max_unique:
            columns.append(column)

    return columns


def _looks_like_ordered_x(value):
    text = _token(value)
    terms = [
        "time",
        "date",
        "timestamp",
        "year",
        "month",
        "day",
        "index",
        "step",
        "sequence",
        "wavelength",
        "frequency",
        "position",
        "distance"
    ]

    return text in {"t", "x"} or any(term in text for term in terms)


def _looks_like_x_header(value):
    text = _token(value)

    if not text:
        return False

    terms = [
        "wavelength",
        "wave length",
        "nm",
        "time",
        "date",
        "timestamp",
        "year",
        "month",
        "day",
        "index",
        "step",
        "potential",
        "voltage",
        "frequency",
        "freq",
        "energy",
        "position",
        "distance",
        "angle",
        "x"
    ]

    return any(term in text for term in terms)


def _looks_like_y_header(value):
    text = _token(value)

    if not text:
        return False

    terms = [
        "abs",
        "absorbance",
        "current",
        "intensity",
        "signal",
        "response",
        "counts",
        "count",
        "value",
        "measurement",
        "result",
        "score",
        "amount",
        "total",
        "rate",
        "price",
        "sales",
        "revenue",
        "transmittance",
        "reflectance",
        "y"
    ]

    return any(term in text for term in terms)


def _detect_wide_pair_table(path):
    try:
        raw = _read_table_headerless(path, nrows=8)
    except Exception:
        return None

    if raw.shape[0] < 3 or raw.shape[1] < 2:
        return None

    best = None

    for condition_row_index in range(min(4, raw.shape[0] - 1)):
        variable_row_index = condition_row_index + 1
        pairs = []
        column = 0

        while column < raw.shape[1] - 1:
            condition = _text(raw.iloc[condition_row_index, column])
            x_header = _text(raw.iloc[variable_row_index, column])
            y_header = _text(raw.iloc[variable_row_index, column + 1])

            if condition and _looks_like_x_header(x_header) and _looks_like_y_header(y_header):
                pairs.append({
                    "condition": condition,
                    "x_col_index": column,
                    "y_col_index": column + 1,
                    "x_header": x_header,
                    "y_header": y_header
                })
                column += 2
                continue

            column += 1

        if len(pairs) >= 2:
            score = len(pairs)
            current = {
                "parser": "wide_pair_table",
                "condition_row_index": condition_row_index,
                "variable_row_index": variable_row_index,
                "data_start_row": variable_row_index + 1,
                "pairs": pairs,
                "confidence": "high" if len(pairs) >= 3 else "medium",
                "score": score
            }

            if best is None or current["score"] > best["score"]:
                best = current

    return best


def _find_column(columns, candidates, df=None, min_numeric=2):
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
            key = str(candidate).lower()

            if len(key) < 2:
                continue

            if key in lower:
                if df is None or _to_number(df[column]).notna().sum() >= min_numeric:
                    return column

    return ""


def _detect_long_table(path):
    try:
        df = _read_table_with_header(path)
    except Exception:
        return None

    if df.shape[1] < 2:
        return None

    numeric_columns = _numeric_columns(df)

    x_col = _find_column(
        df.columns,
        ["x", "x_value", "wavelength_nm", "wavelength", "lambda", "time_min", "time", "t", "date", "timestamp", "year", "month", "day", "index", "step", "potential", "voltage", "E_RHE", "Vf", "frequency", "position", "distance"],
        df
    )
    y_col = _find_column(
        df.columns,
        ["y", "y_value", "value", "measurement", "result", "score", "amount", "total", "rate", "price", "sales", "revenue", "absorbance", "abs", "y_mean", "j_mA_cm2", "current_density", "current", "Im", "I", "intensity", "signal", "response"],
        df
    )

    if not x_col and numeric_columns:
        ordered = [column for column in numeric_columns if _looks_like_ordered_x(column)]
        x_col = ordered[0] if ordered else numeric_columns[0]

    if not y_col:
        y_candidates = [column for column in numeric_columns if column != x_col]

        if y_candidates:
            y_col = y_candidates[0]

    condition_col = _find_column(
        df.columns,
        ["condition", "group", "category", "class", "segment", "sample", "label", "treatment", "type", "series", "cohort", "region", "electrolyte", "catalyst"],
        None
    )

    if not condition_col:
        candidates = [column for column in _categorical_columns(df) if column not in {x_col, y_col}]
        condition_col = candidates[0] if candidates else ""

    if not x_col or not y_col:
        return None

    return {
        "parser": "long_table",
        "x_col": str(x_col),
        "y_col": str(y_col),
        "condition_col": str(condition_col) if condition_col else "",
        "confidence": "high" if condition_col else "medium"
    }


def _classify_entry(entry):
    path = entry["path"]
    extension = entry["extension"]

    if extension == "dta" and _has_dta_curve(path):
        return {
            "parser": "gamry_dta_curve",
            "confidence": "high",
            "reason": "DTA file contains a CURVE table."
        }

    if extension in {"csv", "txt", "tsv", "dat", "xlsx", "xls"}:
        wide = _detect_wide_pair_table(path)

        if wide is not None:
            return wide

        long_table = _detect_long_table(path)

        if long_table is not None:
            return long_table

    return {
        "parser": "inspect_only",
        "confidence": "low",
        "reason": "No supported data shape was detected."
    }


def _inspect_entry(entry):
    report = {
        "file_name": entry["original_name"],
        "extension": entry["extension"],
        "content_hash": entry["content_hash"][:16],
        "readable": False,
        "rows": 0,
        "columns": 0,
        "preview": [],
        "classification": None,
        "error": ""
    }

    try:
        if entry["extension"] == "dta" and _has_dta_curve(entry["path"]):
            df = _read_dta_curve(entry["path"])
        else:
            df = _read_table_headerless(entry["path"], nrows=8)

        report["readable"] = True
        report["rows"] = int(len(df))
        report["columns"] = int(len(df.columns))
        report["preview"] = df.head(5).astype(str).values.tolist()
        report["classification"] = _classify_entry(entry)
    except Exception as error:
        report["classification"] = _classify_entry(entry)
        report["error"] = str(error)

    return report


def _output_x_name(x_header):
    text = _token(x_header)

    if "wavelength" in text or "nm" in text:
        return "wavelength_nm"

    if "time" in text and "min" in text:
        return "time_min"

    if "time" in text:
        return "time"

    if "potential" in text or "voltage" in text or text in {"e", "vf"}:
        return "potential"

    if "frequency" in text or "freq" in text:
        return "frequency"

    return _safe_column_name(x_header, "x_value")


def _output_y_name(y_header):
    text = _token(y_header)

    if text == "abs" or "absorbance" in text:
        return "absorbance"

    if "current" in text:
        return "current"

    if "intensity" in text:
        return "intensity"

    if "signal" in text:
        return "signal"

    if "response" in text:
        return "response"

    return _safe_column_name(y_header, "y_value")


def _parse_wide_pair_table(entry, classification):
    raw = _read_table_headerless(entry["path"])
    data_start = classification["data_start_row"]
    rows = []
    diagnostics = []

    for pair in classification["pairs"]:
        condition = pair["condition"]
        x_header = pair["x_header"]
        y_header = pair["y_header"]
        x_col_index = pair["x_col_index"]
        y_col_index = pair["y_col_index"]
        x_name = _output_x_name(x_header)
        y_name = _output_y_name(y_header)
        x = _to_number(raw.iloc[data_start:, x_col_index])
        y = _to_number(raw.iloc[data_start:, y_col_index])
        temp = pd.DataFrame({
            "condition": condition,
            x_name: x,
            y_name: y,
            "x_value": x,
            "y_value": y,
            "source_file": entry["original_name"],
            "source_x_col": f"{condition}::{x_header}",
            "source_y_col": f"{condition}::{y_header}",
            "dataset_label": condition
        }).dropna(subset=["x_value", "y_value"])

        if y_name == "absorbance":
            temp["saturated_or_clipped"] = temp[y_name] >= 9.99

        if not temp.empty:
            rows.append(temp.sort_values("x_value"))

        diagnostics.append({
            "condition": condition,
            "x_header": x_header,
            "y_header": y_header,
            "rows": int(len(temp)),
            "output_x_column": x_name,
            "output_y_column": y_name
        })

    if not rows:
        raise ValueError(f"{entry['original_name']}: no valid wide-pair data rows were parsed.")

    combined = pd.concat(rows, ignore_index=True)
    preferred_x = diagnostics[0]["output_x_column"] if diagnostics else "x_value"
    preferred_y = diagnostics[0]["output_y_column"] if diagnostics else "y_value"
    x_candidates = [column for column in [preferred_x, "wavelength_nm", "time_min", "time", "potential", "frequency", "x_value"] if column in combined.columns]
    y_candidates = [column for column in [preferred_y, "absorbance", "current", "intensity", "signal", "response", "y_value"] if column in combined.columns]

    return combined, {
        "parser": "wide_pair_table",
        "x_column": x_candidates[0] if x_candidates else "x_value",
        "y_column": y_candidates[0] if y_candidates else "y_value",
        "group_column": "condition",
        "diagnostics": diagnostics
    }


def _parse_long_table(entry, classification):
    df = _read_table_with_header(entry["path"])
    x_col = classification["x_col"]
    y_col = classification["y_col"]
    condition_col = classification.get("condition_col") or ""
    condition = df[condition_col].astype(str) if condition_col and condition_col in df.columns else Path(entry["original_name"]).stem
    x = _to_number(df[x_col])
    y = _to_number(df[y_col])
    x_name = _output_x_name(x_col)
    y_name = _output_y_name(y_col)
    output_data = {
        "condition": condition,
        x_name: x,
        y_name: y,
        "x_value": x,
        "y_value": y,
        "source_file": entry["original_name"],
        "source_x_col": x_col,
        "source_y_col": y_col,
        "dataset_label": condition
    }

    used_names = set(output_data)

    for column in df.columns:
        if column in {x_col, y_col, condition_col}:
            continue

        values = _to_number(df[column])

        if values.notna().sum() < 2:
            continue

        output_name = _safe_column_name(column, "value")
        base_name = output_name
        suffix = 2

        while output_name in used_names:
            output_name = f"{base_name}_{suffix}"
            suffix += 1

        output_data[output_name] = values
        used_names.add(output_name)

    out = pd.DataFrame(output_data).dropna(subset=["x_value", "y_value"])

    if y_name == "absorbance":
        out["saturated_or_clipped"] = out[y_name] >= 9.99

    if out.empty:
        raise ValueError(f"{entry['original_name']}: no valid long-table numeric rows were parsed.")

    return out.sort_values(["condition", "x_value"]), {
        "parser": "long_table",
        "x_column": x_name,
        "y_column": y_name,
        "group_column": "condition",
        "diagnostics": [{
            "x_header": x_col,
            "y_header": y_col,
            "condition_header": condition_col
        }]
    }


def _infer_condition(file_name):
    lower = str(file_name).lower()

    if "no3" in lower or "nano3" in lower or "pbsno3" in lower or "pbs+no3" in lower:
        return "PBS+NaNO3"

    if "pbs" in lower:
        return "PBS"

    return Path(str(file_name)).stem


def _sequence_number(file_name):
    stem = Path(str(file_name)).stem
    match = re.search(r"#\s*(\d+)", stem)

    if match:
        return int(match.group(1))

    return 0


def _is_b_sequence(file_name):
    lower = str(file_name).lower()
    return "chronoa_b" in lower or "_b_" in lower


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

    if condition == "PBS+NaNO3":
        prefix = re.sub(r"[^A-Za-z0-9]+", "_", _sequence_prefix(file_name)).strip("_")
        return prefix or "PBS_NaNO3_stitched_sequence"

    return _sequence_prefix(file_name)


def _parse_dta_entry(entry, electrode_area_cm2=0.283, reference_offset_v=-1.0, ph_value=0.0):
    raw = _read_dta_curve(entry["path"])
    columns = list(raw.columns)
    time_col = _find_column(columns, ["T", "Time", "time_s", "T_s"], raw)
    potential_col = _find_column(columns, ["Vf", "E", "Potential", "V"], raw)
    current_col = _find_column(columns, ["Im", "I", "Current", "current_A"], raw)

    if not time_col or not potential_col or not current_col:
        raise ValueError(f"{entry['original_name']}: could not find time, potential, and current columns. Columns: {columns}")

    time_values = _to_number(raw[time_col])
    current_values = _to_number(raw[current_col])
    potential_values = _to_number(raw[potential_col])
    local_time_min = time_values if "min" in str(time_col).lower() else time_values / 60.0

    if "ma" in str(current_col).lower() and "/a" not in str(current_col).lower():
        j_values = current_values / electrode_area_cm2
    else:
        j_values = current_values * 1000.0 / electrode_area_cm2

    name = entry["original_name"]

    return pd.DataFrame({
        "local_time_min": local_time_min,
        "E_RHE": potential_values + reference_offset_v + 0.0591 * ph_value,
        "j_mA_cm2": j_values,
        "condition": _infer_condition(name),
        "dataset_label": _dataset_label(name),
        "source_file": name,
        "sequence_number": _sequence_number(name),
        "is_b_sequence": _is_b_sequence(name),
        "sequence_prefix": _sequence_prefix(name)
    }).dropna(subset=["local_time_min", "j_mA_cm2", "condition", "dataset_label"])


def _stitch_dta_segments(segments):
    if not segments:
        raise ValueError("No valid DTA segments were available for stitching.")

    raw = pd.concat(segments, ignore_index=True)
    stitched = []

    for (condition, dataset_label), group in raw.groupby(["condition", "dataset_label"], sort=False):
        pieces = [piece.copy() for _, piece in group.groupby("source_file", sort=False)]
        pieces = sorted(
            pieces,
            key=lambda frame: (
                int(frame["is_b_sequence"].iloc[0]),
                int(frame["sequence_number"].iloc[0]),
                str(frame["source_file"].iloc[0])
            )
        )
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

    combined = pd.concat(stitched, ignore_index=True).drop(columns=["local_time_min"], errors="ignore")
    return combined


def _average_dta_sequences(combined):
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
            averaged_parts.append(pd.DataFrame({
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
            }))
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

            values.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
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

        averaged_parts.append(pd.DataFrame({
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
            "x_grid_method": "overlap"
        }))

    if not averaged_parts:
        raise ValueError("No averaged DTA data was created.")

    return pd.concat(averaged_parts, ignore_index=True)


def _summarize_by_condition(df, x_col, y_col):
    summaries = []

    for condition, group in df.groupby("condition", sort=False):
        x = _to_number(group[x_col]).dropna() if x_col in group.columns else pd.Series(dtype=float)
        y = _to_number(group[y_col]).dropna() if y_col in group.columns else pd.Series(dtype=float)

        row = {
            "condition": str(condition),
            "rows": int(len(group)),
            "x_min": float(x.min()) if not x.empty else None,
            "x_max": float(x.max()) if not x.empty else None,
            "y_min": float(y.min()) if not y.empty else None,
            "y_max": float(y.max()) if not y.empty else None
        }

        if "saturated_or_clipped" in group.columns:
            row["saturated_or_clipped_points"] = int(group["saturated_or_clipped"].fillna(False).sum())

        summaries.append(row)

    return summaries


def _recommended_group_column(df):
    if "condition" in df.columns and df["condition"].astype(str).nunique(dropna=True) > 1:
        return "condition"

    return "none"


def _recommended_plot_type(x_column):
    return "line" if _looks_like_ordered_x(x_column) else "scatter"


def _recommended_plot_mapping(df, x_column, y_column):
    return {
        "x_column": x_column,
        "y_column": y_column,
        "group_column": _recommended_group_column(df),
        "plot_type": _recommended_plot_type(x_column)
    }


def _wants_plots(parameters):
    if not isinstance(parameters, dict):
        return False

    if parameters.get("generate_plots") is not None:
        return bool(parameters.get("generate_plots"))

    text = _text(parameters.get("user_request")).lower()

    if any(term in text for term in ["no plot", "without plot", "do not plot", "don't plot"]):
        return False

    return any(term in text for term in ["plot", "figure", "chart", "graph", "visual"])


def _max_plot_count(parameters):
    try:
        value = int(parameters.get("max_plots", 1))
    except (TypeError, ValueError):
        value = 1

    return max(0, min(value, 8))


def _is_auxiliary_numeric_column(column):
    lower = str(column).strip().lower()
    exact = {
        "n",
        "n_replicates",
        "sequence_number",
        "source_index",
        "sequence_index",
        "x_value",
        "y_value"
    }

    if lower in exact:
        return True

    return any(term in lower for term in ["_std", "_sem", "_min", "_max", "replicate", "index"])


def _plot_mappings_for_df(df, base_mapping, max_plots):
    if max_plots <= 0:
        return []

    base_mapping = dict(base_mapping or {})
    mappings = [base_mapping]

    if max_plots <= 1:
        return mappings[:max_plots]

    x_column = base_mapping.get("x_column")
    base_y = base_mapping.get("y_column")
    group_column = base_mapping.get("group_column", "none")

    for column in _numeric_columns(df):
        if column in {x_column, base_y, group_column}:
            continue

        if _is_auxiliary_numeric_column(column):
            continue

        mapping = {
            "x_column": x_column,
            "y_column": column,
            "group_column": group_column,
            "plot_type": _recommended_plot_type(x_column)
        }
        mappings.append(mapping)

        if len(mappings) >= max_plots:
            break

    return mappings[:max_plots]


def _generate_workflow_plots(context, plot_sources, parameters):
    if not _wants_plots(parameters):
        return [], []

    create_workflow_plot = context.get("create_workflow_plot")

    if not callable(create_workflow_plot):
        return [], ["Workflow plot generation is unavailable."]

    max_plots = _max_plot_count(parameters)
    plot_urls = []
    errors = []

    for source in plot_sources:
        dataset = source.get("dataset")
        df = source.get("df")
        base_mapping = source.get("mapping")

        if not isinstance(dataset, dict) or df is None:
            continue

        dataset_id = dataset.get("dataset_id")

        if not dataset_id:
            continue

        remaining = max_plots - len(plot_urls)

        if remaining <= 0:
            break

        for mapping in _plot_mappings_for_df(df, base_mapping, remaining):
            try:
                plot_urls.append(create_workflow_plot(dataset_id, mapping))
            except Exception as error:
                errors.append(str(error))

            if len(plot_urls) >= max_plots:
                break

    return plot_urls, errors


def _register_dataset(context, df, prefix, dataset_type, source, description):
    processed_dir = Path(context["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / f"{_timestamp()}_{prefix}.csv"
    df.to_csv(path, index=False)
    dataset = context["register_dataset"](
        file_path=path,
        dataset_type=dataset_type,
        source=source,
        uploaded_by="ai_workflow",
        description=description
    )
    return dataset, path


def _all_same_parser(classifications):
    parsers = {item.get("parser") for item in classifications}
    return len(parsers) == 1


def execute_workflow_plan(plan, context):
    parameters = _plan_parameters(plan)
    entries = _uploaded_entries(context) + _selected_dataset_entries(context, plan)

    if not entries:
        raise ValueError("No uploaded or selected files are available for workflow execution.")

    inspections = [_inspect_entry(entry) for entry in entries]
    classifications = [item["classification"] for item in inspections if item.get("classification")]

    if not classifications:
        raise ValueError("No supported file classification was detected.")

    parsers = [item.get("parser") for item in classifications]
    errors = []
    created_dataset_ids = []
    last_dataset_ids = []
    step_results = []
    plot_sources = []

    if all(parser == "gamry_dta_curve" for parser in parsers):
        segments = []

        for entry in entries:
            try:
                segments.append(_parse_dta_entry(entry))
            except Exception as error:
                errors.append(f"{entry['original_name']}: {error}")

        if not segments:
            raise ValueError("No DTA segments could be parsed. " + "; ".join(errors[:5]))

        combined = _stitch_dta_segments(segments)
        averaged = _average_dta_sequences(combined)
        combined_dataset, _ = _register_dataset(
            context,
            combined,
            "dta_combined",
            "combined_data",
            "Generic parser registry DTA combined dataset",
            "dta_combined"
        )
        averaged_dataset, _ = _register_dataset(
            context,
            averaged,
            "dta_averaged",
            "averaged_replicates",
            "Generic parser registry DTA averaged dataset",
            "dta_averaged"
        )
        created_dataset_ids = [combined_dataset["dataset_id"], averaged_dataset["dataset_id"]]
        last_dataset_ids = [averaged_dataset["dataset_id"]]
        plot_sources.append({
            "dataset": averaged_dataset,
            "df": averaged,
            "mapping": _recommended_plot_mapping(averaged, "global_time_min", "y_mean")
        })

        if _max_plot_count(parameters) > 1:
            plot_sources.append({
                "dataset": combined_dataset,
                "df": combined,
                "mapping": _recommended_plot_mapping(combined, "global_time_min", "j_mA_cm2")
            })

        step_results.append({
            "step_id": 1,
            "action": "gamry_dta_curve_parser",
            "created_dataset_ids": created_dataset_ids,
            "file_inspection": inspections,
            "combined_summary": _summarize_by_condition(combined, "global_time_min", "j_mA_cm2"),
            "averaged_summary": _summarize_by_condition(averaged.rename(columns={"y_mean": "plot_y"}), "global_time_min", "plot_y"),
            "recommended_plot_mapping": _recommended_plot_mapping(averaged, "global_time_min", "y_mean"),
            "errors": errors[:20]
        })

    elif all(parser == "wide_pair_table" for parser in parsers):
        frames = []
        parse_reports = []

        for entry, classification in zip(entries, classifications):
            try:
                frame, parse_report = _parse_wide_pair_table(entry, classification)
                frames.append(frame)
                parse_reports.append(parse_report)
            except Exception as error:
                errors.append(f"{entry['original_name']}: {error}")

        if not frames:
            raise ValueError("No wide-pair table data could be parsed. " + "; ".join(errors[:5]))

        combined = pd.concat(frames, ignore_index=True)
        x_column = parse_reports[0]["x_column"]
        y_column = parse_reports[0]["y_column"]
        dataset, _ = _register_dataset(
            context,
            combined,
            "wide_pair_combined",
            "combined_data",
            "Generic wide-pair table converted to long-format dataset",
            "wide_pair_combined_long_format"
        )
        created_dataset_ids = [dataset["dataset_id"]]
        last_dataset_ids = [dataset["dataset_id"]]
        plot_sources.append({
            "dataset": dataset,
            "df": combined,
            "mapping": _recommended_plot_mapping(combined, x_column, y_column)
        })
        step_results.append({
            "step_id": 1,
            "action": "wide_pair_table_parser",
            "created_dataset_ids": created_dataset_ids,
            "file_inspection": inspections,
            "parse_reports": parse_reports,
            "condition_summary": _summarize_by_condition(combined, x_column, y_column),
            "dataset_columns": list(combined.columns),
            "recommended_plot_mapping": _recommended_plot_mapping(combined, x_column, y_column),
            "errors": errors[:20]
        })

    elif all(parser == "long_table" for parser in parsers):
        frames = []
        parse_reports = []

        for entry, classification in zip(entries, classifications):
            try:
                frame, parse_report = _parse_long_table(entry, classification)
                frames.append(frame)
                parse_reports.append(parse_report)
            except Exception as error:
                errors.append(f"{entry['original_name']}: {error}")

        if not frames:
            raise ValueError("No long-table data could be parsed. " + "; ".join(errors[:5]))

        combined = pd.concat(frames, ignore_index=True)
        x_column = parse_reports[0]["x_column"]
        y_column = parse_reports[0]["y_column"]
        dataset, _ = _register_dataset(
            context,
            combined,
            "long_table_combined",
            "combined_data",
            "Generic long table processed dataset",
            "long_table_combined"
        )
        created_dataset_ids = [dataset["dataset_id"]]
        last_dataset_ids = [dataset["dataset_id"]]
        plot_sources.append({
            "dataset": dataset,
            "df": combined,
            "mapping": _recommended_plot_mapping(combined, x_column, y_column)
        })
        step_results.append({
            "step_id": 1,
            "action": "long_table_parser",
            "created_dataset_ids": created_dataset_ids,
            "file_inspection": inspections,
            "parse_reports": parse_reports,
            "condition_summary": _summarize_by_condition(combined, x_column, y_column),
            "dataset_columns": list(combined.columns),
            "recommended_plot_mapping": _recommended_plot_mapping(combined, x_column, y_column),
            "errors": errors[:20]
        })

    else:
        raise ValueError(
            "Mixed or unsupported data shapes were detected. "
            f"Detected parsers: {parsers}. Review file_inspection diagnostics before processing."
        )

    plot_urls, plot_errors = _generate_workflow_plots(context, plot_sources, parameters)
    errors.extend(plot_errors)

    result = {
        "changed": bool(created_dataset_ids or plot_urls),
        "created_dataset_ids": created_dataset_ids,
        "last_dataset_ids": last_dataset_ids,
        "plot_urls": plot_urls,
        "step_results": step_results,
        "file_inspection": inspections
    }

    return _json_safe(result)
