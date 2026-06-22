from pathlib import Path
import math

import numpy as np
import pandas as pd



def _to_numeric_series(values):
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    if isinstance(values, pd.Series):
        cleaned = (
            values.astype(str)
            .str.strip()
            .str.replace("\u2212", "-", regex=False)
            .str.replace(",", ".", regex=False)
        )
        cleaned = cleaned.mask(cleaned.str.lower().isin(["", "none", "nan", "na", "n/a"]))
        return pd.to_numeric(cleaned, errors="coerce")

    return pd.to_numeric(values, errors="coerce")


def _as_text(value):
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"", "none", "null", "nan", "na", "n/a"}:
        return ""

    return text


def _ensure_column(df, column, role):
    if column not in df.columns:
        raise ValueError(f"{role} column not found: {column}")


def _choose_replicate_column(df, replicate_col):
    replicate_col = _as_text(replicate_col)

    if replicate_col and replicate_col in df.columns:
        return replicate_col

    for candidate in ["dataset_label", "source_file", "source_index", "replicate"]:
        if candidate in df.columns:
            return candidate

    df["replicate"] = np.arange(len(df)).astype(str)

    return "replicate"


def _prepare_dataframe(input_data, x_col, y_col, condition_col, replicate_col):
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        df = pd.read_csv(input_data)

    _ensure_column(df, x_col, "X")
    _ensure_column(df, y_col, "Y")
    _ensure_column(df, condition_col, "Condition")

    replicate_col = _choose_replicate_column(df, replicate_col)

    df[condition_col] = df[condition_col].map(_as_text)
    df[replicate_col] = df[replicate_col].map(_as_text)

    df.loc[df[condition_col] == "", condition_col] = "Condition"
    df.loc[df[replicate_col] == "", replicate_col] = "Replicate"

    df[x_col] = _to_numeric_series(df[x_col])
    df[y_col] = _to_numeric_series(df[y_col])

    df = df.dropna(subset=[x_col, y_col, condition_col, replicate_col])

    if df.empty:
        raise ValueError("No valid rows remain after cleaning X, Y, condition, and replicate columns.")

    return df, replicate_col


def _replicate_curves(df, x_col, y_col, condition_col, replicate_col):
    curves = {}

    for (condition, replicate), group in df.groupby([condition_col, replicate_col], sort=False):
        curve = (
            group[[x_col, y_col]]
            .dropna()
            .groupby(x_col, as_index=False)[y_col]
            .mean()
            .sort_values(x_col)
        )

        if len(curve) >= 2:
            curves.setdefault(condition, {})[replicate] = curve

    if not curves:
        raise ValueError("No replicate curves contain at least two valid points.")

    return curves


def _linear_grid(replicates, x_grid_method, grid_points):
    mins = [float(curve.iloc[0, 0]) for curve in replicates]
    maxs = [float(curve.iloc[-1, 0]) for curve in replicates]

    if x_grid_method == "union":
        start = min(mins)
        end = max(maxs)
    else:
        start = max(mins)
        end = min(maxs)

    if not math.isfinite(start) or not math.isfinite(end) or start >= end:
        raise ValueError("Cannot create a valid interpolation grid. Check whether replicate X ranges overlap.")

    return np.linspace(start, end, int(grid_points))


def _reference_grid(replicates):
    reference = max(replicates, key=len)
    start = max(float(curve.iloc[0, 0]) for curve in replicates)
    end = min(float(curve.iloc[-1, 0]) for curve in replicates)

    grid = reference.iloc[:, 0].to_numpy(dtype=float)
    grid = grid[(grid >= start) & (grid <= end)]

    if len(grid) < 2:
        raise ValueError("Reference grid has fewer than two points inside the common X range.")

    return grid


def _interpolate_curve(curve, grid):
    x = curve.iloc[:, 0].to_numpy(dtype=float)
    y = curve.iloc[:, 1].to_numpy(dtype=float)

    interpolated = np.interp(grid, x, y)
    interpolated[(grid < x.min()) | (grid > x.max())] = np.nan

    return interpolated


def _summarize_matrix(values):
    n = np.sum(~np.isnan(values), axis=1)
    mean = np.nanmean(values, axis=1)
    std = np.nanstd(values, axis=1, ddof=1)
    sem = std / np.sqrt(n)

    std[n <= 1] = 0.0
    sem[n <= 1] = 0.0

    return mean, std, sem, np.nanmin(values, axis=1), np.nanmax(values, axis=1), n


def _average_exact(df, x_col, y_col, condition_col, replicate_col, x_round_decimals, min_replicates):
    working = df[[condition_col, replicate_col, x_col, y_col]].copy()
    working["_x_key"] = working[x_col].round(int(x_round_decimals))

    per_replicate = (
        working
        .groupby([condition_col, replicate_col, "_x_key"], as_index=False)
        .agg(
            x_value=(x_col, "mean"),
            y_value=(y_col, "mean")
        )
    )

    rows = []

    for (condition, x_key), group in per_replicate.groupby([condition_col, "_x_key"], sort=False):
        values = group["y_value"].to_numpy(dtype=float)
        n = len(values)

        if n < min_replicates:
            continue

        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n > 1 else 0.0

        rows.append({
            condition_col: condition,
            x_col: float(group["x_value"].mean()),
            "y_mean": float(np.mean(values)),
            "y_std": std,
            "y_sem": sem,
            "y_min": float(np.min(values)),
            "y_max": float(np.max(values)),
            "n_replicates": int(n),
            "replicate_names": ", ".join(map(str, group[replicate_col].tolist()))
        })

    output = pd.DataFrame(rows)

    if output.empty:
        raise ValueError("No averaged rows were produced. Try interpolation or lower min_replicates.")

    return output.sort_values([condition_col, x_col]).reset_index(drop=True)


def _average_interpolate(df, x_col, y_col, condition_col, replicate_col, x_grid_method, grid_points, min_replicates):
    curves = _replicate_curves(df, x_col, y_col, condition_col, replicate_col)
    rows = []

    for condition, replicate_map in curves.items():
        replicates = list(replicate_map.values())

        if len(replicates) < min_replicates:
            continue

        if x_grid_method == "reference":
            grid = _reference_grid(replicates)
        else:
            grid = _linear_grid(replicates, x_grid_method, grid_points)

        matrix = np.vstack([_interpolate_curve(curve, grid) for curve in replicates]).T
        mean, std, sem, ymin, ymax, n = _summarize_matrix(matrix)
        replicate_names = ", ".join(map(str, replicate_map.keys()))

        for index, x_value in enumerate(grid):
            if n[index] < min_replicates:
                continue

            rows.append({
                condition_col: condition,
                x_col: float(x_value),
                "y_mean": float(mean[index]),
                "y_std": float(std[index]),
                "y_sem": float(sem[index]),
                "y_min": float(ymin[index]),
                "y_max": float(ymax[index]),
                "n_replicates": int(n[index]),
                "replicate_names": replicate_names
            })

    output = pd.DataFrame(rows)

    if output.empty:
        raise ValueError("No averaged rows were produced. Check X overlap, grid settings, and min_replicates.")

    return output.sort_values([condition_col, x_col]).reset_index(drop=True)


def average_condition_replicates(
    input_path,
    output_path,
    x_col,
    y_col,
    condition_col="condition",
    replicate_col="dataset_label",
    method="interpolate",
    x_grid_method="overlap",
    grid_points=500,
    x_round_decimals=6,
    min_replicates=1
):
    method = _as_text(method).lower() or "interpolate"
    x_grid_method = _as_text(x_grid_method).lower() or "overlap"

    if method not in {"exact", "interpolate"}:
        raise ValueError("method must be 'exact' or 'interpolate'.")

    if x_grid_method not in {"overlap", "union", "reference"}:
        raise ValueError("x_grid_method must be 'overlap', 'union', or 'reference'.")

    grid_points = int(grid_points)

    if grid_points < 2:
        raise ValueError("grid_points must be at least 2.")

    min_replicates = int(min_replicates)

    if min_replicates < 1:
        raise ValueError("min_replicates must be at least 1.")

    df, replicate_col = _prepare_dataframe(
        input_data=input_path,
        x_col=x_col,
        y_col=y_col,
        condition_col=condition_col,
        replicate_col=replicate_col
    )

    if method == "exact":
        averaged = _average_exact(
            df=df,
            x_col=x_col,
            y_col=y_col,
            condition_col=condition_col,
            replicate_col=replicate_col,
            x_round_decimals=x_round_decimals,
            min_replicates=min_replicates
        )
    else:
        averaged = _average_interpolate(
            df=df,
            x_col=x_col,
            y_col=y_col,
            condition_col=condition_col,
            replicate_col=replicate_col,
            x_grid_method=x_grid_method,
            grid_points=grid_points,
            min_replicates=min_replicates
        )

    averaged["source_x_col"] = x_col
    averaged["source_y_col"] = y_col
    averaged["averaging_method"] = method
    averaged["x_grid_method"] = x_grid_method

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    averaged.to_csv(output_path, index=False)

    condition_summaries = []

    for condition, group in averaged.groupby(condition_col, sort=False):
        condition_summaries.append({
            "condition": condition,
            "rows": int(len(group)),
            "x_min": float(group[x_col].min()),
            "x_max": float(group[x_col].max()),
            "mean_n_replicates": float(group["n_replicates"].mean())
        })

    return {
        "output_path": str(output_path),
        "rows": int(len(averaged)),
        "columns": list(averaged.columns),
        "conditions": list(averaged[condition_col].drop_duplicates()),
        "condition_summaries": condition_summaries,
        "x_col": x_col,
        "source_y_col": y_col,
        "y_mean_col": "y_mean",
        "y_std_col": "y_std",
        "y_sem_col": "y_sem",
        "condition_col": condition_col,
        "replicate_col": replicate_col,
        "method": method,
        "x_grid_method": x_grid_method
    }
