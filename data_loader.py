from pathlib import Path
import io
import json
import re
import csv

import pandas as pd

from processing_utils import clean_numeric_values as _clean_numeric_values
from processing_utils import normalize_numeric_text


COMMENT_PREFIXES = ("#", "//", "%", "!", ";")
SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".dat", ".dta", ".txt", ".xlsx", ".xls"}


def read_text_file(file_path):
    path = Path(file_path)

    for encoding in ["utf-8-sig", "utf-8", "latin1", "cp1252"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            pass

    return path.read_text(errors="replace")


def clean_column_name(value):
    value = str(value).strip()
    value = value.replace("vs.", "vs")
    value = re.sub(r"[^\w]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")

    if not value:
        return "column"

    return value


def make_unique_columns(columns):
    seen = {}
    output = []

    for column in columns:
        base = clean_column_name(column)
        count = seen.get(base, 0)

        if count == 0:
            output.append(base)
        else:
            output.append(f"{base}_{count + 1}")

        seen[base] = count + 1

    return output


def clean_numeric_values(df):
    return _clean_numeric_values(df)


def split_gamry_line(line):
    parts = [item.strip() for item in line.split("\t")]

    while parts and parts[0] == "":
        parts.pop(0)

    return parts


def make_gamry_columns(labels, units):
    columns = []

    for index, label in enumerate(labels):
        unit = units[index] if index < len(units) else ""
        label_clean = clean_column_name(label)
        unit_clean = clean_column_name(unit)

        if unit_clean.lower() in ["", "column", "bits"]:
            columns.append(label_clean)
        elif unit_clean == "#":
            columns.append(label_clean)
        else:
            columns.append(f"{label_clean}_{unit_clean}")

    return make_unique_columns(columns)


def parse_gamry_dta(file_path):
    text = read_text_file(file_path)
    lines = text.splitlines()

    table_index = None

    for index, line in enumerate(lines):
        parts = split_gamry_line(line)

        if len(parts) >= 3 and parts[0].upper() == "CURVE" and parts[1].upper() == "TABLE":
            table_index = index
            break

    if table_index is None:
        raise ValueError("No CURVE TABLE block was found in this DTA/DAT file.")

    metadata = {}

    for line in lines[:table_index]:
        parts = split_gamry_line(line)

        if len(parts) >= 3 and parts[0]:
            metadata[parts[0]] = parts[2]

    header_index = table_index + 1
    unit_index = table_index + 2

    labels = split_gamry_line(lines[header_index])
    units = split_gamry_line(lines[unit_index])
    columns = make_gamry_columns(labels, units)

    rows = []

    for line in lines[unit_index + 1:]:
        if not line.strip():
            continue

        parts = split_gamry_line(line)

        if len(parts) < len(columns):
            continue

        rows.append(parts[:len(columns)])

    df = pd.DataFrame(rows, columns=columns)
    df = clean_numeric_values(df)

    return df, metadata


def sniff_delimiter(text):
    sample = "\n".join(text.splitlines()[:30])

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,; ")
        return dialect.delimiter
    except csv.Error:
        if "\t" in sample:
            return "\t"

        if ";" in sample:
            return ";"

        if "," in sample:
            return ","

        return None


def read_delimited_candidate(lines, delimiter, header):
    candidate_text = "\n".join(lines)

    if delimiter is None or delimiter == " ":
        kwargs = {"sep": r"\s+", "engine": "python"}
    else:
        kwargs = {"sep": delimiter, "engine": "python"}

    df = pd.read_csv(
        io.StringIO(candidate_text),
        header=header,
        on_bad_lines="skip",
        **kwargs
    )

    if header is None:
        df.columns = [f"column_{index + 1}" for index in range(df.shape[1])]

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if df.shape[1] < 2 or df.empty:
        return None

    df.columns = make_unique_columns(df.columns)
    df = clean_numeric_values(df)

    return df


def score_table(df, start_index, header):
    numeric_columns = len(df.select_dtypes(include="number").columns)
    non_empty_cells = int(df.notna().sum().sum())
    named_columns = sum(1 for column in df.columns if re.search(r"[A-Za-z]", str(column)))
    header_bonus = 1 if header == 0 and named_columns >= max(1, len(df.columns) // 2) else 0

    return (
        numeric_columns,
        header_bonus,
        min(len(df), 5000),
        non_empty_cells,
        -start_index
    )


def parse_generic_text_table(file_path):
    text = read_text_file(file_path)
    lines = text.splitlines()
    useful_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith(COMMENT_PREFIXES):
            continue

        useful_lines.append(line)

    if not useful_lines:
        raise ValueError("No readable table lines were found.")

    best = None
    max_start = min(len(useful_lines), 30)

    for start_index in range(max_start):
        candidate_lines = useful_lines[start_index:]

        if len(candidate_lines) < 2:
            continue

        candidate_text = "\n".join(candidate_lines[:30])
        delimiters = [sniff_delimiter(candidate_text), "\t", ",", ";", " "]
        seen_delimiters = set()

        for delimiter in delimiters:
            delimiter_key = delimiter or "whitespace"

            if delimiter_key in seen_delimiters:
                continue

            seen_delimiters.add(delimiter_key)

            for header in [0, None]:
                try:
                    df = read_delimited_candidate(candidate_lines, delimiter, header)
                except Exception:
                    continue

                if df is None:
                    continue

                score = score_table(df, start_index, header)

                if best is None or score > best[0]:
                    best = (score, df)

    if best is None:
        raise ValueError("No readable tabular data block was found.")

    return best[1], {}


def parse_excel_table(file_path):
    raw = pd.read_excel(file_path, header=None)
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if raw.empty or raw.shape[1] < 2:
        raise ValueError("No readable worksheet table was found.")

    best = None
    max_start = min(len(raw) - 1, 30)

    for start_index in range(max_start):
        headers = raw.iloc[start_index].map(normalize_numeric_text).tolist()
        non_empty_headers = [value for value in headers if pd.notna(value) and str(value).strip()]

        if len(non_empty_headers) < 2:
            continue

        df = raw.iloc[start_index + 1:].copy()
        df.columns = make_unique_columns(headers)
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

        if df.shape[1] < 2 or df.empty:
            continue

        df = clean_numeric_values(df)
        score = score_table(df, start_index, 0)

        if best is None or score > best[0]:
            best = (score, df)

    if best is None:
        df = pd.read_excel(file_path)
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        df.columns = make_unique_columns(df.columns)
        df = clean_numeric_values(df)
        return df, {}

    return best[1], {}


def read_dataset(file_path):
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in [".csv", ".tsv"]:
        df, metadata = parse_generic_text_table(path)
        return df

    if suffix in [".xlsx", ".xls"]:
        df, metadata = parse_excel_table(path)
        return df

    if suffix in [".dta", ".dat"]:
        text = read_text_file(path)

        if "CURVE" in text and "TABLE" in text:
            df, metadata = parse_gamry_dta(path)
        else:
            df, metadata = parse_generic_text_table(path)

        return df

    if suffix == ".txt":
        df, metadata = parse_generic_text_table(path)
        return df

    raise ValueError(f"Unsupported file format: {suffix}")


def inspect_dataset(file_path):
    df = read_dataset(file_path)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns
    }


def save_cleaned_dataset(input_path, output_path):
    df = read_dataset(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
