from pathlib import Path
import io
import json
import re
import csv

import pandas as pd


COMMENT_PREFIXES = ("#", "//", "%", "!", ";")
SUPPORTED_EXTENSIONS = {".csv", ".dat", ".dta", ".txt"}


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
    cleaned = df.copy()

    for column in cleaned.columns:
        series = cleaned[column].astype(str).str.strip()
        series = series.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        numeric_candidate = series.str.replace(",", ".", regex=False)
        numeric_candidate = pd.to_numeric(numeric_candidate, errors="coerce")

        non_empty_count = series.notna().sum()
        numeric_count = numeric_candidate.notna().sum()

        if non_empty_count > 0 and numeric_count / non_empty_count >= 0.8:
            cleaned[column] = numeric_candidate
        else:
            cleaned[column] = series

    return cleaned


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

    useful_text = "\n".join(useful_lines)
    delimiter = sniff_delimiter(useful_text)

    if delimiter is None or delimiter == " ":
        df = pd.read_csv(io.StringIO(useful_text), sep=r"\s+", engine="python")
    else:
        df = pd.read_csv(io.StringIO(useful_text), sep=delimiter, engine="python")

    df.columns = make_unique_columns(df.columns)
    df = clean_numeric_values(df)

    return df, {}


def read_dataset(file_path):
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        df.columns = make_unique_columns(df.columns)
        df = clean_numeric_values(df)
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