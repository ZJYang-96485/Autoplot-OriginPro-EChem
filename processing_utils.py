from pathlib import Path
import re

import pandas as pd


MISSING_TOKENS = {
    "",
    "-",
    "--",
    "none",
    "null",
    "nan",
    "na",
    "n/a",
    "#n/a",
}

SEQUENCE_PATTERNS = [
    r"#\s*(\d+)",
    r"(?:chronoa|step|seq|sequence|scan|cycle)[_\-\s]*(\d+)$",
    r"(?:^|[_\-\s])(\d+)$",
]


def as_text(value):
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in MISSING_TOKENS:
        return ""

    return text


def normalize_column_key(value):
    text = as_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def normalize_numeric_text(value):
    if value is None:
        return pd.NA

    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass

    text = str(value).strip().strip('"').strip("'")

    if text.lower() in MISSING_TOKENS:
        return pd.NA

    text = (
        text.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
        .strip()
    )
    negative_parentheses = False

    if re.fullmatch(r"\(\s*.+\s*\)", text):
        negative_parentheses = True
        text = text[1:-1].strip()

    text = re.sub(r"^[<>~=]+\s*", "", text)
    match = re.match(
        r"^([+-]?(?:\d[\d\s,\.]*|[,.]\d+)(?:[eE][+-]?\d+)?)(.*)$",
        text,
    )

    if not match:
        return text

    suffix = match.group(2).strip()

    if suffix and not re.fullmatch(r"[%A-Za-zµμΩ°/_^\-0-9]*", suffix):
        return text

    number_text = re.sub(r"\s+", "", match.group(1))
    number_text = normalize_number_separators(number_text)

    if negative_parentheses and not number_text.startswith("-"):
        number_text = f"-{number_text}"

    return number_text


def normalize_number_separators(text):
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            return text.replace(".", "").replace(",", ".")

        return text.replace(",", "")

    if "," in text:
        if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(?:[eE][+-]?\d+)?", text):
            return text.replace(",", "")

        return text.replace(",", ".")

    if text.count(".") > 1 and re.fullmatch(r"[+-]?\d{1,3}(\.\d{3})+(?:[eE][+-]?\d+)?", text):
        return text.replace(".", "")

    return text


def to_numeric_series(values):
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    if isinstance(values, pd.Series):
        cleaned = values.map(normalize_numeric_text)
        return pd.to_numeric(cleaned, errors="coerce")

    return pd.to_numeric(normalize_numeric_text(values), errors="coerce")


def clean_numeric_values(df, min_numeric_ratio=0.8):
    cleaned = df.copy()

    for column in cleaned.columns:
        series = cleaned[column]
        normalized = series.map(normalize_numeric_text)
        non_empty_count = normalized.notna().sum()
        numeric_candidate = pd.to_numeric(normalized, errors="coerce")
        numeric_count = numeric_candidate.notna().sum()

        if non_empty_count > 0 and numeric_count / non_empty_count >= min_numeric_ratio:
            cleaned[column] = numeric_candidate
        else:
            cleaned[column] = normalized

    return cleaned


def resolve_column(columns, preferred=None, aliases=None, df=None, min_numeric=0):
    aliases = list(aliases or [])

    if preferred:
        aliases.insert(0, preferred)

    key_to_column = {}

    for column in columns:
        key_to_column.setdefault(normalize_column_key(column), column)

    for alias in aliases:
        key = normalize_column_key(alias)

        if not key:
            continue

        column = key_to_column.get(key)

        if column is not None and _column_has_enough_numeric_values(df, column, min_numeric):
            return column

    for column in columns:
        column_key = normalize_column_key(column)

        if not column_key:
            continue

        for alias in aliases:
            alias_key = normalize_column_key(alias)

            if len(alias_key) < 2:
                continue

            if alias_key in column_key and _column_has_enough_numeric_values(df, column, min_numeric):
                return column

    return ""


def _column_has_enough_numeric_values(df, column, min_numeric):
    if df is None or min_numeric <= 0:
        return True

    if column not in df.columns:
        return False

    return to_numeric_series(df[column]).notna().sum() >= min_numeric


def strip_upload_prefix(file_name):
    stem = Path(str(file_name)).stem
    stem = re.sub(r"^\d{8}_\d{6}_(?:ai_workflow|sequence)_\d{3}_", "", stem)
    stem = re.sub(r"^\d{8}_\d{6}_\d{3}_", "", stem)
    stem = re.sub(r"^\d{8}_\d{6}_", "", stem)
    return stem


def extract_sequence_index(file_name, fallback_index=0, sequence_regex=""):
    stem = strip_upload_prefix(file_name)
    patterns = []

    if sequence_regex:
        patterns.append(sequence_regex)

    patterns.extend(SEQUENCE_PATTERNS)

    for pattern in patterns:
        try:
            match = re.search(pattern, stem, flags=re.IGNORECASE)
        except re.error:
            continue

        if match:
            return int(match.group(1))

    return fallback_index


def sequence_prefix(file_name):
    stem = strip_upload_prefix(file_name)
    stem = re.sub(r"\(\d+\)$", "", stem)
    stem = re.sub(r"#\s*\d+.*$", "", stem)
    stem = re.sub(
        r"((?:chronoa|step|seq|sequence|scan|cycle))[_\-\s]*\d+$",
        r"\1",
        stem,
        flags=re.IGNORECASE,
    )
    stem = re.sub(r"[_\-\s]+\d+$", "", stem)
    stem = re.sub(r"CHRONOA[_-]*B$", "CHRONOA", stem, flags=re.IGNORECASE)
    stem = stem.strip("_- ")
    return stem or Path(str(file_name)).stem
