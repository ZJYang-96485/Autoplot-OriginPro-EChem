from pathlib import Path

from data_loader import read_dataset
from processing_utils import normalize_numeric_text
from processing_utils import resolve_column
from processing_utils import to_numeric_series


POTENTIAL_COLUMN_ALIASES = [
    "step_variable_value",
    "E_RHE",
    "Vf_V_vs_Ref",
    "Vf_V",
    "Vf",
    "E",
    "Potential",
    "Potential_V",
    "voltage",
]
CURRENT_COLUMN_ALIASES = [
    "Im_A",
    "Im",
    "I_A",
    "I",
    "Current",
    "Current_A",
    "current_A",
]


def to_float(value, default=None):
    try:
        return float(normalize_numeric_text(value))
    except (TypeError, ValueError):
        return default


def clean_column_name(name):
    name = str(name).strip()

    if not name:
        return "converted_column"

    name = name.replace(" ", "_")
    name = name.replace("/", "_per_")
    name = name.replace("^", "")
    name = name.replace("-", "_")
    name = name.replace(".", "_")

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


def ensure_unique_column(df, column_name):
    column_name = clean_column_name(column_name)

    if column_name not in df.columns:
        return column_name

    index = 2

    while f"{column_name}_{index}" in df.columns:
        index += 1

    return f"{column_name}_{index}"


def convert_dataset_variables(
    input_path,
    output_path,
    convert_potential=True,
    potential_col="step_variable_value",
    reference_offset_v=0.0,
    ph_value=0.0,
    potential_output_col="E_RHE",
    convert_current=True,
    current_col="Im_A",
    electrode_area_cm2=1.0,
    current_density_output_col="j_mA_cm2"
):
    df = read_dataset(input_path)

    reference_offset_v = to_float(reference_offset_v, 0.0)
    ph_value = to_float(ph_value, 0.0)
    electrode_area_cm2 = to_float(electrode_area_cm2, 1.0)

    if convert_potential:
        resolved_potential_col = resolve_column(
            df.columns,
            preferred=potential_col,
            aliases=POTENTIAL_COLUMN_ALIASES,
            df=df,
            min_numeric=1
        )

        if resolved_potential_col not in df.columns:
            raise ValueError(f"Potential column not found: {potential_col}")

        output_col = ensure_unique_column(df, potential_output_col)
        potential = to_numeric_series(df[resolved_potential_col])

        df[output_col] = potential + reference_offset_v + 0.05916 * ph_value

    if convert_current:
        resolved_current_col = resolve_column(
            df.columns,
            preferred=current_col,
            aliases=CURRENT_COLUMN_ALIASES,
            df=df,
            min_numeric=1
        )

        if resolved_current_col not in df.columns:
            raise ValueError(f"Current column not found: {current_col}")

        if electrode_area_cm2 is None or electrode_area_cm2 <= 0:
            raise ValueError("Electrode area must be larger than 0.")

        output_col = ensure_unique_column(df, current_density_output_col)
        current = to_numeric_series(df[resolved_current_col])

        df[output_col] = current * 1000.0 / electrode_area_cm2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return output_path
