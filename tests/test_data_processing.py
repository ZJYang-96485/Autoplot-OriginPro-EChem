from pathlib import Path
import tempfile
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conversion_tools import convert_dataset_variables
from data_loader import read_dataset
from sequential_tools import combine_sequential_file_paths
from workflow_executor import execute_workflow_plan


def write_text(path, text):
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_metadata_semicolon_csv_with_comma_decimals(tmp_path):
    path = tmp_path / "instrument_export.csv"
    write_text(
        path,
        """
        Instrument: potentiostat
        Operator: test
        Time / s;Current / A;Label
        0;"1,0E-3 A";start
        1;"1,1E-3 A";middle
        2;"1,2E-3 A";end
        """,
    )

    df = read_dataset(path)

    assert list(df.columns) == ["Time_s", "Current_A", "Label"]
    assert len(df) == 3
    assert abs(float(df["Current_A"].iloc[0]) - 0.001) < 1e-12


def test_headerless_numeric_table_keeps_first_row(tmp_path):
    path = tmp_path / "headerless.txt"
    write_text(
        path,
        """
        0 1
        1 2
        2 3
        """,
    )

    df = read_dataset(path)

    assert list(df.columns) == ["column_1", "column_2"]
    assert len(df) == 3
    assert df["column_1"].tolist() == [0, 1, 2]


def test_excel_table_after_metadata_rows(tmp_path):
    path = tmp_path / "spreadsheet.xlsx"
    raw = pd.DataFrame([
        ["Instrument", "Example", None],
        ["Operator", "Test", None],
        ["Time / s", "Current / A", "Condition"],
        [0, 0.001, "A"],
        [1, 0.0011, "A"],
    ])
    raw.to_excel(path, header=False, index=False)

    df = read_dataset(path)

    assert list(df.columns) == ["Time_s", "Current_A", "Condition"]
    assert len(df) == 2
    assert abs(float(df["Current_A"].iloc[1]) - 0.0011) < 1e-12


def test_sequential_stitching_resolves_aliases_and_sequence_suffixes(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    file_paths = []

    for index in [1, 2]:
        path = input_dir / f"sample_step_{index}.csv"
        write_text(
            path,
            """
            Time / s,Vf,Im
            0,0.20,0.001
            1,0.21,0.0011
            """,
        )
        file_paths.append(path)

    output_path = tmp_path / "stitched.csv"
    combine_sequential_file_paths(
        file_paths=file_paths,
        output_path=output_path,
        condition_label="A",
        dataset_label="A",
        time_col="T_s",
        step_variable_col="Vf_V_vs_Ref",
        step_duration_s=10,
    )

    stitched = pd.read_csv(output_path)

    assert stitched["sequence_index"].drop_duplicates().tolist() == [1, 2]
    assert stitched["step_variable_name"].drop_duplicates().tolist() == ["Vf"]
    assert stitched["global_time_s"].max() == 11


def test_conversion_resolves_common_echem_aliases(tmp_path):
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "converted.csv"
    write_text(
        input_path,
        """
        Vf,Im
        0.20,0.001
        0.21,0.0011
        """,
    )

    convert_dataset_variables(
        input_path=input_path,
        output_path=output_path,
        potential_col="Vf_V_vs_Ref",
        current_col="Im_A",
        reference_offset_v=0.1,
        electrode_area_cm2=0.5,
    )

    converted = pd.read_csv(output_path)

    assert "E_RHE" in converted.columns
    assert "j_mA_cm2" in converted.columns
    assert abs(float(converted["E_RHE"].iloc[0]) - 0.3) < 1e-12
    assert abs(float(converted["j_mA_cm2"].iloc[0]) - 2.0) < 1e-12


def test_workflow_skips_unprocessable_files_when_valid_files_exist(tmp_path):
    valid_path = tmp_path / "valid.csv"
    invalid_path = tmp_path / "notes.txt"
    output_dir = tmp_path / "processed"
    datasets = {}
    counter = {"value": 0}

    write_text(
        valid_path,
        """
        time,current,condition
        0,1,A
        1,2,A
        2,3,A
        """,
    )
    write_text(
        invalid_path,
        """
        These are experiment notes.
        There is no table here.
        """,
    )

    def register_dataset(file_path, dataset_type, source, uploaded_by, description):
        counter["value"] += 1
        dataset_id = f"ds_{counter['value']}"
        dataset = {
            "dataset_id": dataset_id,
            "file_path": str(file_path),
            "file_name": Path(file_path).name,
            "dataset_type": dataset_type,
            "source": source,
            "uploaded_by": uploaded_by,
            "description": description,
        }
        datasets[dataset_id] = dataset
        return dataset

    result = execute_workflow_plan(
        plan={"steps": [{"parameters": {"generate_plots": False}}]},
        context={
            "uploaded_files": [
                {"original_name": valid_path.name, "saved_path": str(valid_path)},
                {"original_name": invalid_path.name, "saved_path": str(invalid_path)},
            ],
            "processed_data_dir": output_dir,
            "register_dataset": register_dataset,
            "get_dataset": lambda dataset_id: datasets.get(dataset_id),
        },
    )

    assert result["changed"] is True
    assert len(result["created_dataset_ids"]) == 1


def write_dta(path, current_offset):
    lines = [
        "EXPLAIN\tSynthetic verification DTA",
        "CURVE\tTABLE",
        "\tT\tVf\tIm",
        "\ts\tV\tA",
    ]

    for index in range(40):
        current = 0.001 + current_offset + index * 0.00001
        lines.append(f"\t{index}\t0.20\t{current:.8f}")

    path.write_text("\n".join(lines), encoding="utf-8")


def test_workflow_dta_numbered_steps_create_steady_state_summary(tmp_path):
    output_dir = tmp_path / "processed"
    uploaded_files = []
    datasets = {}
    counter = {"value": 0}

    for name, offset in [
        ("CrCoNi_0rpm_0.1MNaCl_5.8pH_passv_corrCHRONOA_1.DTA", 0.0),
        ("CrCoNi_0rpm_0.1MNaCl_5.8pH_passv_corrCHRONOA_2.DTA", 0.0002),
        ("CrCoNi_0rpm_0.1MNaCl_5.8pH_passv_corrCHRONOA_condi.DTA", 0.0004),
    ]:
        path = tmp_path / name
        write_dta(path, offset)
        uploaded_files.append({"original_name": name, "saved_path": str(path)})

    def register_dataset(file_path, dataset_type, source, uploaded_by, description):
        counter["value"] += 1
        dataset_id = f"dta_ds_{counter['value']}"
        dataset = {
            "dataset_id": dataset_id,
            "file_path": str(file_path),
            "file_name": Path(file_path).name,
            "dataset_type": dataset_type,
            "source": source,
            "uploaded_by": uploaded_by,
            "description": description,
        }
        datasets[dataset_id] = dataset
        return dataset

    result = execute_workflow_plan(
        plan={"steps": [{"parameters": {"generate_plots": False, "max_plots": 0}}]},
        context={
            "uploaded_files": uploaded_files,
            "processed_data_dir": output_dir,
            "register_dataset": register_dataset,
            "get_dataset": lambda dataset_id: datasets.get(dataset_id),
        },
    )

    steady_dataset = next(dataset for dataset in datasets.values() if dataset["description"] == "dta_steady_state_summary")
    combined_dataset = next(dataset for dataset in datasets.values() if dataset["description"] == "dta_combined")
    steady = pd.read_csv(steady_dataset["file_path"])
    combined = pd.read_csv(combined_dataset["file_path"])

    assert result["changed"] is True
    assert steady["sequence_number"].astype(int).tolist() == [1, 2]
    assert not steady["source_file"].str.contains("condi", case=False).any()
    assert "local_time_min" in combined.columns
    assert "global_time_min" in combined.columns


def run_all():
    tests = [
        test_metadata_semicolon_csv_with_comma_decimals,
        test_headerless_numeric_table_keeps_first_row,
        test_excel_table_after_metadata_rows,
        test_sequential_stitching_resolves_aliases_and_sequence_suffixes,
        test_conversion_resolves_common_echem_aliases,
        test_workflow_skips_unprocessable_files_when_valid_files_exist,
        test_workflow_dta_numbered_steps_create_steady_state_summary,
    ]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        for test in tests:
            test_path = root / test.__name__
            test_path.mkdir()
            test(test_path)


if __name__ == "__main__":
    run_all()
    print("data processing checks passed")
