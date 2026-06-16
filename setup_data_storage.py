from pathlib import Path
from urllib.request import urlretrieve
from datetime import datetime
import json
import pandas as pd


def create_storage_dirs(project_dir):
    storage_dir = project_dir / "data_storage"
    dirs = {
        "storage": storage_dir,
        "test_data": storage_dir / "test_data",
        "uploaded_data": storage_dir / "uploaded_data",
        "processed_data": storage_dir / "processed_data",
        "metadata": storage_dir / "metadata"
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def download_penguins_dataset(output_path):
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    urlretrieve(url, output_path)


def inspect_csv(csv_path):
    df = pd.read_csv(csv_path)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude="number").columns.tolist()
    }


def update_manifest(manifest_path, dataset_info):
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
    else:
        manifest = {"datasets": []}

    manifest["datasets"] = [
        item for item in manifest["datasets"]
        if item["dataset_id"] != dataset_info["dataset_id"]
    ]

    manifest["datasets"].append(dataset_info)

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)


def main():
    project_dir = Path(__file__).resolve().parent
    dirs = create_storage_dirs(project_dir)

    penguins_path = dirs["test_data"] / "penguins.csv"
    download_penguins_dataset(penguins_path)

    csv_info = inspect_csv(penguins_path)

    dataset_info = {
        "dataset_id": "penguins_test_data",
        "file_name": "penguins.csv",
        "file_path": str(penguins_path),
        "source": "Palmer Penguins",
        "dataset_type": "test_data",
        "uploaded_by": "system",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Mixed numerical and categorical dataset for testing automatic plotting.",
        **csv_info
    }

    manifest_path = dirs["metadata"] / "dataset_manifest.json"
    update_manifest(manifest_path, dataset_info)

    print("Data storage created.")
    print(f"Test dataset saved: {penguins_path}")
    print(f"Manifest saved: {manifest_path}")
    print(f"Rows: {csv_info['rows']}")
    print(f"Columns: {csv_info['columns']}")
    print("Numeric columns:", csv_info["numeric_columns"])
    print("Categorical columns:", csv_info["categorical_columns"])


if __name__ == "__main__":
    main()