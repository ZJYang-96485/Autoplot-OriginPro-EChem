from pathlib import Path

import pandas as pd

from data_loader import read_dataset


def normalize_labels(file_paths, labels):
    if labels is None:
        return [Path(path).stem for path in file_paths]

    if len(labels) == 1 and len(file_paths) > 1:
        return labels * len(file_paths)

    if len(labels) != len(file_paths):
        return [Path(path).stem for path in file_paths]

    output = []

    for index, label in enumerate(labels):
        label = str(label).strip()

        if label:
            output.append(label)
        else:
            output.append(Path(file_paths[index]).stem)

    return output


def protect_existing_column(df, column_name):
    if column_name in df.columns:
        new_name = f"original_{column_name}"
        counter = 1

        while new_name in df.columns:
            counter += 1
            new_name = f"original_{column_name}_{counter}"

        df = df.rename(columns={column_name: new_name})

    return df


def combine_file_paths(file_paths, output_path, condition_labels=None, dataset_labels=None):
    file_paths = [Path(path) for path in file_paths]
    condition_labels = normalize_labels(file_paths, condition_labels)
    dataset_labels = normalize_labels(file_paths, dataset_labels)

    frames = []

    for index, file_path in enumerate(file_paths):
        df = read_dataset(file_path)

        df = protect_existing_column(df, "source_file")
        df = protect_existing_column(df, "source_path")
        df = protect_existing_column(df, "source_index")
        df = protect_existing_column(df, "condition")
        df = protect_existing_column(df, "dataset_label")
        df = protect_existing_column(df, "replicate")

        df.insert(0, "replicate", index + 1)
        df.insert(0, "dataset_label", dataset_labels[index])
        df.insert(0, "condition", condition_labels[index])
        df.insert(0, "source_index", index + 1)
        df.insert(0, "source_path", str(file_path))
        df.insert(0, "source_file", file_path.name)

        frames.append(df)

    if not frames:
        raise ValueError("No files were available to combine.")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    return output_path