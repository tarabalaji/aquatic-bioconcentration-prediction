import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def find_column(dataframe, possible_names):
    normalized_columns = {
        column.lower().replace(" ", "").replace("_", ""): column
        for column in dataframe.columns
    }

    for name in possible_names:
        normalized_name = name.lower().replace(" ", "").replace("_", "")
        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    raise ValueError(
        f"Could not find any of these columns: {possible_names}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def calculate_descriptors(molecule, descriptor_functions):
    values = {}

    for descriptor_name, descriptor_function in descriptor_functions:
        try:
            value = descriptor_function(molecule)
            values[f"descriptor_{descriptor_name}"] = float(value)
        except Exception:
            values[f"descriptor_{descriptor_name}"] = np.nan

    return values


def calculate_fingerprint(molecule, fingerprint_generator, fingerprint_size):
    fingerprint = fingerprint_generator.GetFingerprint(molecule)
    fingerprint_array = np.zeros(fingerprint_size, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)

    return {
        f"fingerprint_{index}": int(value)
        for index, value in enumerate(fingerprint_array)
    }


def generate_features(dataframe, smiles_column, fingerprint_radius, fingerprint_size):
    descriptor_functions = Descriptors.descList
    fingerprint_generator = GetMorganGenerator(
        radius=fingerprint_radius, fpSize=fingerprint_size
    )

    feature_rows = []
    invalid_indices = []

    for index, smiles in dataframe[smiles_column].items():
        molecule = Chem.MolFromSmiles(str(smiles))

        if molecule is None:
            invalid_indices.append(index)
            continue

        descriptor_values = calculate_descriptors(molecule, descriptor_functions)

        fingerprint_values = calculate_fingerprint(
            molecule, fingerprint_generator, fingerprint_size
        )

        feature_rows.append(
            {"_original_index": index, **descriptor_values, **fingerprint_values}
        )

    feature_dataframe = pd.DataFrame(feature_rows)

    return feature_dataframe, invalid_indices


def remove_unusable_descriptors(feature_dataframe):
    descriptor_columns = [
        column
        for column in feature_dataframe.columns
        if column.startswith("descriptor_")
    ]

    fingerprint_columns = [
        column
        for column in feature_dataframe.columns
        if column.startswith("fingerprint_")
    ]

    feature_dataframe[descriptor_columns] = feature_dataframe[
        descriptor_columns
    ].replace([np.inf, -np.inf], np.nan)

    all_missing_descriptors = [
        column
        for column in descriptor_columns
        if feature_dataframe[column].isna().all()
    ]

    feature_dataframe = feature_dataframe.drop(columns=all_missing_descriptors)

    remaining_descriptor_columns = [
        column for column in descriptor_columns if column not in all_missing_descriptors
    ]

    descriptor_medians = feature_dataframe[remaining_descriptor_columns].median()

    feature_dataframe[remaining_descriptor_columns] = feature_dataframe[
        remaining_descriptor_columns
    ].fillna(descriptor_medians)

    constant_descriptors = [
        column
        for column in remaining_descriptor_columns
        if feature_dataframe[column].nunique(dropna=False) <= 1
    ]

    constant_fingerprints = [
        column
        for column in fingerprint_columns
        if feature_dataframe[column].nunique(dropna=False) <= 1
    ]

    feature_dataframe = feature_dataframe.drop(
        columns=constant_descriptors + constant_fingerprints
    )

    return (
        feature_dataframe,
        all_missing_descriptors,
        constant_descriptors,
        constant_fingerprints,
    )


def build_output_dataframe(
    original_dataframe, feature_dataframe, smiles_column, target_column, logkow_column
):
    valid_indices = feature_dataframe["_original_index"].tolist()

    metadata_dataframe = original_dataframe.loc[valid_indices].copy()
    metadata_dataframe = metadata_dataframe.reset_index(drop=True)

    feature_dataframe = feature_dataframe.drop(columns="_original_index").reset_index(
        drop=True
    )

    preferred_metadata = []

    for possible_column in [
        "cas",
        "CAS",
        "cas_number",
        "name",
        "Name",
        smiles_column,
        logkow_column,
        target_column,
    ]:
        if (
            possible_column in metadata_dataframe.columns
            and possible_column not in preferred_metadata
        ):
            preferred_metadata.append(possible_column)

    metadata_dataframe = metadata_dataframe[preferred_metadata]

    return pd.concat([metadata_dataframe, feature_dataframe], axis=1)


def save_feature_groups(dataframe, output_directory):
    descriptor_columns = [
        column for column in dataframe.columns if column.startswith("descriptor_")
    ]

    fingerprint_columns = [
        column for column in dataframe.columns if column.startswith("fingerprint_")
    ]

    non_feature_columns = [
        column
        for column in dataframe.columns
        if column not in descriptor_columns + fingerprint_columns
    ]

    dataframe[non_feature_columns + descriptor_columns].to_csv(
        output_directory / "descriptor_features.csv", index=False
    )

    dataframe[non_feature_columns + fingerprint_columns].to_csv(
        output_directory / "fingerprint_features.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="data/processed/qsar_bcf_clean.csv")

    parser.add_argument("--output", default="data/processed/features.csv")

    parser.add_argument("--radius", type=int, default=2)

    parser.add_argument("--fingerprint-size", type=int, default=2048)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(input_path)

    smiles_column = find_column(
        dataframe, ["canonical_smiles", "canonical smiles", "smiles"]
    )

    target_column = find_column(dataframe, ["logbcf", "log_bcf", "log bcf"])

    logkow_column = find_column(dataframe, ["logkow", "log_kow", "log kow"])

    feature_dataframe, invalid_indices = generate_features(
        dataframe, smiles_column, args.radius, args.fingerprint_size
    )

    (
        feature_dataframe,
        all_missing_descriptors,
        constant_descriptors,
        constant_fingerprints,
    ) = remove_unusable_descriptors(feature_dataframe)

    output_dataframe = build_output_dataframe(
        dataframe, feature_dataframe, smiles_column, target_column, logkow_column
    )

    output_dataframe.to_csv(output_path, index=False)

    save_feature_groups(output_dataframe, output_path.parent)

    descriptor_count = sum(
        column.startswith("descriptor_") for column in output_dataframe.columns
    )

    fingerprint_count = sum(
        column.startswith("fingerprint_") for column in output_dataframe.columns
    )

    report = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_rows": int(len(dataframe)),
        "output_rows": int(len(output_dataframe)),
        "invalid_smiles_rows": int(len(invalid_indices)),
        "descriptor_count": int(descriptor_count),
        "fingerprint_count": int(fingerprint_count),
        "total_feature_count": int(descriptor_count + fingerprint_count + 1),
        "fingerprint_radius": int(args.radius),
        "fingerprint_size_requested": int(args.fingerprint_size),
        "all_missing_descriptors_removed": all_missing_descriptors,
        "constant_descriptors_removed": constant_descriptors,
        "constant_fingerprints_removed": constant_fingerprints,
    }

    report_path = output_path.parent / "feature_engineering_report.json"

    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4)

    print(f"Loaded {len(dataframe)} compounds")
    print(f"Generated features for {len(output_dataframe)} compounds")
    print(f"Generated {descriptor_count} molecular descriptors")
    print(f"Retained {fingerprint_count} Morgan fingerprint bits")
    print(f"Saved combined features to {output_path}")
    print(
        f"Saved descriptor features to {output_path.parent / 'descriptor_features.csv'}"
    )
    print(
        f"Saved fingerprint features to {output_path.parent / 'fingerprint_features.csv'}"
    )
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
