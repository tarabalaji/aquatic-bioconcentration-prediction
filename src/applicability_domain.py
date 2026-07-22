import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy.stats import spearmanr


def normalize_name(name):
    return str(name).lower().replace(" ", "").replace("_", "").replace("-", "")


def find_column(dataframe, possible_names):
    normalized_columns = {
        normalize_name(column): column for column in dataframe.columns
    }

    for name in possible_names:
        normalized_name = normalize_name(name)

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    raise ValueError(
        f"Could not find any of these columns: {possible_names}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_best_model(results_directory):
    path = results_directory / "best_model.json"

    if not path.exists():
        raise FileNotFoundError(
            "results/best_model.json was not found. Run src/evaluate.py first."
        )

    return load_json(path)


def load_predictions(results_directory, best_model):
    path = results_directory / "predictions.csv"

    if not path.exists():
        raise FileNotFoundError(
            "results/predictions.csv was not found. Run src/evaluate.py first."
        )

    dataframe = pd.read_csv(path)

    selected = dataframe[
        (dataframe["model_name"] == best_model["model_name"])
        & (dataframe["feature_group"] == best_model["feature_group"])
    ].copy()

    if selected.empty:
        raise ValueError("Predictions for the best model were not found.")

    return selected.reset_index(drop=True)


def build_fingerprints(smiles_values, radius, fingerprint_size):
    generator = GetMorganGenerator(radius=radius, fpSize=fingerprint_size)

    fingerprints = []
    valid_indices = []
    invalid_indices = []

    for index, smiles in enumerate(smiles_values):
        molecule = Chem.MolFromSmiles(str(smiles))

        if molecule is None:
            fingerprints.append(None)
            invalid_indices.append(index)
            continue

        fingerprints.append(generator.GetFingerprint(molecule))

        valid_indices.append(index)

    return fingerprints, valid_indices, invalid_indices


def calculate_nearest_neighbor_similarity(train_fingerprints, test_fingerprints):
    valid_train_fingerprints = [
        fingerprint for fingerprint in train_fingerprints if fingerprint is not None
    ]

    if not valid_train_fingerprints:
        raise ValueError("No valid training fingerprints were generated.")

    maximum_similarities = []
    nearest_neighbor_indices = []

    for test_fingerprint in test_fingerprints:
        if test_fingerprint is None:
            maximum_similarities.append(np.nan)
            nearest_neighbor_indices.append(None)
            continue

        similarities = DataStructs.BulkTanimotoSimilarity(
            test_fingerprint, valid_train_fingerprints
        )

        nearest_index = int(np.argmax(similarities))
        maximum_similarity = float(similarities[nearest_index])

        maximum_similarities.append(maximum_similarity)

        nearest_neighbor_indices.append(nearest_index)

    return maximum_similarities, nearest_neighbor_indices


def assign_domain_status(similarities, threshold):
    statuses = []

    for similarity in similarities:
        if pd.isna(similarity):
            statuses.append("invalid_structure")
        elif similarity >= threshold:
            statuses.append("in_domain")
        else:
            statuses.append("out_of_domain")

    return statuses


def align_predictions(predictions, test_dataframe, smiles_column):
    if len(predictions) != len(test_dataframe):
        raise ValueError(
            "The number of best-model predictions does not match "
            "the number of rows in the test split."
        )

    output = predictions.copy()

    if smiles_column not in output.columns:
        output[smiles_column] = test_dataframe[smiles_column].reset_index(drop=True)

    return output


def add_nearest_neighbor_information(
    output,
    train_dataframe,
    train_smiles_column,
    nearest_neighbor_indices,
    similarities,
    threshold,
):
    train_smiles = train_dataframe[train_smiles_column].reset_index(drop=True)

    nearest_smiles = []
    nearest_names = []
    nearest_cas_values = []

    name_column = None
    cas_column = None

    for possible_name in ["name", "Name"]:
        if possible_name in train_dataframe.columns:
            name_column = possible_name
            break

    for possible_cas in ["cas", "CAS", "cas_number"]:
        if possible_cas in train_dataframe.columns:
            cas_column = possible_cas
            break

    for nearest_index in nearest_neighbor_indices:
        if nearest_index is None:
            nearest_smiles.append(None)
            nearest_names.append(None)
            nearest_cas_values.append(None)
            continue

        nearest_smiles.append(train_smiles.iloc[nearest_index])

        if name_column is not None:
            nearest_names.append(
                train_dataframe[name_column].reset_index(drop=True).iloc[nearest_index]
            )
        else:
            nearest_names.append(None)

        if cas_column is not None:
            nearest_cas_values.append(
                train_dataframe[cas_column].reset_index(drop=True).iloc[nearest_index]
            )
        else:
            nearest_cas_values.append(None)

    output["maximum_tanimoto_similarity"] = similarities
    output["nearest_training_smiles"] = nearest_smiles
    output["nearest_training_name"] = nearest_names
    output["nearest_training_cas"] = nearest_cas_values
    output["applicability_domain_status"] = assign_domain_status(
        similarities, threshold
    )

    return output


def calculate_domain_summary(output):
    valid = output[output["applicability_domain_status"] != "invalid_structure"].copy()

    summary = (
        valid.groupby("applicability_domain_status", observed=True)
        .agg(
            compound_count=("applicability_domain_status", "size"),
            mean_similarity=("maximum_tanimoto_similarity", "mean"),
            median_similarity=("maximum_tanimoto_similarity", "median"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            maximum_absolute_error=("absolute_error", "max"),
            mean_residual=("residual", "mean"),
        )
        .reset_index()
    )

    return summary


def calculate_similarity_bins(output, bin_count):
    valid = output.dropna(
        subset=["maximum_tanimoto_similarity", "absolute_error"]
    ).copy()

    bins = np.linspace(0, 1, bin_count + 1)

    valid["similarity_range"] = pd.cut(
        valid["maximum_tanimoto_similarity"], bins=bins, include_lowest=True
    )

    grouped = (
        valid.groupby("similarity_range", observed=True)
        .agg(
            compound_count=("maximum_tanimoto_similarity", "size"),
            mean_similarity=("maximum_tanimoto_similarity", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            rmse=(
                "residual",
                lambda values: float(np.sqrt(np.mean(np.square(values)))),
            ),
        )
        .reset_index()
    )

    grouped["similarity_range"] = grouped["similarity_range"].astype(str)

    return grouped


def calculate_threshold_analysis(output, thresholds):
    valid = output.dropna(subset=["maximum_tanimoto_similarity", "absolute_error"])

    records = []

    for threshold in thresholds:
        in_domain = valid[valid["maximum_tanimoto_similarity"] >= threshold]

        out_of_domain = valid[valid["maximum_tanimoto_similarity"] < threshold]

        records.append(
            {
                "threshold": float(threshold),
                "in_domain_count": int(len(in_domain)),
                "out_of_domain_count": int(len(out_of_domain)),
                "in_domain_fraction": float(len(in_domain) / len(valid)),
                "in_domain_mae": (
                    float(in_domain["absolute_error"].mean())
                    if not in_domain.empty
                    else None
                ),
                "out_of_domain_mae": (
                    float(out_of_domain["absolute_error"].mean())
                    if not out_of_domain.empty
                    else None
                ),
            }
        )

    return pd.DataFrame(records)


def calculate_correlation(output):
    valid = output.dropna(subset=["maximum_tanimoto_similarity", "absolute_error"])

    if len(valid) < 3:
        return None, None

    correlation, p_value = spearmanr(
        valid["maximum_tanimoto_similarity"], valid["absolute_error"]
    )

    return float(correlation), float(p_value)


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_similarity_vs_error(output, threshold, figures_directory):
    valid = output.dropna(subset=["maximum_tanimoto_similarity", "absolute_error"])

    plt.figure(figsize=(8, 6))

    plt.scatter(
        valid["maximum_tanimoto_similarity"], valid["absolute_error"], alpha=0.7
    )

    plt.axvline(threshold, linestyle="--", label=f"Domain threshold = {threshold:.2f}")

    plt.xlabel("Maximum Tanimoto Similarity to Training Set")

    plt.ylabel("Absolute Prediction Error")

    plt.title("Prediction Error Across the Applicability Domain")

    plt.legend()

    save_figure(figures_directory / "similarity_vs_absolute_error.png")


def plot_similarity_distribution(output, threshold, figures_directory):
    valid = output["maximum_tanimoto_similarity"].dropna()

    plt.figure(figsize=(8, 6))

    plt.hist(valid, bins=20, edgecolor="black")

    plt.axvline(threshold, linestyle="--", label=f"Domain threshold = {threshold:.2f}")

    plt.xlabel("Maximum Tanimoto Similarity to Training Set")

    plt.ylabel("Number of Test Compounds")

    plt.title("Test-Set Structural Similarity Distribution")

    plt.legend()

    save_figure(figures_directory / "applicability_domain_distribution.png")


def plot_error_by_domain(output, figures_directory):
    valid = output[
        output["applicability_domain_status"].isin(["in_domain", "out_of_domain"])
    ].copy()

    grouped = valid.groupby("applicability_domain_status", observed=True)[
        "absolute_error"
    ].mean()

    ordered_labels = [
        label for label in ["in_domain", "out_of_domain"] if label in grouped.index
    ]

    values = [grouped[label] for label in ordered_labels]

    labels = [label.replace("_", " ").title() for label in ordered_labels]

    plt.figure(figsize=(7, 6))

    plt.bar(labels, values)

    plt.ylabel("Mean Absolute Error")

    plt.xlabel("Applicability Domain Status")

    plt.title("Prediction Error Inside and Outside the Domain")

    save_figure(figures_directory / "error_by_applicability_domain.png")


def plot_binned_similarity_error(similarity_bins, figures_directory):
    valid = similarity_bins.dropna(subset=["mean_absolute_error"])

    positions = np.arange(len(valid))

    plt.figure(figsize=(10, 6))

    plt.bar(positions, valid["mean_absolute_error"])

    plt.xticks(positions, valid["similarity_range"], rotation=35, ha="right")

    plt.xlabel("Maximum Tanimoto Similarity Range")

    plt.ylabel("Mean Absolute Error")

    plt.title("Prediction Error by Structural Similarity Range")

    save_figure(figures_directory / "error_by_similarity_range.png")


def plot_threshold_analysis(threshold_analysis, figures_directory):
    valid = threshold_analysis.dropna(subset=["in_domain_mae", "out_of_domain_mae"])

    plt.figure(figsize=(8, 6))

    plt.plot(
        valid["threshold"], valid["in_domain_mae"], marker="o", label="In-domain MAE"
    )

    plt.plot(
        valid["threshold"],
        valid["out_of_domain_mae"],
        marker="o",
        label="Out-of-domain MAE",
    )

    plt.xlabel("Tanimoto Similarity Threshold")

    plt.ylabel("Mean Absolute Error")

    plt.title("Applicability Domain Threshold Sensitivity")

    plt.legend()

    save_figure(figures_directory / "domain_threshold_sensitivity.png")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", default="results/train_split.csv")

    parser.add_argument("--test-data", default="results/test_split.csv")

    parser.add_argument("--results-directory", default="results")

    parser.add_argument("--figures-directory", default="figures")

    parser.add_argument("--threshold", type=float, default=0.40)

    parser.add_argument("--radius", type=int, default=2)

    parser.add_argument("--fingerprint-size", type=int, default=2048)

    parser.add_argument("--similarity-bins", type=int, default=5)

    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")

    train_path = Path(args.train_data)
    test_path = Path(args.test_data)
    results_directory = Path(args.results_directory)
    figures_directory = Path(args.figures_directory)

    results_directory.mkdir(parents=True, exist_ok=True)

    figures_directory.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training split not found: {train_path}. Run src/train.py first."
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Testing split not found: {test_path}. Run src/train.py first."
        )

    best_model = load_best_model(results_directory)

    predictions = load_predictions(results_directory, best_model)

    train_dataframe = pd.read_csv(train_path)

    test_dataframe = pd.read_csv(test_path)

    train_smiles_column = find_column(
        train_dataframe, ["canonical_smiles", "canonical smiles", "smiles"]
    )

    test_smiles_column = find_column(
        test_dataframe, ["canonical_smiles", "canonical smiles", "smiles"]
    )

    predictions = align_predictions(predictions, test_dataframe, test_smiles_column)

    (train_fingerprints, valid_train_indices, invalid_train_indices) = (
        build_fingerprints(
            train_dataframe[train_smiles_column].tolist(),
            args.radius,
            args.fingerprint_size,
        )
    )

    (test_fingerprints, valid_test_indices, invalid_test_indices) = build_fingerprints(
        test_dataframe[test_smiles_column].tolist(), args.radius, args.fingerprint_size
    )

    (similarities, nearest_neighbor_indices) = calculate_nearest_neighbor_similarity(
        train_fingerprints, test_fingerprints
    )

    output = add_nearest_neighbor_information(
        predictions,
        train_dataframe,
        train_smiles_column,
        nearest_neighbor_indices,
        similarities,
        args.threshold,
    )

    domain_summary = calculate_domain_summary(output)

    similarity_bins = calculate_similarity_bins(output, args.similarity_bins)

    thresholds = np.arange(0.20, 0.81, 0.05)

    threshold_analysis = calculate_threshold_analysis(output, thresholds)

    correlation, p_value = calculate_correlation(output)

    predictions_path = results_directory / "applicability_domain_predictions.csv"

    summary_path = results_directory / "applicability_domain_summary.csv"

    bins_path = results_directory / "applicability_domain_similarity_bins.csv"

    threshold_path = results_directory / "applicability_domain_threshold_analysis.csv"

    report_path = results_directory / "applicability_domain_report.json"

    output.to_csv(predictions_path, index=False)

    domain_summary.to_csv(summary_path, index=False)

    similarity_bins.to_csv(bins_path, index=False)

    threshold_analysis.to_csv(threshold_path, index=False)

    in_domain = output[output["applicability_domain_status"] == "in_domain"]

    out_of_domain = output[output["applicability_domain_status"] == "out_of_domain"]

    report = {
        "method": ("nearest-neighbor Morgan fingerprint Tanimoto similarity"),
        "model_name": best_model["model_name"],
        "feature_group": best_model["feature_group"],
        "fingerprint_radius": int(args.radius),
        "fingerprint_size": int(args.fingerprint_size),
        "similarity_threshold": float(args.threshold),
        "training_compounds": int(len(train_dataframe)),
        "test_compounds": int(len(test_dataframe)),
        "valid_training_structures": int(len(valid_train_indices)),
        "invalid_training_structures": int(len(invalid_train_indices)),
        "valid_test_structures": int(len(valid_test_indices)),
        "invalid_test_structures": int(len(invalid_test_indices)),
        "in_domain_count": int(len(in_domain)),
        "out_of_domain_count": int(len(out_of_domain)),
        "in_domain_fraction": float(len(in_domain) / len(output)),
        "mean_test_similarity": float(output["maximum_tanimoto_similarity"].mean()),
        "median_test_similarity": float(output["maximum_tanimoto_similarity"].median()),
        "in_domain_mae": (
            float(in_domain["absolute_error"].mean()) if not in_domain.empty else None
        ),
        "out_of_domain_mae": (
            float(out_of_domain["absolute_error"].mean())
            if not out_of_domain.empty
            else None
        ),
        "similarity_error_spearman_correlation": (correlation),
        "similarity_error_p_value": (p_value),
    }

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    plot_similarity_vs_error(output, args.threshold, figures_directory)

    plot_similarity_distribution(output, args.threshold, figures_directory)

    plot_error_by_domain(output, figures_directory)

    plot_binned_similarity_error(similarity_bins, figures_directory)

    plot_threshold_analysis(threshold_analysis, figures_directory)

    print(f"Best model: {best_model['model_name']} using {best_model['feature_group']}")

    print(f"Similarity threshold: {args.threshold:.2f}")

    print(f"In-domain compounds: {len(in_domain)}")

    print(f"Out-of-domain compounds: {len(out_of_domain)}")

    if not in_domain.empty:
        print(f"In-domain MAE: {in_domain['absolute_error'].mean():.4f}")

    if not out_of_domain.empty:
        print(f"Out-of-domain MAE: {out_of_domain['absolute_error'].mean():.4f}")

    if correlation is not None:
        print(f"Similarity-error Spearman correlation: {correlation:.4f}")

        print(f"Correlation p-value: {p_value:.4g}")

    print(f"Saved applicability-domain predictions to {predictions_path}")

    print(f"Saved applicability-domain report to {report_path}")


if __name__ == "__main__":
    main()
