import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42


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


def identify_feature_columns(dataframe):
    descriptor_columns = [
        column for column in dataframe.columns if str(column).startswith("descriptor_")
    ]

    fingerprint_columns = [
        column for column in dataframe.columns if str(column).startswith("fingerprint_")
    ]

    logkow_column = find_column(dataframe, ["logkow", "log_kow", "log kow", "logKOW"])

    feature_groups = {
        "logkow_only": [logkow_column],
        "descriptors_only": descriptor_columns,
        "fingerprints_only": fingerprint_columns,
        "descriptors_logkow": (descriptor_columns + [logkow_column]),
        "combined": (descriptor_columns + fingerprint_columns + [logkow_column]),
    }

    empty_groups = [
        group_name for group_name, columns in feature_groups.items() if not columns
    ]

    if empty_groups:
        raise ValueError(f"No features were found for these groups: {empty_groups}")

    return feature_groups


def generate_scaffold(smiles):
    if pd.isna(smiles):
        return None

    molecule = Chem.MolFromSmiles(str(smiles))

    if molecule is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)

    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return "NO_SCAFFOLD"

    return Chem.MolToSmiles(scaffold, canonical=True)


def assign_scaffolds(dataframe, smiles_column):
    scaffold_values = [generate_scaffold(smiles) for smiles in dataframe[smiles_column]]

    output = dataframe.copy()
    output["murcko_scaffold"] = scaffold_values

    invalid_count = int(output["murcko_scaffold"].isna().sum())

    output = output.dropna(subset=["murcko_scaffold"]).reset_index(drop=True)

    return output, invalid_count


def create_scaffold_split(dataframe, test_fraction, random_state):
    scaffold_groups = {}

    for index, scaffold in enumerate(dataframe["murcko_scaffold"]):
        scaffold_groups.setdefault(scaffold, []).append(index)

    scaffold_sets = list(scaffold_groups.values())

    rng = np.random.default_rng(random_state)

    rng.shuffle(scaffold_sets)

    scaffold_sets.sort(key=len, reverse=True)

    target_test_size = max(1, int(round(len(dataframe) * test_fraction)))

    train_indices = []
    test_indices = []

    for scaffold_indices in scaffold_sets:
        projected_test_size = len(test_indices) + len(scaffold_indices)

        current_difference = abs(len(test_indices) - target_test_size)

        projected_difference = abs(projected_test_size - target_test_size)

        if projected_difference <= current_difference or not test_indices:
            test_indices.extend(scaffold_indices)
        else:
            train_indices.extend(scaffold_indices)

    if not train_indices or not test_indices:
        raise ValueError(
            "The scaffold split produced an empty training or "
            "testing set. Try a different test fraction."
        )

    train_dataframe = dataframe.iloc[sorted(train_indices)].reset_index(drop=True)

    test_dataframe = dataframe.iloc[sorted(test_indices)].reset_index(drop=True)

    train_scaffolds = set(train_dataframe["murcko_scaffold"])

    test_scaffolds = set(test_dataframe["murcko_scaffold"])

    overlap = train_scaffolds.intersection(test_scaffolds)

    if overlap:
        raise RuntimeError(
            "Scaffold leakage detected between training and testing sets."
        )

    return train_dataframe, test_dataframe


def build_models():
    return {
        "linear_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=1,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.05,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "catboost": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=500,
                        learning_rate=0.05,
                        depth=6,
                        loss_function="RMSE",
                        random_seed=RANDOM_STATE,
                        verbose=False,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def train_and_evaluate(
    train_dataframe,
    test_dataframe,
    target_column,
    feature_groups,
    models,
    model_output_directory,
):
    metric_records = []
    prediction_records = []

    model_output_directory.mkdir(parents=True, exist_ok=True)

    y_train = pd.to_numeric(train_dataframe[target_column], errors="coerce")

    y_test = pd.to_numeric(test_dataframe[target_column], errors="coerce")

    valid_train_rows = y_train.notna()
    valid_test_rows = y_test.notna()

    y_train = y_train.loc[valid_train_rows].reset_index(drop=True)

    y_test = y_test.loc[valid_test_rows].reset_index(drop=True)

    for feature_group, feature_columns in feature_groups.items():
        group_directory = model_output_directory / feature_group

        group_directory.mkdir(parents=True, exist_ok=True)

        with open(
            group_directory / "feature_columns.json", "w", encoding="utf-8"
        ) as file:
            json.dump(feature_columns, file, indent=4)

        X_train = (
            train_dataframe.loc[valid_train_rows, feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .reset_index(drop=True)
        )

        X_test = (
            test_dataframe.loc[valid_test_rows, feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .reset_index(drop=True)
        )

        for model_name, model_template in models.items():
            print(f"Training {model_name} with {feature_group}")

            model = clone(model_template)

            try:
                model.fit(X_train, y_train)

                predictions = np.asarray(model.predict(X_test))

                metrics = calculate_metrics(y_test, predictions)

                model_path = group_directory / f"{model_name}.joblib"

                joblib.dump(model, model_path)

                metric_records.append(
                    {
                        "model_name": model_name,
                        "feature_group": feature_group,
                        "feature_count": int(len(feature_columns)),
                        "train_rows": int(len(X_train)),
                        "test_rows": int(len(X_test)),
                        **metrics,
                        "model_path": str(model_path),
                    }
                )

                selected_metadata = test_dataframe.loc[valid_test_rows].reset_index(
                    drop=True
                )

                for row_index in range(len(predictions)):
                    prediction_records.append(
                        {
                            "row_index": int(row_index),
                            "model_name": model_name,
                            "feature_group": feature_group,
                            "murcko_scaffold": (
                                selected_metadata.loc[row_index, "murcko_scaffold"]
                            ),
                            "actual_logbcf": float(y_test.iloc[row_index]),
                            "predicted_logbcf": float(predictions[row_index]),
                            "residual": float(
                                y_test.iloc[row_index] - predictions[row_index]
                            ),
                            "absolute_error": float(
                                abs(y_test.iloc[row_index] - predictions[row_index])
                            ),
                        }
                    )

                print(
                    f"MAE={metrics['mae']:.4f}, "
                    f"RMSE={metrics['rmse']:.4f}, "
                    f"R²={metrics['r2']:.4f}"
                )

            except Exception as error:
                warnings.warn(
                    f"Failed to train {model_name} with {feature_group}: {error}"
                )

    if not metric_records:
        raise RuntimeError("No models were successfully trained.")

    metrics_dataframe = (
        pd.DataFrame(metric_records)
        .sort_values(by=["mae", "rmse", "r2"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    metrics_dataframe.insert(0, "rank", np.arange(1, len(metrics_dataframe) + 1))

    predictions_dataframe = pd.DataFrame(prediction_records)

    return (metrics_dataframe, predictions_dataframe)


def compare_with_random_split(scaffold_metrics, random_metrics_path):
    if not random_metrics_path.exists():
        return pd.DataFrame()

    random_metrics = pd.read_csv(random_metrics_path)

    required_columns = {"model_name", "feature_group", "mae", "rmse", "r2"}

    if not required_columns.issubset(random_metrics.columns):
        return pd.DataFrame()

    random_subset = random_metrics[
        ["model_name", "feature_group", "mae", "rmse", "r2"]
    ].rename(
        columns={
            "mae": "random_split_mae",
            "rmse": "random_split_rmse",
            "r2": "random_split_r2",
        }
    )

    scaffold_subset = scaffold_metrics[
        ["model_name", "feature_group", "mae", "rmse", "r2"]
    ].rename(
        columns={
            "mae": "scaffold_split_mae",
            "rmse": "scaffold_split_rmse",
            "r2": "scaffold_split_r2",
        }
    )

    comparison = scaffold_subset.merge(
        random_subset, on=["model_name", "feature_group"], how="inner"
    )

    comparison["mae_increase"] = (
        comparison["scaffold_split_mae"] - comparison["random_split_mae"]
    )

    comparison["rmse_increase"] = (
        comparison["scaffold_split_rmse"] - comparison["random_split_rmse"]
    )

    comparison["r2_change"] = (
        comparison["scaffold_split_r2"] - comparison["random_split_r2"]
    )

    comparison["mae_percent_increase"] = (
        comparison["mae_increase"] / comparison["random_split_mae"] * 100
    )

    comparison = comparison.sort_values("scaffold_split_mae").reset_index(drop=True)

    return comparison


def calculate_scaffold_statistics(full_dataframe, train_dataframe, test_dataframe):
    scaffold_counts = full_dataframe["murcko_scaffold"].value_counts()

    return {
        "total_compounds": int(len(full_dataframe)),
        "training_compounds": int(len(train_dataframe)),
        "testing_compounds": int(len(test_dataframe)),
        "unique_total_scaffolds": int(full_dataframe["murcko_scaffold"].nunique()),
        "unique_training_scaffolds": int(train_dataframe["murcko_scaffold"].nunique()),
        "unique_testing_scaffolds": int(test_dataframe["murcko_scaffold"].nunique()),
        "scaffold_overlap_count": int(
            len(
                set(train_dataframe["murcko_scaffold"]).intersection(
                    set(test_dataframe["murcko_scaffold"])
                )
            )
        ),
        "largest_scaffold_group": int(scaffold_counts.max()),
        "median_scaffold_group_size": float(scaffold_counts.median()),
        "no_scaffold_compounds": int(
            (full_dataframe["murcko_scaffold"] == "NO_SCAFFOLD").sum()
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", default="data/processed/features.csv")

    parser.add_argument("--results-directory", default="results")

    parser.add_argument("--models-directory", default="models/scaffold_validation")

    parser.add_argument("--test-size", type=float, default=0.20)

    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)

    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    features_path = Path(args.features)

    results_directory = Path(args.results_directory)

    models_directory = Path(args.models_directory)

    results_directory.mkdir(parents=True, exist_ok=True)

    models_directory.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}. "
            "Run src/feature_engineering.py first."
        )

    dataframe = pd.read_csv(features_path)

    target_column = find_column(dataframe, ["logbcf", "log_bcf", "log bcf"])

    smiles_column = find_column(
        dataframe, ["canonical_smiles", "canonical smiles", "smiles"]
    )

    dataframe[target_column] = pd.to_numeric(dataframe[target_column], errors="coerce")

    dataframe = dataframe.dropna(subset=[target_column, smiles_column]).reset_index(
        drop=True
    )

    dataframe, invalid_structure_count = assign_scaffolds(dataframe, smiles_column)

    (train_dataframe, test_dataframe) = create_scaffold_split(
        dataframe, args.test_size, args.random_state
    )

    feature_groups = identify_feature_columns(dataframe)

    models = build_models()

    (metrics_dataframe, predictions_dataframe) = train_and_evaluate(
        train_dataframe,
        test_dataframe,
        target_column,
        feature_groups,
        models,
        models_directory,
    )

    comparison_dataframe = compare_with_random_split(
        metrics_dataframe, results_directory / "model_metrics.csv"
    )

    scaffold_statistics = calculate_scaffold_statistics(
        dataframe, train_dataframe, test_dataframe
    )

    scaffold_statistics["invalid_structure_count"] = int(invalid_structure_count)

    best_model = metrics_dataframe.iloc[0]

    report = {
        "split_method": ("Bemis-Murcko scaffold split"),
        "random_state": int(args.random_state),
        "requested_test_fraction": float(args.test_size),
        "actual_test_fraction": float(len(test_dataframe) / len(dataframe)),
        **scaffold_statistics,
        "best_model_name": best_model["model_name"],
        "best_feature_group": best_model["feature_group"],
        "best_mae": float(best_model["mae"]),
        "best_rmse": float(best_model["rmse"]),
        "best_r2": float(best_model["r2"]),
    }

    train_path = results_directory / "scaffold_train_split.csv"

    test_path = results_directory / "scaffold_test_split.csv"

    metrics_path = results_directory / "scaffold_model_metrics.csv"

    predictions_path = results_directory / "scaffold_predictions.csv"

    comparison_path = results_directory / "random_vs_scaffold_comparison.csv"

    report_path = results_directory / "scaffold_validation_report.json"

    train_dataframe.to_csv(train_path, index=False)

    test_dataframe.to_csv(test_path, index=False)

    metrics_dataframe.to_csv(metrics_path, index=False)

    predictions_dataframe.to_csv(predictions_path, index=False)

    if not comparison_dataframe.empty:
        comparison_dataframe.to_csv(comparison_path, index=False)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print()
    print("Scaffold validation complete")
    print(f"Total compounds: {len(dataframe)}")
    print(f"Training compounds: {len(train_dataframe)}")
    print(f"Testing compounds: {len(test_dataframe)}")
    print(f"Training scaffolds: {train_dataframe['murcko_scaffold'].nunique()}")
    print(f"Testing scaffolds: {test_dataframe['murcko_scaffold'].nunique()}")
    print("Scaffold overlap: 0")
    print()
    print(f"Best model: {best_model['model_name']}")
    print(f"Best feature group: {best_model['feature_group']}")
    print(f"Scaffold MAE: {best_model['mae']:.4f}")
    print(f"Scaffold RMSE: {best_model['rmse']:.4f}")
    print(f"Scaffold R²: {best_model['r2']:.4f}")
    print()
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved report to {report_path}")

    if not comparison_dataframe.empty:
        print(f"Saved random-vs-scaffold comparison to {comparison_path}")


if __name__ == "__main__":
    main()
