import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


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


def load_feature_columns(feature_columns_path):
    with open(feature_columns_path, "r", encoding="utf-8") as feature_file:
        return json.load(feature_file)


def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
        "explained_variance": float(explained_variance_score(actual, predicted)),
    }


def get_identifier_columns(dataframe):
    possible_identifiers = [
        "cas",
        "CAS",
        "cas_number",
        "name",
        "Name",
        "canonical_smiles",
        "smiles",
        "SMILES",
    ]

    identifier_columns = []

    for column in possible_identifiers:
        if column in dataframe.columns and column not in identifier_columns:
            identifier_columns.append(column)

    return identifier_columns


def evaluate_models(test_dataframe, target_column, models_directory):
    metrics_records = []
    prediction_dataframes = []

    feature_group_directories = sorted(
        directory for directory in models_directory.iterdir() if directory.is_dir()
    )

    identifier_columns = get_identifier_columns(test_dataframe)

    for feature_group_directory in feature_group_directories:
        feature_columns_path = feature_group_directory / "feature_columns.json"

        if not feature_columns_path.exists():
            continue

        feature_columns = load_feature_columns(feature_columns_path)

        missing_columns = [
            column for column in feature_columns if column not in test_dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing features for "
                f"{feature_group_directory.name}: "
                f"{missing_columns[:10]}"
            )

        X_test = test_dataframe[feature_columns].apply(pd.to_numeric, errors="coerce")

        y_test = pd.to_numeric(test_dataframe[target_column], errors="coerce")

        valid_target_rows = y_test.notna()

        X_test = X_test.loc[valid_target_rows]
        y_test = y_test.loc[valid_target_rows]

        model_files = sorted(feature_group_directory.glob("*.joblib"))

        for model_path in model_files:
            model = joblib.load(model_path)
            predictions = model.predict(X_test)

            metrics = calculate_metrics(y_test, predictions)

            residuals = y_test.to_numpy() - np.asarray(predictions)

            absolute_errors = np.abs(residuals)

            metrics_record = {
                "model_name": model_path.stem,
                "feature_group": (feature_group_directory.name),
                "feature_count": int(len(feature_columns)),
                "test_rows": int(len(y_test)),
                **metrics,
                "mean_residual": float(np.mean(residuals)),
                "residual_standard_deviation": float(np.std(residuals)),
                "maximum_absolute_error": float(np.max(absolute_errors)),
                "model_path": str(model_path),
            }

            metrics_records.append(metrics_record)

            prediction_dataframe = (
                test_dataframe.loc[valid_target_rows, identifier_columns]
                .copy()
                .reset_index(drop=True)
            )

            prediction_dataframe["row_index"] = test_dataframe.loc[
                valid_target_rows
            ].index.to_numpy()

            prediction_dataframe["model_name"] = model_path.stem

            prediction_dataframe["feature_group"] = feature_group_directory.name

            prediction_dataframe["actual_logbcf"] = y_test.to_numpy()

            prediction_dataframe["predicted_logbcf"] = predictions

            prediction_dataframe["residual"] = residuals

            prediction_dataframe["absolute_error"] = absolute_errors

            prediction_dataframes.append(prediction_dataframe)

            print(
                f"Evaluated {model_path.stem} using "
                f"{feature_group_directory.name}: "
                f"MAE={metrics['mae']:.4f}, "
                f"RMSE={metrics['rmse']:.4f}, "
                f"R²={metrics['r2']:.4f}"
            )

    if not metrics_records:
        raise RuntimeError(
            "No trained models were found. Run src/train.py before evaluation."
        )

    metrics_dataframe = pd.DataFrame(metrics_records)

    metrics_dataframe = metrics_dataframe.sort_values(
        by=["mae", "rmse", "r2"], ascending=[True, True, False]
    ).reset_index(drop=True)

    metrics_dataframe.insert(0, "rank", np.arange(1, len(metrics_dataframe) + 1))

    predictions_dataframe = pd.concat(prediction_dataframes, ignore_index=True)

    return metrics_dataframe, predictions_dataframe


def build_feature_group_comparison(metrics_dataframe):
    best_indices = metrics_dataframe.groupby("feature_group")["mae"].idxmin()

    comparison_dataframe = (
        metrics_dataframe.loc[
            best_indices,
            ["feature_group", "model_name", "feature_count", "mae", "rmse", "r2"],
        ]
        .sort_values("mae")
        .reset_index(drop=True)
    )

    comparison_dataframe = comparison_dataframe.rename(
        columns={"model_name": "best_model"}
    )

    return comparison_dataframe


def build_model_comparison(metrics_dataframe):
    best_indices = metrics_dataframe.groupby("model_name")["mae"].idxmin()

    comparison_dataframe = (
        metrics_dataframe.loc[
            best_indices,
            ["model_name", "feature_group", "feature_count", "mae", "rmse", "r2"],
        ]
        .sort_values("mae")
        .reset_index(drop=True)
    )

    comparison_dataframe = comparison_dataframe.rename(
        columns={"feature_group": "best_feature_group"}
    )

    return comparison_dataframe


def save_best_model_information(metrics_dataframe, output_path):
    best_row = metrics_dataframe.iloc[0]

    best_model_information = {
        "rank": int(best_row["rank"]),
        "model_name": best_row["model_name"],
        "feature_group": best_row["feature_group"],
        "feature_count": int(best_row["feature_count"]),
        "test_rows": int(best_row["test_rows"]),
        "mae": float(best_row["mae"]),
        "rmse": float(best_row["rmse"]),
        "r2": float(best_row["r2"]),
        "median_absolute_error": float(best_row["median_absolute_error"]),
        "explained_variance": float(best_row["explained_variance"]),
        "mean_residual": float(best_row["mean_residual"]),
        "maximum_absolute_error": float(best_row["maximum_absolute_error"]),
        "model_path": best_row["model_path"],
    }

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(best_model_information, output_file, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-data", default="results/test_split.csv")

    parser.add_argument("--models-directory", default="models")

    parser.add_argument("--results-directory", default="results")

    args = parser.parse_args()

    test_data_path = Path(args.test_data)
    models_directory = Path(args.models_directory)
    results_directory = Path(args.results_directory)

    results_directory.mkdir(parents=True, exist_ok=True)

    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_data_path}. Run src/train.py first."
        )

    if not models_directory.exists():
        raise FileNotFoundError(
            f"Models directory not found: {models_directory}. Run src/train.py first."
        )

    test_dataframe = pd.read_csv(test_data_path)

    target_column = find_column(test_dataframe, ["logbcf", "log_bcf", "log bcf"])

    (metrics_dataframe, predictions_dataframe) = evaluate_models(
        test_dataframe, target_column, models_directory
    )

    feature_group_comparison = build_feature_group_comparison(metrics_dataframe)

    model_comparison = build_model_comparison(metrics_dataframe)

    metrics_path = results_directory / "model_metrics.csv"

    predictions_path = results_directory / "predictions.csv"

    feature_comparison_path = results_directory / "feature_group_comparison.csv"

    model_comparison_path = results_directory / "model_comparison.csv"

    best_model_path = results_directory / "best_model.json"

    metrics_dataframe.to_csv(metrics_path, index=False)

    predictions_dataframe.to_csv(predictions_path, index=False)

    feature_group_comparison.to_csv(feature_comparison_path, index=False)

    model_comparison.to_csv(model_comparison_path, index=False)

    save_best_model_information(metrics_dataframe, best_model_path)

    best_result = metrics_dataframe.iloc[0]

    print()
    print(f"Evaluated {len(metrics_dataframe)} models")

    print(f"Best model: {best_result['model_name']}")

    print(f"Best feature group: {best_result['feature_group']}")

    print(f"MAE: {best_result['mae']:.4f}")

    print(f"RMSE: {best_result['rmse']:.4f}")

    print(f"R²: {best_result['r2']:.4f}")

    print(f"Saved metrics to {metrics_path}")

    print(f"Saved predictions to {predictions_path}")

    print(f"Saved best model information to {best_model_path}")


if __name__ == "__main__":
    main()
