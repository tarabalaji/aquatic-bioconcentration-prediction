import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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


def load_feature_columns(models_directory, feature_group):
    path = models_directory / feature_group / "feature_columns.json"

    if not path.exists():
        raise FileNotFoundError(f"Feature list not found: {path}")

    return load_json(path)


def load_trained_model(models_directory, best_model):
    path = (
        models_directory
        / best_model["feature_group"]
        / f"{best_model['model_name']}.joblib"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Run src/train.py again if the model files were deleted."
        )

    return joblib.load(path), path


def prepare_data(dataframe, feature_columns, target_column):
    missing_columns = [
        column for column in feature_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns[:10]}")

    X = dataframe[feature_columns].apply(pd.to_numeric, errors="coerce")

    y = pd.to_numeric(dataframe[target_column], errors="coerce")

    valid_rows = y.notna()

    return (
        X.loc[valid_rows].reset_index(drop=True),
        y.loc[valid_rows].reset_index(drop=True),
        dataframe.loc[valid_rows].reset_index(drop=True),
    )


def conformal_quantile(calibration_errors, alpha):
    calibration_errors = np.asarray(calibration_errors)

    sample_size = len(calibration_errors)

    quantile_level = np.ceil((sample_size + 1) * (1 - alpha)) / sample_size

    quantile_level = min(quantile_level, 1.0)

    try:
        return float(np.quantile(calibration_errors, quantile_level, method="higher"))
    except TypeError:
        return float(
            np.quantile(calibration_errors, quantile_level, interpolation="higher")
        )


def train_conformal_model(base_model, X_train, y_train, calibration_size):
    (X_proper_train, X_calibration, y_proper_train, y_calibration) = train_test_split(
        X_train, y_train, test_size=calibration_size, random_state=RANDOM_STATE
    )

    conformal_model = clone(base_model)

    conformal_model.fit(X_proper_train, y_proper_train)

    calibration_predictions = conformal_model.predict(X_calibration)

    calibration_errors = np.abs(
        y_calibration.to_numpy() - np.asarray(calibration_predictions)
    )

    return (
        conformal_model,
        X_proper_train,
        y_proper_train,
        X_calibration,
        y_calibration,
        calibration_predictions,
        calibration_errors,
    )


def create_prediction_intervals(model, X_test, calibration_errors, alpha):
    predictions = np.asarray(model.predict(X_test))

    error_quantile = conformal_quantile(calibration_errors, alpha)

    lower_bounds = predictions - error_quantile
    upper_bounds = predictions + error_quantile

    return (predictions, lower_bounds, upper_bounds, error_quantile)


def calculate_interval_metrics(actual, predictions, lower_bounds, upper_bounds):
    actual = np.asarray(actual)
    predictions = np.asarray(predictions)
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    covered = (actual >= lower_bounds) & (actual <= upper_bounds)

    interval_widths = upper_bounds - lower_bounds

    errors = np.abs(actual - predictions)

    return {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "r2": float(r2_score(actual, predictions)),
        "empirical_coverage": float(covered.mean()),
        "mean_interval_width": float(interval_widths.mean()),
        "median_interval_width": float(np.median(interval_widths)),
        "minimum_interval_width": float(interval_widths.min()),
        "maximum_interval_width": float(interval_widths.max()),
        "mean_absolute_error_covered": float(errors[covered].mean())
        if covered.any()
        else None,
        "mean_absolute_error_not_covered": float(errors[~covered].mean())
        if (~covered).any()
        else None,
        "covered_count": int(covered.sum()),
        "not_covered_count": int((~covered).sum()),
    }


def get_identifier_columns(dataframe):
    possible_columns = [
        "cas",
        "CAS",
        "cas_number",
        "name",
        "Name",
        "canonical_smiles",
        "smiles",
        "SMILES",
    ]

    return [column for column in possible_columns if column in dataframe.columns]


def build_prediction_dataframe(
    metadata, actual, predictions, lower_bounds, upper_bounds
):
    identifier_columns = get_identifier_columns(metadata)

    if identifier_columns:
        output = metadata[identifier_columns].copy()
    else:
        output = pd.DataFrame(index=np.arange(len(actual)))

    output["actual_logbcf"] = np.asarray(actual)
    output["predicted_logbcf"] = predictions
    output["lower_bound"] = lower_bounds
    output["upper_bound"] = upper_bounds
    output["interval_width"] = upper_bounds - lower_bounds
    output["residual"] = np.asarray(actual) - predictions
    output["absolute_error"] = np.abs(output["residual"])
    output["covered"] = (output["actual_logbcf"] >= output["lower_bound"]) & (
        output["actual_logbcf"] <= output["upper_bound"]
    )

    return output


def calculate_coverage_by_target_range(prediction_dataframe, bin_count):
    dataframe = prediction_dataframe.copy()

    try:
        dataframe["target_range"] = pd.qcut(
            dataframe["actual_logbcf"], q=bin_count, duplicates="drop"
        )
    except ValueError:
        dataframe["target_range"] = pd.cut(dataframe["actual_logbcf"], bins=bin_count)

    grouped = (
        dataframe.groupby("target_range", observed=True)
        .agg(
            sample_count=("covered", "size"),
            empirical_coverage=("covered", "mean"),
            mean_interval_width=("interval_width", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            minimum_actual_logbcf=("actual_logbcf", "min"),
            maximum_actual_logbcf=("actual_logbcf", "max"),
        )
        .reset_index()
    )

    grouped["target_range"] = grouped["target_range"].astype(str)

    return grouped


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_prediction_intervals(prediction_dataframe, figures_directory):
    plot_data = prediction_dataframe.sort_values("actual_logbcf").reset_index(drop=True)

    positions = np.arange(len(plot_data))

    lower_errors = plot_data["predicted_logbcf"] - plot_data["lower_bound"]

    upper_errors = plot_data["upper_bound"] - plot_data["predicted_logbcf"]

    plt.figure(figsize=(12, 7))

    plt.errorbar(
        positions,
        plot_data["predicted_logbcf"],
        yerr=[lower_errors, upper_errors],
        fmt="o",
        markersize=3,
        alpha=0.6,
        capsize=2,
    )

    plt.scatter(
        positions, plot_data["actual_logbcf"], marker="x", label="Experimental logBCF"
    )

    plt.xlabel("Test compounds sorted by experimental logBCF")
    plt.ylabel("logBCF")
    plt.title("Conformal Prediction Intervals")
    plt.legend()

    save_figure(figures_directory / "conformal_prediction_intervals.png")


def plot_actual_vs_predicted_intervals(prediction_dataframe, figures_directory):
    actual = prediction_dataframe["actual_logbcf"].to_numpy()

    predicted = prediction_dataframe["predicted_logbcf"].to_numpy()

    lower_errors = predicted - prediction_dataframe["lower_bound"].to_numpy()

    upper_errors = prediction_dataframe["upper_bound"].to_numpy() - predicted

    minimum = min(actual.min(), prediction_dataframe["lower_bound"].min())

    maximum = max(actual.max(), prediction_dataframe["upper_bound"].max())

    plt.figure(figsize=(8, 8))

    plt.errorbar(
        actual,
        predicted,
        yerr=[lower_errors, upper_errors],
        fmt="o",
        alpha=0.6,
        capsize=2,
    )

    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")

    plt.xlabel("Experimental logBCF")
    plt.ylabel("Predicted logBCF")
    plt.title("Actual vs Predicted logBCF with Prediction Intervals")

    save_figure(figures_directory / "parity_plot_with_uncertainty.png")


def plot_calibration_errors(calibration_errors, error_quantile, figures_directory):
    plt.figure(figsize=(8, 6))

    plt.hist(calibration_errors, bins=20, edgecolor="black")

    plt.axvline(
        error_quantile,
        linestyle="--",
        label=f"Conformal quantile = {error_quantile:.3f}",
    )

    plt.xlabel("Absolute Calibration Error")
    plt.ylabel("Frequency")
    plt.title("Calibration Error Distribution")
    plt.legend()

    save_figure(figures_directory / "calibration_error_distribution.png")


def plot_coverage_by_target_range(
    coverage_dataframe, target_coverage, figures_directory
):
    positions = np.arange(len(coverage_dataframe))

    labels = coverage_dataframe["target_range"].tolist()

    plt.figure(figsize=(10, 6))

    plt.bar(positions, coverage_dataframe["empirical_coverage"])

    plt.axhline(
        target_coverage,
        linestyle="--",
        label=f"Target coverage = {target_coverage:.0%}",
    )

    plt.xticks(positions, labels, rotation=35, ha="right")

    plt.ylim(0, 1.05)
    plt.xlabel("Experimental logBCF Range")
    plt.ylabel("Empirical Coverage")
    plt.title("Prediction Interval Coverage by Target Range")
    plt.legend()

    save_figure(figures_directory / "coverage_by_logbcf_range.png")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", default="results/train_split.csv")

    parser.add_argument("--test-data", default="results/test_split.csv")

    parser.add_argument("--models-directory", default="models")

    parser.add_argument("--results-directory", default="results")

    parser.add_argument("--figures-directory", default="figures")

    parser.add_argument("--alpha", type=float, default=0.10)

    parser.add_argument("--calibration-size", type=float, default=0.25)

    parser.add_argument("--coverage-bins", type=int, default=5)

    args = parser.parse_args()

    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1.")

    if not 0 < args.calibration_size < 1:
        raise ValueError("--calibration-size must be between 0 and 1.")

    train_path = Path(args.train_data)
    test_path = Path(args.test_data)
    models_directory = Path(args.models_directory)
    results_directory = Path(args.results_directory)
    figures_directory = Path(args.figures_directory)

    results_directory.mkdir(parents=True, exist_ok=True)

    figures_directory.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}. Run src/train.py first."
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Testing data not found: {test_path}. Run src/train.py first."
        )

    best_model = load_best_model(results_directory)

    feature_columns = load_feature_columns(
        models_directory, best_model["feature_group"]
    )

    base_model, model_path = load_trained_model(models_directory, best_model)

    train_dataframe = pd.read_csv(train_path)

    test_dataframe = pd.read_csv(test_path)

    target_column = find_column(train_dataframe, ["logbcf", "log_bcf", "log bcf"])

    test_target_column = find_column(test_dataframe, ["logbcf", "log_bcf", "log bcf"])

    X_train, y_train, train_metadata = prepare_data(
        train_dataframe, feature_columns, target_column
    )

    X_test, y_test, test_metadata = prepare_data(
        test_dataframe, feature_columns, test_target_column
    )

    (
        conformal_model,
        X_proper_train,
        y_proper_train,
        X_calibration,
        y_calibration,
        calibration_predictions,
        calibration_errors,
    ) = train_conformal_model(base_model, X_train, y_train, args.calibration_size)

    (predictions, lower_bounds, upper_bounds, error_quantile) = (
        create_prediction_intervals(
            conformal_model, X_test, calibration_errors, args.alpha
        )
    )

    metrics = calculate_interval_metrics(
        y_test, predictions, lower_bounds, upper_bounds
    )

    prediction_dataframe = build_prediction_dataframe(
        test_metadata, y_test, predictions, lower_bounds, upper_bounds
    )

    coverage_dataframe = calculate_coverage_by_target_range(
        prediction_dataframe, args.coverage_bins
    )

    predictions_path = results_directory / "uncertainty_predictions.csv"

    coverage_path = results_directory / "coverage_by_logbcf_range.csv"

    report_path = results_directory / "uncertainty_report.json"

    conformal_model_path = (
        models_directory
        / best_model["feature_group"]
        / f"{best_model['model_name']}_conformal.joblib"
    )

    prediction_dataframe.to_csv(predictions_path, index=False)

    coverage_dataframe.to_csv(coverage_path, index=False)

    joblib.dump(conformal_model, conformal_model_path)

    report = {
        "method": "split_conformal_prediction",
        "model_name": best_model["model_name"],
        "feature_group": best_model["feature_group"],
        "original_model_path": str(model_path),
        "conformal_model_path": str(conformal_model_path),
        "alpha": float(args.alpha),
        "target_coverage": float(1 - args.alpha),
        "calibration_fraction": float(args.calibration_size),
        "original_training_rows": int(len(X_train)),
        "proper_training_rows": int(len(X_proper_train)),
        "calibration_rows": int(len(X_calibration)),
        "test_rows": int(len(X_test)),
        "conformal_error_quantile": float(error_quantile),
        "calibration_mae": float(
            mean_absolute_error(y_calibration, calibration_predictions)
        ),
        **metrics,
    }

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    plot_prediction_intervals(prediction_dataframe, figures_directory)

    plot_actual_vs_predicted_intervals(prediction_dataframe, figures_directory)

    plot_calibration_errors(calibration_errors, error_quantile, figures_directory)

    plot_coverage_by_target_range(coverage_dataframe, 1 - args.alpha, figures_directory)

    print(f"Best model: {best_model['model_name']} using {best_model['feature_group']}")

    print(f"Target coverage: {1 - args.alpha:.1%}")

    print(f"Empirical test coverage: {metrics['empirical_coverage']:.1%}")

    print(f"Conformal interval half-width: {error_quantile:.4f} logBCF")

    print(f"Test MAE: {metrics['mae']:.4f}")

    print(f"Test RMSE: {metrics['rmse']:.4f}")

    print(f"Test R²: {metrics['r2']:.4f}")

    print(f"Saved uncertainty predictions to {predictions_path}")

    print(f"Saved uncertainty report to {report_path}")


if __name__ == "__main__":
    main()
