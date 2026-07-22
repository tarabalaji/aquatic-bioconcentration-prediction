import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_best_model(results_directory):
    best_model_path = results_directory / "best_model.json"

    if not best_model_path.exists():
        raise FileNotFoundError(
            "best_model.json was not found. Run src/evaluate.py first."
        )

    with open(best_model_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_data(results_directory):
    metrics_path = results_directory / "model_metrics.csv"
    predictions_path = results_directory / "predictions.csv"
    feature_comparison_path = results_directory / "feature_group_comparison.csv"
    model_comparison_path = results_directory / "model_comparison.csv"

    required_paths = [
        metrics_path,
        predictions_path,
        feature_comparison_path,
        model_comparison_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} was not found. Run src/evaluate.py first.")

    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    feature_comparison = pd.read_csv(feature_comparison_path)
    model_comparison = pd.read_csv(model_comparison_path)

    return (metrics, predictions, feature_comparison, model_comparison)


def format_label(value):
    return str(value).replace("_", " ").title()


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_mae(metrics, figures_directory):
    plot_data = metrics.sort_values("mae", ascending=True).copy()

    labels = [
        f"{format_label(row.model_name)}\n{format_label(row.feature_group)}"
        for row in plot_data.itertuples()
    ]

    positions = np.arange(len(plot_data))

    plt.figure(figsize=(12, 9))
    plt.barh(positions, plot_data["mae"])
    plt.yticks(positions, labels, fontsize=8)
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Model and Feature Group")
    plt.title("Model Performance Ranked by Mean Absolute Error")
    plt.gca().invert_yaxis()

    save_figure(figures_directory / "model_mae_comparison.png")


def plot_model_rmse(metrics, figures_directory):
    plot_data = metrics.sort_values("rmse", ascending=True).copy()

    labels = [
        f"{format_label(row.model_name)}\n{format_label(row.feature_group)}"
        for row in plot_data.itertuples()
    ]

    positions = np.arange(len(plot_data))

    plt.figure(figsize=(12, 9))
    plt.barh(positions, plot_data["rmse"])
    plt.yticks(positions, labels, fontsize=8)
    plt.xlabel("Root Mean Squared Error")
    plt.ylabel("Model and Feature Group")
    plt.title("Model Performance Ranked by RMSE")
    plt.gca().invert_yaxis()

    save_figure(figures_directory / "model_rmse_comparison.png")


def plot_model_r2(metrics, figures_directory):
    plot_data = metrics.sort_values("r2", ascending=False).copy()

    labels = [
        f"{format_label(row.model_name)}\n{format_label(row.feature_group)}"
        for row in plot_data.itertuples()
    ]

    positions = np.arange(len(plot_data))

    plt.figure(figsize=(12, 9))
    plt.barh(positions, plot_data["r2"])
    plt.yticks(positions, labels, fontsize=8)
    plt.xlabel("R²")
    plt.ylabel("Model and Feature Group")
    plt.title("Model Performance Ranked by R²")
    plt.gca().invert_yaxis()

    save_figure(figures_directory / "model_r2_comparison.png")


def plot_feature_group_comparison(feature_comparison, figures_directory):
    plot_data = feature_comparison.sort_values("mae", ascending=True).copy()

    labels = [format_label(value) for value in plot_data["feature_group"]]

    positions = np.arange(len(plot_data))

    plt.figure(figsize=(9, 6))
    plt.barh(positions, plot_data["mae"])
    plt.yticks(positions, labels)
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Feature Group")
    plt.title("Best Model Performance by Feature Group")
    plt.gca().invert_yaxis()

    save_figure(figures_directory / "feature_group_comparison.png")


def plot_algorithm_comparison(model_comparison, figures_directory):
    plot_data = model_comparison.sort_values("mae", ascending=True).copy()

    labels = [format_label(value) for value in plot_data["model_name"]]

    positions = np.arange(len(plot_data))

    plt.figure(figsize=(9, 6))
    plt.barh(positions, plot_data["mae"])
    plt.yticks(positions, labels)
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Model")
    plt.title("Best Performance for Each Algorithm")
    plt.gca().invert_yaxis()

    save_figure(figures_directory / "algorithm_comparison.png")


def get_best_predictions(predictions, best_model):
    selected = predictions[
        (predictions["model_name"] == best_model["model_name"])
        & (predictions["feature_group"] == best_model["feature_group"])
    ].copy()

    if selected.empty:
        raise ValueError("Predictions for the best model were not found.")

    return selected


def plot_parity(best_predictions, best_model, figures_directory):
    actual = best_predictions["actual_logbcf"]
    predicted = best_predictions["predicted_logbcf"]

    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())

    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, alpha=0.7)
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    plt.xlabel("Experimental logBCF")
    plt.ylabel("Predicted logBCF")
    plt.title(
        f"Actual vs Predicted logBCF\n"
        f"{format_label(best_model['model_name'])} with "
        f"{format_label(best_model['feature_group'])}"
    )

    save_figure(figures_directory / "parity_plot.png")


def plot_residuals(best_predictions, best_model, figures_directory):
    predicted = best_predictions["predicted_logbcf"]
    residuals = best_predictions["residual"]

    plt.figure(figsize=(8, 6))
    plt.scatter(predicted, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted logBCF")
    plt.ylabel("Residual")
    plt.title(
        f"Residual Plot\n"
        f"{format_label(best_model['model_name'])} with "
        f"{format_label(best_model['feature_group'])}"
    )

    save_figure(figures_directory / "residual_plot.png")


def plot_absolute_error_distribution(best_predictions, best_model, figures_directory):
    absolute_errors = best_predictions["absolute_error"]

    plt.figure(figsize=(8, 6))
    plt.hist(absolute_errors, bins=20, edgecolor="black")
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.title(
        f"Absolute Error Distribution\n"
        f"{format_label(best_model['model_name'])} with "
        f"{format_label(best_model['feature_group'])}"
    )

    save_figure(figures_directory / "absolute_error_distribution.png")


def plot_residual_distribution(best_predictions, best_model, figures_directory):
    residuals = best_predictions["residual"]

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(
        f"Residual Distribution\n"
        f"{format_label(best_model['model_name'])} with "
        f"{format_label(best_model['feature_group'])}"
    )

    save_figure(figures_directory / "residual_distribution.png")


def plot_error_by_actual_value(best_predictions, best_model, figures_directory):
    actual = best_predictions["actual_logbcf"]
    absolute_error = best_predictions["absolute_error"]

    plt.figure(figsize=(8, 6))
    plt.scatter(actual, absolute_error, alpha=0.7)
    plt.xlabel("Experimental logBCF")
    plt.ylabel("Absolute Error")
    plt.title(
        f"Prediction Error by Experimental logBCF\n"
        f"{format_label(best_model['model_name'])} with "
        f"{format_label(best_model['feature_group'])}"
    )

    save_figure(figures_directory / "error_by_actual_logbcf.png")


def save_best_model_summary(best_model, figures_directory):
    summary_path = figures_directory / "best_model_summary.txt"

    lines = [
        f"Model: {format_label(best_model['model_name'])}",
        f"Feature group: {format_label(best_model['feature_group'])}",
        f"Feature count: {best_model['feature_count']}",
        f"Test rows: {best_model['test_rows']}",
        f"MAE: {best_model['mae']:.4f}",
        f"RMSE: {best_model['rmse']:.4f}",
        f"R²: {best_model['r2']:.4f}",
        (f"Median absolute error: {best_model['median_absolute_error']:.4f}"),
        (f"Explained variance: {best_model['explained_variance']:.4f}"),
        (f"Maximum absolute error: {best_model['maximum_absolute_error']:.4f}"),
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results-directory", default="results")

    parser.add_argument("--figures-directory", default="figures")

    args = parser.parse_args()

    results_directory = Path(args.results_directory)
    figures_directory = Path(args.figures_directory)

    figures_directory.mkdir(parents=True, exist_ok=True)

    best_model = load_best_model(results_directory)

    (metrics, predictions, feature_comparison, model_comparison) = load_data(
        results_directory
    )

    best_predictions = get_best_predictions(predictions, best_model)

    plot_model_mae(metrics, figures_directory)
    plot_model_rmse(metrics, figures_directory)
    plot_model_r2(metrics, figures_directory)

    plot_feature_group_comparison(feature_comparison, figures_directory)

    plot_algorithm_comparison(model_comparison, figures_directory)

    plot_parity(best_predictions, best_model, figures_directory)

    plot_residuals(best_predictions, best_model, figures_directory)

    plot_absolute_error_distribution(best_predictions, best_model, figures_directory)

    plot_residual_distribution(best_predictions, best_model, figures_directory)

    plot_error_by_actual_value(best_predictions, best_model, figures_directory)

    save_best_model_summary(best_model, figures_directory)

    print(
        f"Created figures for "
        f"{best_model['model_name']} using "
        f"{best_model['feature_group']}"
    )

    print(f"Figures saved to {figures_directory}")


if __name__ == "__main__":
    main()
