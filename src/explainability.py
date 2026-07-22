import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


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


def format_feature_name(feature_name):
    formatted = str(feature_name)

    if formatted.startswith("descriptor_"):
        formatted = formatted.replace("descriptor_", "", 1)

    if formatted.startswith("fingerprint_"):
        formatted = formatted.replace("fingerprint_", "Morgan bit ", 1)

    return formatted.replace("_", " ")


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def load_best_model_information(results_directory):
    path = results_directory / "best_model.json"

    if not path.exists():
        raise FileNotFoundError(
            "results/best_model.json was not found. Run src/evaluate.py first."
        )

    return load_json(path)


def load_feature_columns(models_directory, feature_group):
    path = models_directory / feature_group / "feature_columns.json"

    if not path.exists():
        raise FileNotFoundError(f"Feature column file not found: {path}")

    return load_json(path)


def load_model(models_directory, best_model):
    model_path = (
        models_directory
        / best_model["feature_group"]
        / f"{best_model['model_name']}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run src/train.py again if the model files were deleted."
        )

    return joblib.load(model_path), model_path


def prepare_test_data(test_dataframe, feature_columns, target_column):
    missing_columns = [
        column for column in feature_columns if column not in test_dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns[:10]}")

    X_test = test_dataframe[feature_columns].apply(pd.to_numeric, errors="coerce")

    y_test = pd.to_numeric(test_dataframe[target_column], errors="coerce")

    valid_rows = y_test.notna()

    return (
        X_test.loc[valid_rows].reset_index(drop=True),
        y_test.loc[valid_rows].reset_index(drop=True),
    )


def calculate_permutation_importance(model, X_test, y_test, repeats):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=repeats,
        random_state=42,
        n_jobs=-1,
    )

    importance_dataframe = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_standard_deviation": (result.importances_std),
        }
    )

    importance_dataframe["absolute_importance"] = importance_dataframe[
        "importance_mean"
    ].abs()

    importance_dataframe = importance_dataframe.sort_values(
        "absolute_importance", ascending=False
    ).reset_index(drop=True)

    importance_dataframe.insert(0, "rank", np.arange(1, len(importance_dataframe) + 1))

    return importance_dataframe


def plot_permutation_importance(importance_dataframe, figures_directory, top_features):
    plot_data = importance_dataframe.head(top_features).copy()
    plot_data = plot_data.sort_values("importance_mean", ascending=True)

    labels = [format_feature_name(feature) for feature in plot_data["feature"]]

    positions = np.arange(len(plot_data))

    plt.figure(figsize=(10, 8))
    plt.barh(
        positions,
        plot_data["importance_mean"],
        xerr=plot_data["importance_standard_deviation"],
    )
    plt.yticks(positions, labels)
    plt.xlabel("Increase in MAE After Feature Permutation")
    plt.ylabel("Feature")
    plt.title("Permutation Feature Importance")

    save_figure(figures_directory / "permutation_importance.png")


def transform_pipeline_features(model, X):
    transformed = X.copy()

    if not hasattr(model, "named_steps"):
        return transformed

    for step_name, transformer in model.named_steps.items():
        if step_name == "model":
            break

        transformed_values = transformer.transform(transformed)

        transformed = pd.DataFrame(transformed_values, columns=X.columns, index=X.index)

    return transformed


def get_final_estimator(model):
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model")

    return model


def supports_tree_shap(model_name):
    return model_name in {"random_forest", "xgboost", "catboost"}


def calculate_tree_shap(pipeline, X_test, sample_size):
    final_estimator = get_final_estimator(pipeline)
    transformed_data = transform_pipeline_features(pipeline, X_test)

    if len(transformed_data) > sample_size:
        shap_data = transformed_data.sample(n=sample_size, random_state=42)
    else:
        shap_data = transformed_data.copy()

    explainer = shap.TreeExplainer(final_estimator)
    shap_values = explainer.shap_values(shap_data)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    return shap_data, shap_values


def save_shap_importance(shap_data, shap_values, results_directory):
    mean_absolute_shap = np.mean(np.abs(shap_values), axis=0)

    shap_importance = (
        pd.DataFrame(
            {"feature": shap_data.columns, "mean_absolute_shap": mean_absolute_shap}
        )
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap_importance.insert(0, "rank", np.arange(1, len(shap_importance) + 1))

    shap_importance.to_csv(
        results_directory / "shap_feature_importance.csv", index=False
    )

    return shap_importance


def plot_shap_bar(shap_data, shap_values, figures_directory, top_features):
    shap.summary_plot(
        shap_values, shap_data, plot_type="bar", max_display=top_features, show=False
    )

    plt.title("Mean Absolute SHAP Feature Importance")

    save_figure(figures_directory / "shap_bar.png")


def plot_shap_summary(shap_data, shap_values, figures_directory, top_features):
    shap.summary_plot(shap_values, shap_data, max_display=top_features, show=False)

    plt.title("SHAP Feature Effects")

    save_figure(figures_directory / "shap_summary.png")


def plot_shap_dependence(
    shap_data, shap_values, shap_importance, figures_directory, feature_count
):
    selected_features = shap_importance["feature"].head(feature_count)

    for feature in selected_features:
        feature_index = shap_data.columns.get_loc(feature)

        shap.dependence_plot(
            feature_index, shap_values, shap_data, interaction_index=None, show=False
        )

        plt.title(f"SHAP Dependence: {format_feature_name(feature)}")

        safe_name = str(feature).replace("/", "_").replace("\\", "_").replace(" ", "_")

        save_figure(figures_directory / f"shap_dependence_{safe_name}.png")


def save_explanation_report(
    path,
    best_model,
    model_path,
    feature_columns,
    permutation_importance_dataframe,
    shap_generated,
):
    top_permutation_features = permutation_importance_dataframe.head(20)[
        ["feature", "importance_mean", "importance_standard_deviation"]
    ].to_dict(orient="records")

    report = {
        "model_name": best_model["model_name"],
        "feature_group": best_model["feature_group"],
        "model_path": str(model_path),
        "feature_count": len(feature_columns),
        "permutation_importance_generated": True,
        "shap_generated": shap_generated,
        "top_permutation_features": top_permutation_features,
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-data", default="results/test_split.csv")

    parser.add_argument("--models-directory", default="models")

    parser.add_argument("--results-directory", default="results")

    parser.add_argument("--figures-directory", default="figures")

    parser.add_argument("--top-features", type=int, default=20)

    parser.add_argument("--permutation-repeats", type=int, default=20)

    parser.add_argument("--shap-sample-size", type=int, default=300)

    parser.add_argument("--dependence-features", type=int, default=5)

    args = parser.parse_args()

    test_data_path = Path(args.test_data)
    models_directory = Path(args.models_directory)
    results_directory = Path(args.results_directory)
    figures_directory = Path(args.figures_directory)

    results_directory.mkdir(parents=True, exist_ok=True)

    figures_directory.mkdir(parents=True, exist_ok=True)

    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_data_path}. Run src/train.py first."
        )

    best_model = load_best_model_information(results_directory)

    feature_columns = load_feature_columns(
        models_directory, best_model["feature_group"]
    )

    model, model_path = load_model(models_directory, best_model)

    test_dataframe = pd.read_csv(test_data_path)

    target_column = find_column(test_dataframe, ["logbcf", "log_bcf", "log bcf"])

    X_test, y_test = prepare_test_data(test_dataframe, feature_columns, target_column)

    permutation_dataframe = calculate_permutation_importance(
        model, X_test, y_test, args.permutation_repeats
    )

    permutation_path = results_directory / "permutation_feature_importance.csv"

    permutation_dataframe.to_csv(permutation_path, index=False)

    plot_permutation_importance(
        permutation_dataframe, figures_directory, args.top_features
    )

    shap_generated = False

    if supports_tree_shap(best_model["model_name"]):
        shap_data, shap_values = calculate_tree_shap(
            model, X_test, args.shap_sample_size
        )

        shap_importance = save_shap_importance(
            shap_data, shap_values, results_directory
        )

        plot_shap_bar(shap_data, shap_values, figures_directory, args.top_features)

        plot_shap_summary(shap_data, shap_values, figures_directory, args.top_features)

        plot_shap_dependence(
            shap_data,
            shap_values,
            shap_importance,
            figures_directory,
            args.dependence_features,
        )

        shap_generated = True

    report_path = results_directory / "explainability_report.json"

    save_explanation_report(
        report_path,
        best_model,
        model_path,
        feature_columns,
        permutation_dataframe,
        shap_generated,
    )

    print(f"Explained {best_model['model_name']} using {best_model['feature_group']}")

    print(f"Permutation importance saved to {permutation_path}")

    if shap_generated:
        print("SHAP feature importance and plots were created")
    else:
        print("SHAP was skipped because the best model was not tree-based")

    print(f"Explainability report saved to {report_path}")


if __name__ == "__main__":
    main()
