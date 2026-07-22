import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
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


def get_feature_columns(dataframe, target_column, logkow_column):
    descriptor_columns = [
        column for column in dataframe.columns if column.startswith("descriptor_")
    ]

    fingerprint_columns = [
        column for column in dataframe.columns if column.startswith("fingerprint_")
    ]

    feature_groups = {
        "logkow_only": [logkow_column],
        "descriptors_only": descriptor_columns,
        "fingerprints_only": fingerprint_columns,
        "descriptors_logkow": descriptor_columns + [logkow_column],
        "combined": (descriptor_columns + fingerprint_columns + [logkow_column]),
    }

    for feature_group_name, columns in feature_groups.items():
        columns = [column for column in columns if column != target_column]

        feature_groups[feature_group_name] = list(dict.fromkeys(columns))

    return feature_groups


def build_linear_regression():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_ridge_regression():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )


def build_random_forest():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=5,
                    min_child_weight=2,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.05,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_catboost():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                CatBoostRegressor(
                    iterations=500,
                    learning_rate=0.03,
                    depth=6,
                    loss_function="RMSE",
                    verbose=False,
                    random_seed=RANDOM_STATE,
                    allow_writing_files=False,
                ),
            ),
        ]
    )


def get_models():
    return {
        "linear_regression": build_linear_regression,
        "ridge_regression": build_ridge_regression,
        "random_forest": build_random_forest,
        "xgboost": build_xgboost,
        "catboost": build_catboost,
    }


def make_split(dataframe, target_column, test_size):
    valid_rows = dataframe[target_column].notna()
    filtered_dataframe = dataframe.loc[valid_rows].reset_index(drop=True)

    indices = np.arange(len(filtered_dataframe))

    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=RANDOM_STATE
    )

    return filtered_dataframe, train_indices, test_indices


def save_split_files(dataframe, train_indices, test_indices, output_directory):
    train_split = dataframe.iloc[train_indices].copy()
    test_split = dataframe.iloc[test_indices].copy()

    train_split.to_csv(output_directory / "train_split.csv", index=False)

    test_split.to_csv(output_directory / "test_split.csv", index=False)


def train_models(
    dataframe,
    target_column,
    feature_groups,
    train_indices,
    test_indices,
    models_directory,
):
    y = pd.to_numeric(dataframe[target_column], errors="coerce")

    training_records = []

    for feature_group_name, feature_columns in feature_groups.items():
        if not feature_columns:
            continue

        X = dataframe[feature_columns].apply(pd.to_numeric, errors="coerce")

        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        feature_group_directory = models_directory / feature_group_name

        feature_group_directory.mkdir(parents=True, exist_ok=True)

        for model_name, model_builder in get_models().items():
            model = model_builder()

            model.fit(X_train, y_train)

            model_path = feature_group_directory / f"{model_name}.joblib"

            joblib.dump(model, model_path)

            prediction = model.predict(X_test)

            record = {
                "model_name": model_name,
                "feature_group": feature_group_name,
                "model_path": str(model_path),
                "feature_count": int(len(feature_columns)),
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
                "prediction_min": float(np.min(prediction)),
                "prediction_max": float(np.max(prediction)),
                "prediction_mean": float(np.mean(prediction)),
            }

            training_records.append(record)

            print(
                f"Trained {model_name} using "
                f"{feature_group_name} "
                f"({len(feature_columns)} features)"
            )

        feature_list_path = feature_group_directory / "feature_columns.json"

        with open(feature_list_path, "w", encoding="utf-8") as feature_file:
            json.dump(feature_columns, feature_file, indent=4)

    return training_records


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="data/processed/features.csv")

    parser.add_argument("--models-directory", default="models")

    parser.add_argument("--results-directory", default="results")

    parser.add_argument("--test-size", type=float, default=0.2)

    args = parser.parse_args()

    input_path = Path(args.input)
    models_directory = Path(args.models_directory)
    results_directory = Path(args.results_directory)

    models_directory.mkdir(parents=True, exist_ok=True)

    results_directory.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(input_path)

    target_column = find_column(dataframe, ["logbcf", "log_bcf", "log bcf"])

    logkow_column = find_column(dataframe, ["logkow", "log_kow", "log kow"])

    feature_groups = get_feature_columns(dataframe, target_column, logkow_column)

    (dataframe, train_indices, test_indices) = make_split(
        dataframe, target_column, args.test_size
    )

    save_split_files(dataframe, train_indices, test_indices, results_directory)

    training_records = train_models(
        dataframe,
        target_column,
        feature_groups,
        train_indices,
        test_indices,
        models_directory,
    )

    training_dataframe = pd.DataFrame(training_records)

    training_dataframe.to_csv(results_directory / "training_summary.csv", index=False)

    metadata = {
        "input_file": str(input_path),
        "target_column": target_column,
        "logkow_column": logkow_column,
        "random_state": RANDOM_STATE,
        "test_size": float(args.test_size),
        "total_rows": int(len(dataframe)),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "feature_groups": {
            group_name: {
                "feature_count": len(columns),
                "features_file": str(
                    models_directory / group_name / "feature_columns.json"
                ),
            }
            for group_name, columns in feature_groups.items()
        },
        "models": list(get_models().keys()),
        "total_models_trained": int(len(training_records)),
    }

    with open(
        models_directory / "training_metadata.json", "w", encoding="utf-8"
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    print()
    print(f"Training rows: {len(train_indices)}")
    print(f"Testing rows: {len(test_indices)}")
    print(f"Models trained: {len(training_records)}")
    print(f"Models saved in {models_directory}")
    print(f"Split files saved in {results_directory}")


if __name__ == "__main__":
    main()
