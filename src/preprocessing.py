from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem

LOGGER = logging.getLogger(__name__)

COLUMN_MAP = {
    "CAS": "cas",
    "Name": "name",
    "SMILES": "smiles",
    "LogKOW": "log_kow",
    "KOW type": "kow_type",
    "logBCF": "log_bcf",
}


def repair_split_records(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    repaired = df.copy()
    rows_to_drop: list[int] = []
    repair_count = 0

    for position in range(len(repaired) - 1):
        current_index = repaired.index[position]
        next_index = repaired.index[position + 1]
        current = repaired.loc[current_index]
        following = repaired.loc[next_index]

        looks_like_split_start = (
            pd.isna(current["SMILES"])
            and pd.isna(current["LogKOW"])
            and pd.isna(current["logBCF"])
        )
        looks_like_continuation = (
            isinstance(following["Name"], str)
            and isinstance(following["SMILES"], str)
            and following["LogKOW"] == "Merged"
            and pd.notna(following["logBCF"])
        )

        if not (looks_like_split_start and looks_like_continuation):
            continue

        name_suffix = str(following["CAS"]).strip().strip('"')
        repaired.at[current_index, "Name"] = (
            f"{str(current['Name']).rstrip()} {name_suffix}".strip()
        )
        repaired.at[current_index, "SMILES"] = following["Name"]
        repaired.at[current_index, "LogKOW"] = following["SMILES"]
        repaired.at[current_index, "KOW type"] = following["LogKOW"]
        repaired.at[current_index, "logBCF"] = following["logBCF"]

        rows_to_drop.append(next_index)
        repair_count += 1

    if rows_to_drop:
        repaired = repaired.drop(index=rows_to_drop).reset_index(drop=True)

    return repaired, repair_count


def canonicalize_smiles(smiles: object) -> str | None:
    if pd.isna(smiles):
        return None

    text = str(smiles).strip()
    if not text:
        return None

    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None

    return Chem.MolToSmiles(molecule, canonical=True)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    initial_rows = len(df)

    df, repaired_rows = repair_split_records(df)
    df = df.rename(columns=COLUMN_MAP)

    missing_columns = set(COLUMN_MAP.values()) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    for column in ["cas", "name", "smiles", "kow_type"]:
        df[column] = df[column].astype("string").str.strip()

    df["name"] = df["name"].replace({"n.a.": pd.NA, "": pd.NA})
    df["kow_type"] = df["kow_type"].str.lower()

    df["log_kow"] = pd.to_numeric(df["log_kow"], errors="coerce")
    df["log_bcf"] = pd.to_numeric(df["log_bcf"], errors="coerce")

    df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
    invalid_smiles = int(df["canonical_smiles"].isna().sum())
    missing_targets = int(df["log_bcf"].isna().sum())

    df = df.dropna(subset=["canonical_smiles", "log_bcf"]).copy()

    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    df["structure_duplicate_count"] = df.groupby("canonical_smiles")[
        "canonical_smiles"
    ].transform("size")

    df.insert(0, "chemical_id", range(1, len(df) + 1))
    df = df[
        [
            "chemical_id",
            "cas",
            "name",
            "smiles",
            "canonical_smiles",
            "log_kow",
            "kow_type",
            "log_bcf",
            "structure_duplicate_count",
        ]
    ].reset_index(drop=True)

    report: dict[str, int | float] = {
        "raw_rows": initial_rows,
        "repaired_split_records": repaired_rows,
        "invalid_or_missing_smiles_removed": invalid_smiles,
        "missing_target_rows_removed": missing_targets,
        "exact_duplicate_rows_removed": exact_duplicates,
        "clean_rows": len(df),
        "unique_canonical_structures": int(df["canonical_smiles"].nunique()),
        "missing_log_kow_retained": int(df["log_kow"].isna().sum()),
        "log_bcf_min": float(df["log_bcf"].min()),
        "log_bcf_max": float(df["log_bcf"].max()),
    }
    return df, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw CSV path")
    parser.add_argument("--output", type=Path, required=True, help="Clean CSV path")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path; defaults beside output CSV",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    raw_df = pd.read_csv(args.input)
    clean_df, report = clean_dataset(raw_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(args.output, index=False)

    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    LOGGER.info("Saved %d clean rows to %s", len(clean_df), args.output)
    LOGGER.info("Saved preprocessing report to %s", report_path)
    LOGGER.info("Report: %s", report)


if __name__ == "__main__":
    main()
