# Aquatic Bioconcentration Prediction

An explainable machine learning project for predicting the aquatic bioconcentration factor of organic chemicals using molecular structure, physicochemical properties, and quantitative structure–activity relationship modeling.

## Overview

The bioconcentration factor, or BCF, measures how strongly a chemical accumulates in an aquatic organism relative to its concentration in the surrounding water. Experimental BCF testing can be expensive and time-consuming, making computational prediction useful for environmental chemical screening.

This project develops machine learning models to predict experimental `logBCF` values from molecular information. It also evaluates model interpretability, uncertainty, and applicability to structurally unfamiliar chemicals.

## Research Questions

This project investigates the following questions:

* How accurately can machine learning models predict aquatic `logBCF` values?
* How much predictive value does molecular structure provide beyond LogKOW alone?
* Which molecular properties are most strongly associated with bioconcentration?
* How reliable are predictions for chemicals that differ from those in the training data?
* Can uncertainty and applicability-domain analysis identify predictions that should be treated cautiously?

## Dataset

The project uses the QSAR Fish Bioconcentration Factor dataset.

The original dataset contains information for approximately 1,058 organic chemicals, including:

* CAS numbers
* Chemical names
* SMILES molecular structures
* LogKOW values
* LogKOW measurement type
* Experimental `logBCF` values

The preprocessing pipeline repairs malformed records, validates molecular structures, standardizes numeric columns, removes unusable rows, and generates canonical SMILES representations.

## Project Structure

```text
aquatic-bioconcentration-prediction/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── QSAR_BCF_Kow.csv
│   │
│   ├── processed/
│   │   ├── qsar_bcf_clean.csv
│   │   └── preprocessing_report.json
│   │
│   └── external/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualization.py
│   ├── explainability.py
│   ├── uncertainty.py
│   ├── applicability_domain.py
│   └── utils.py
│
├── models/
│   └── .gitkeep
│
├── results/
│
└── figures/
```

## Methodology

The project follows this general pipeline:

1. Clean and validate the raw chemical dataset.
2. Convert SMILES strings into canonical molecular representations.
3. Generate molecular descriptors and fingerprints using RDKit.
4. Compare traditional LogKOW-based prediction with molecular machine learning models.
5. Train and evaluate multiple regression algorithms.
6. analyze feature importance and model explanations.
7. Estimate prediction uncertainty.
8. assess whether each chemical lies within the model's applicability domain.

## Feature Sets

The project will compare several feature configurations:

* LogKOW only
* RDKit molecular descriptors only
* Molecular fingerprints only
* RDKit descriptors with LogKOW
* Molecular descriptors, fingerprints, and LogKOW

This comparison will measure how much molecular structure improves prediction beyond the traditional relationship between hydrophobicity and bioconcentration.

## Models

The planned regression models include:

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* XGBoost Regressor
* CatBoost Regressor

Simpler models will serve as baselines for evaluating whether more complex ensemble methods provide meaningful improvements.

## Evaluation

Model performance will be evaluated using:

* Mean Absolute Error
* Root Mean Squared Error
* Coefficient of Determination
* Residual analysis

The project will also compare different data-splitting strategies, including random splits and structure-aware splits, to test generalization to unfamiliar chemicals.

## Explainability

Model behavior will be analyzed using methods such as:

* SHAP values
* Permutation feature importance
* Partial dependence analysis
* Descriptor-level feature rankings

These analyses will help identify molecular characteristics associated with higher predicted bioconcentration.

## Uncertainty Analysis

Prediction uncertainty will be estimated to distinguish reliable predictions from cases where the model has limited confidence.

Potential methods include:

* Model ensembles
* Bootstrap prediction intervals
* Conformal prediction

The uncertainty analysis will evaluate whether larger prediction intervals are associated with larger model errors.

## Applicability Domain

The applicability-domain analysis will determine whether a new chemical is sufficiently similar to the chemicals used during model training.

The analysis may use:

* Molecular fingerprint similarity
* Descriptor-space distance
* Nearest-neighbor similarity
* Leverage-based methods

Prediction error will be compared across different similarity levels to determine when the model becomes less reliable.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/aquatic-bioconcentration-prediction.git
cd aquatic-bioconcentration-prediction
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running Preprocessing

Place the original dataset in:

```text
data/raw/QSAR_BCF_Kow.csv
```

Run:

```bash
python src/preprocessing.py \
  --input data/raw/QSAR_BCF_Kow.csv \
  --output data/processed/qsar_bcf_clean.csv
```

The preprocessing script generates:

```text
data/processed/qsar_bcf_clean.csv
data/processed/preprocessing_report.json
```

## Technologies

* Python
* Pandas
* NumPy
* RDKit
* Scikit-learn
* XGBoost
* CatBoost
* SHAP
* Matplotlib

## Current Status

The repository structure and initial preprocessing pipeline have been created. Molecular feature generation, model training, uncertainty estimation, explainability, and applicability-domain analysis are under development.

## Reproducibility

All major stages of the workflow are implemented as Python scripts rather than notebooks. Generated models and results can be recreated by running the repository pipeline from the raw dataset.

## Limitations

Predicted BCF values are computational estimates and should not replace experimental environmental testing. Model performance may be lower for chemicals that are structurally different from those represented in the training dataset.

## License

This project is released under the MIT License.
