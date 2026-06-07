# Electrical Fault Detection and Classification

Data mining case study for supervised classification of electrical transmission line faults using current and voltage measurements.

This project downloads the Kaggle dataset `esathyaprakash/electrical-fault-detection-and-classification`, performs exploratory data analysis, applies physics-based feature engineering for three-phase systems, builds preprocessing pipelines, trains classification models, compares metrics, and evaluates performance per fault class.

Portuguese version: [readme.br.MD](readme.br.MD)

## Problem

Electrical transmission lines can operate normally or under different fault conditions. The goal is to use measured electrical variables to classify the system condition and identify the fault type.

This is a supervised multiclass classification problem.

## Dataset

- Source: Kaggle
- Dataset: `esathyaprakash/electrical-fault-detection-and-classification`
- Selected file: `classData.csv`
- Input variables: phase currents and voltages (`Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`)
- Target variable: `fault_type`

In `classData.csv`, the columns `G`, `C`, `B`, and `A` indicate fault involvement. The notebook combines these columns into a single multiclass target.

### Fault types

| Output [G C B A] | Type | Description |
| --- | --- | --- |
| [0 0 0 0] | No fault | Normal operation |
| [1 0 0 1] | LG | Phase A to ground |
| [0 0 1 1] | LL | Phase A to Phase B |
| [1 0 1 1] | LLG | Phases A, B and ground |
| [0 1 1 1] | LLL | Three-phase fault |
| [1 1 1 1] | LLLG | Three-phase to ground |

## Feature Engineering

In addition to the six original variables, the notebook creates features derived from three-phase system physics:

| Feature | Description |
| --- | --- |
| `I_seq_zero` | Mean of Ia, Ib, Ic — zero-sequence current component, ground involvement indicator |
| `V_seq_zero` | Mean of Va, Vb, Vc — zero-sequence voltage component |
| `I_unbalance` | Standard deviation of Ia, Ib, Ic — phase current imbalance |
| `V_unbalance` | Standard deviation of Va, Vb, Vc — phase voltage imbalance |
| `I_magnitude` | Total current magnitude (L2 norm) |
| `V_magnitude` | Total voltage magnitude (L2 norm) |

These features significantly improved separation between classes that differ only by ground involvement, such as LLL and LLLG.

## Models

The notebook trains and compares:

- Decision Tree
- Random Forest
- Gradient Boosting
- Logistic Regression

The workflow uses scikit-learn pipelines and column transformers to avoid data leakage. Preprocessing is fitted exclusively on training data.

## Results

Result summary with feature engineering:

| Model | Accuracy | Weighted F1-score | Mean CV F1 |
| --- | ---: | ---: | ---: |
| Random Forest | 0.9981 | 0.9981 | 0.9967 |
| Gradient Boosting | 0.9962 | 0.9962 | 0.9940 |
| Decision Tree | 0.9949 | 0.9949 | 0.9924 |
| Logistic Regression | 0.6682 | 0.6637 | 0.6476 |

Random Forest achieved the best performance. The F1 standard deviation across cross-validation folds was 0.0009, indicating high stability.

Logistic Regression performed poorly because the problem requires nonlinear decision boundaries between fault classes.

## Per-class Evaluation

The notebook displays confusion matrices and per-class classification reports for each model. A configurable minimum recall threshold (`MINIMUM_RECALL_PER_CLASS`, default 0.90) generates automatic alerts for classes performing below the expected level.

## Project Files

- [electrical_fault_classification.ipynb](electrical_fault_classification.ipynb): complete executable notebook.
- [model_comparison_results.csv](model_comparison_results.csv): model comparison metrics.
- [requirements.txt](requirements.txt): Python dependencies.

## How to Run

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook electrical_fault_classification.ipynb
```

Or execute it from the terminal:

```bash
jupyter nbconvert --to notebook --execute electrical_fault_classification.ipynb --inplace
```

## Notes

The dataset is downloaded with `kagglehub`. If Kaggle authentication is required in your environment, configure it locally without committing credentials.

## Limitations

- The analysis uses a single public dataset with simulated measurements.
- External validation with real field data was not performed.
- The model classifies the fault type at an isolated instant — it does not detect the fault onset or its location along the line.

## Possible Improvements

- Validate with real substation operational data.
- Add fault location along the line as an output variable.
- Evaluate model probability calibration.
- Incorporate time windowing for incipient fault detection.