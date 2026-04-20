# Philippine Labour Force Survey — Employee Stagnation Prediction

Predicts employee stagnation from the PSA Labour Force Survey (LFS) 2021–2024 using a rule-based target construction and logistic regression baseline.

---

## Problem

Employee stagnation — being stuck in a low-quality job with no path to upward mobility — is not directly measured in any Philippine survey. This project operationalises it from three PSA-defined labour market indicators, then builds a predictive model using only demographic and structural features available before employment quality is observed.

---

## Dataset

**Source:** Philippine Statistics Authority (PSA) — Labour Force Survey Public Use Files (PUF), Jan 2021 – Dec 2024  
**Files:** 48 monthly CSVs, ~30,000–60,000 respondents per file  
**Schemas:** Three questionnaire versions across the 4-year span (Aug–Dec 2021 used a redesigned form)

**GDP covariate:** Our World in Data — "GDP per person employed (constant 2021 PPP $)" — Philippines rows for 2021–2024

```
data/
├── PHL-PSA-LFS-2021-2024-PUF/   # 48 monthly LFS CSVs + data dictionaries
│   ├── PHL-PSA-LFS-2021-PUF/
│   ├── PHL-PSA-LFS-2022-PUF/
│   ├── PHL-PSA-LFS-2023-PUF/
│   └── PHL-PSA-LFS-2024-PUF/
└── gdp-per-person-employed-constant-ppp/
    └── gdp-per-person-employed-constant-ppp.csv
```

---

## Target Variable: `is_stagnant`

An **employed person (age ≥ 15)** is labelled `is_stagnant = 1` if **2 or more** of the following criteria are met simultaneously:

| Criterion | PSA Variable | Condition |
|-----------|-------------|-----------|
| C1 — Visible underemployment | `PUFC20_PWMORE` | = 1 (wants additional work/hours) |
| C2 — Precarious employment | `PUFC17_NATEM`, `PUFC23_PCLASS` | temporary/casual job OR unpaid family worker |
| C3 — Education-occupation mismatch | `PUFC07_GRADE`, `PUFC14_PROCC` | College+ education AND elementary occupation (PSOC 9) |

The 2-of-3 composite rule reduces noise from each individual indicator. See [arguments.md](arguments.md) for full justification.

**Leakage prevention:** C1/C2 variables are dropped from the feature matrix. C3 variables are retained in encoded form as legitimate independent predictors.

---

## Pipeline

### `notebooks/data-pipeline.ipynb`

1. **Load & harmonise** — detects schema version per file, renames to a unified canonical column set, concatenates all 48 months
2. **Filter** — employed persons only (`emp_status == 1`, age ≥ 15)
3. **Construct target** — applies 2-of-3 stagnation rule
4. **Merge GDP** — year-level join on `survey_year`
5. **Feature engineering** — education level (0–7), occupation major group (1–9), industry sector (1–10 binned), cyclical month encoding (sin/cos)
6. **Temporal split** — train: 2021–2023, test: 2024
7. **Imputation** — fit on train only, apply to both sets (median for continuous, mode for categorical)
8. **Save** — `data/X_train.csv`, `data/X_test.csv`, `data/y_train.csv`, `data/y_test.csv`

### `notebooks/log_reg.ipynb`

1. **Train** — `LogisticRegression(class_weight='balanced', solver='saga', C=1.0)` via `StandardScaler` pipeline
2. **Sigmoid visualisation** — sigmoid function, score distribution, calibration plot
3. **Classification report** — precision, recall, F1 at default threshold (0.5)
4. **PR curve + AUC-PR** — primary metric for imbalanced classification
5. **ROC curve + AUC-ROC** — secondary metric
6. **Optimal threshold** — threshold that maximises F1 for the stagnant class
7. **Feature importance** — standardised logistic coefficients (log-odds scale)

---

## Feature Matrix (14 features)

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Continuous | Respondent age |
| `sex` | Categorical | 1=male, 2=female |
| `marital_status` | Categorical | 1=single … 5=common-law |
| `region` | Categorical | PSA region code (1–17) |
| `urban_rural` | Binary | 1=urban, 2=rural |
| `hh_size` | Continuous | Household size |
| `education_level` | Ordinal | 0=no grade … 7=post-graduate |
| `occupation_major` | Categorical | PSOC 1-digit major group (1–9) |
| `industry_sector` | Categorical | Binned PSIC sector (1–10) |
| `normal_hours` | Continuous | Normal hours per week contracted |
| `actual_hours` | Continuous | Actual hours worked last week |
| `month_sin` | Continuous | sin(2π·month/12) — seasonal encoding |
| `month_cos` | Continuous | cos(2π·month/12) — seasonal encoding |
| `gdp_per_employed` | Continuous | Philippines GDP per person employed (constant 2021 PPP $) |

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Target construction | 2-of-3 composite | Reduces false positives from noisy individual indicators |
| Train/test split | Temporal (2021-2023 / 2024) | Prevents temporal leakage; reflects real deployment scenario |
| Class imbalance | `class_weight='balanced'` | Computationally equivalent to oversampling at 1M+ scale; SMOTE is prohibitive |
| Primary metric | AUC-PR | ROC-AUC is misleading under class imbalance (inflated by true negatives) |
| Model | Logistic regression | Interpretable coefficients; calibrated probabilities; scales to 1M+ rows |
| Imputation | Median/mode from train only | Prevents test leakage; preserves missing-as-unknown semantics |
| Month encoding | sin/cos cyclical | Respects circular nature of calendar months |

Full technical justifications in [arguments.md](arguments.md).

---

## Outputs

After running both notebooks in order:

```
data/
├── X_train.csv                      # Feature matrix, 2021-2023
├── X_test.csv                       # Feature matrix, 2024
├── y_train.csv                      # Stagnation labels, 2021-2023
├── y_test.csv                       # Stagnation labels, 2024
├── employed_processed.csv           # Full processed employed dataset
├── imputation_fill_values.pkl       # Imputation constants (reproducibility)
├── stagnation_class_distribution.png
├── feature_distributions.png
├── sigmoid_diagnostics.png
├── confusion_matrix.png
├── pr_roc_curves.png
├── threshold_f1.png
└── feature_importance.png
models/
└── logistic_regression.pkl
```

---

## How to Run

```bash
# 1. Activate virtual environment
source venv/Scripts/activate  # Windows
# or: source venv/bin/activate  # Unix/Mac

# 2. Install dependencies (if not already installed)
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# 3. Run pipeline first, then model
jupyter nbconvert --to notebook --execute notebooks/data-pipeline.ipynb
jupyter nbconvert --to notebook --execute notebooks/log_reg.ipynb
```

Or open in Jupyter Lab and run cells sequentially.

---

## Limitations

1. **Cross-sectional, not longitudinal:** The LFS does not track individuals across months. Stagnation is inferred from a single point-in-time snapshot per respondent — it cannot capture dynamic transitions in and out of stagnation.

2. **Imputed urban/rural for 2024:** The 2024 questionnaire dropped the urban/rural classification. Mode imputation introduces measurement error for all 2024 test samples.

3. **Target definition subjectivity:** The 2-of-3 rule is theoretically grounded but operationally chosen. Different threshold choices (any-1-of-3, all-3-of-3) produce different prevalence rates and label sets.

4. **No wage data:** The LFS records basic pay (`PUFC25_PBASIC`) but with high missingness and inconsistent reporting. Wage-based stagnation measures (earning below living wage) were excluded due to data quality.

5. **Logistic regression baseline only:** Linear decision boundaries cannot capture interaction effects (e.g., education × region × industry traps). Tree-based models (XGBoost, LightGBM) are natural next steps.
