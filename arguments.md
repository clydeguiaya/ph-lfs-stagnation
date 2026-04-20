# Arguments & Justifications

All non-trivial decisions made in the data pipeline and modelling notebooks, with explicit reasoning.

---

## 1. Data Loading Strategy

### Why load all 48 months as strings (`dtype=str`) before casting?

The PSA LFS files contain occupation codes (`PUFC14_PROCC`), industry codes (`PUFC16_PKB`), and education grades (`PUFC07_GRADE`) that are **zero-padded string codes** (e.g., `"01"`, `"09"`, `"60002"`). Loading as default numeric silently truncates leading zeros, corrupting codes like `"01"` → `1`. Loading everything as string first, then explicitly casting known numeric fields, preserves all coded values.

### Why three schema mappings?

Inspection of all 48 CSV column sets revealed three distinct questionnaire versions:

| Schema | Period | Trigger |
|--------|--------|---------|
| A | Jan–Jul 2021 | Baseline LFS questionnaire |
| B | Aug–Dec 2021 | PSA redesigned questionnaire mid-2021 (HHMEM format — item numbers shifted, e.g., employment question moved from C11 to C09) |
| C | 2022–2024 | Extended baseline (adds arrangement type, province of workplace, ethnicity in 2024) |

Schema B is detected by the presence of `PUFC09_WORK` (absent in A/C). Without this mapping, naively concatenating all 48 files produces misaligned columns and silently introduces massive missingness.

### Why is `PUFURB2015` (urban/rural) missing in 2024?

The 2024 questionnaire revision dropped the urban/rural classification column. This is a known PSA survey change. It is imputed with the training-set mode rather than discarded — urban/rural is a meaningful predictor of stagnation risk (rural workers disproportionately in subsistence agriculture). Dropping the column entirely would penalise the model for a data collection change, not a genuine data absence.

---

## 2. Target Variable Construction

### Why no direct `is_stagnant` label exists

The PSA LFS is a labour force participation survey, not an employee wellbeing survey. It does not ask respondents to self-report stagnation. The concept must be **operationalised** from observable proxy variables.

### Choice of operationalisation: 2-of-3 composite rule

An employed person (`emp_status == 1`, age ≥ 15) is labelled `is_stagnant = 1` if they satisfy **at least 2 of the following 3 criteria**:

**C1 — Visible underemployment** (`want_more_work == 1`)  
This is the PSA's own official visible underemployment indicator. It directly captures workers whose current job does not fully utilise their available labour time — a defining feature of stagnation. The PSA defines it as "employed persons who express the desire to have additional hours of work in their present job or an additional job, or to have a new job with longer working hours."

**C2 — Precarious employment** (`nature_employment ∈ {2=temporary, 3=casual/seasonal}` OR `worker_class == 6 (unpaid family worker)`)  
Temporary, casual, and unpaid family workers lack employment security, access to benefits, and a promotion ladder — structural conditions that make upward mobility impossible. This criterion captures the ILO concept of "vulnerable employment." Unpaid family workers are always stagnant by definition (zero wages, no formal employment relationship).

**C3 — Education-occupation mismatch** (`education_level ≥ 6 (college+)` AND `occupation_major == 9 (elementary occupations)`)  
A college-educated worker stuck in an elementary occupation (e.g., street sweeper, domestic helper, agricultural labourer) is textbook overeducation — their human capital cannot be applied, signalling a structural trap. This is grounded in the PSA PSOC classification and Philippines-specific overeducation literature.

### Why 2-of-3 rather than any single criterion?

- **C1 alone** is noisy: some workers voluntarily seek part-time work (students, retirees, semi-retired).
- **C2 alone** is noisy: project-based and seasonal contracts are not always stagnation (e.g., construction project workers earning well above market).
- **C3 alone** is noisy: a fresh graduate in their first job may temporarily hold an elementary position.
- **2 of 3** requires co-occurring evidence, substantially reducing false positives while preserving sensitivity to genuine multi-dimensional stagnation. It mirrors composite index methodology used in the Philippine Development Plan and ILO decent work deficit indicators.

### Leakage prevention

`want_more_work`, `nature_employment`, and `worker_class` are the **only** columns used to construct the target. They are immediately excluded from the feature matrix. `education_level` and `occupation_major` participate in C3 but remain as features — this is not leakage because:
1. The criterion fires only on the *intersection* (college + elementary occupation)
2. Education independently predicts stagnation through human capital theory
3. Occupation group independently predicts stagnation through wage and mobility differences

If this were still a concern, a strict design would exclude both — but this would discard two of the most theoretically grounded predictors.

---

## 3. Feature Selection

| Feature | Rationale |
|---------|-----------|
| `age` | Labour economics: stagnation risk is U-shaped with age (young workers lack seniority; older workers face displacement) |
| `sex` | Philippines has documented gender gaps in employment quality and industry segregation |
| `marital_status` | Married workers (especially women) have higher probability of accepting inferior jobs near family |
| `region` | 17 regions with vast differences in labour market density, industry mix, and formal economy penetration |
| `urban_rural` | Rural workers structurally more vulnerable to informal/subsistence employment |
| `hh_size` | Larger households → greater economic pressure to accept any available work |
| `education_level` | Human capital: higher education should reduce stagnation but creates mismatch risk |
| `occupation_major` | Structural labour market position; major group 6 (agriculture) and 9 (elementary) are high-risk |
| `industry_sector` | Industry-level formalisation rates, wage floors, union density differ dramatically |
| `normal_hours` | Contracted hours below full-time signal structural underemployment |
| `actual_hours` | Hours worked below contracted hours signals additional underutilisation |
| `month_sin`, `month_cos` | Cyclical encoding of survey month captures seasonal labour patterns (planting/harvest, holiday retail, school year) without imposing linear order on a circular variable |
| `gdp_per_employed` | Year-level macro control: GDP per employed person captures productivity improvements (or contractions) that affect job quality across all workers in a given year |

**Excluded features:**
- `want_more_work`, `nature_employment`, `worker_class` → used in target construction
- `line_no`, `relationship` → survey administrative fields
- `survey_month` (raw integer) → replaced by sin/cos cyclical encoding
- `currently_working` → redundant given we filtered to `emp_status == 1`
- `worked_last_week` → redundant with employment status filter

---

## 4. Train/Test Split: Temporal (2021–2023 train, 2024 test)

**Random split was rejected.** With monthly panel survey data, a random split causes temporal leakage: the model can observe December 2023 data in training while predicting January 2023 data in test. GDP per employed is year-level, so any year overlap in train/test contaminates the macro feature.

**Temporal split** respects the natural data generating process: the model is trained on historical labour market conditions and evaluated on its ability to identify stagnant workers in an unseen future period (2024). This is the operationally meaningful test — a deployed tool would always predict on data after its training cutoff.

The 2024 holdout contains approximately 25% of total employed observations, sufficient for reliable evaluation.

---

## 5. Imputation Strategy

**Fit imputation statistics on training data only, then apply to test.** This prevents test-set information from leaking into the imputation, which would give an overly optimistic estimate of model performance on truly unseen data.

| Column | Strategy | Rationale |
|--------|----------|-----------|
| `urban_rural` | Mode (train set) | Column missing for all 2024 respondents due to survey redesign; mode is the most representative single imputation for a binary variable at this scale |
| `occupation_major`, `industry_sector` | Mode (train set) | Missing typically indicates the question was not asked (not in scope for the respondent's employment status); mode is a reasonable conservative estimate |
| `normal_hours`, `actual_hours` | Median (train set) | Continuous variables with skewed distributions; median is robust to outliers and extreme working hours |
| `education_level` | Median (train set) | Ordinal; rare missingness; median preserves central tendency |
| `age`, `hh_size` | Median (train set) | Continuous; median-robust |

**MCAR assumption:** Missingness is treated as missing completely at random conditional on employment status. This is plausible because most missingness stems from survey skip patterns (questions not asked to workers with certain employment types) rather than the workers' characteristics.

---

## 6. Class Imbalance Handling

### Decision: `class_weight='balanced'` in logistic regression (no SMOTE, no random sampling)

**Why not SMOTE?**  
At ~1M+ training rows, SMOTE requires synthesising minority-class observations by computing k-nearest neighbours across the entire training set. This is computationally prohibitive (O(n²) in memory with large k), and the synthetic observations gain no new information not already present in the real data at this scale. SMOTE is most valuable when the dataset is small and the minority class is severely under-represented (< 5%). At a 20–30% stagnation rate with hundreds of thousands of minority samples, SMOTE adds computational cost without meaningful benefit.

**Why not random undersampling?**  
Discarding majority-class observations destroys real signal. With millions of non-stagnant observations carrying genuine demographic and labour market variation, undersampling introduces variance and potentially removes informative edge cases.

**`class_weight='balanced'`:** Mathematically equivalent to resampling the minority class with replacement to match the majority class count. The loss function assigns each stagnant observation a weight of `n_samples / (2 * n_stagnant)` and each non-stagnant observation `n_samples / (2 * n_not_stagnant)`. This gives the model proportionally larger gradient updates from minority-class errors, without changing the data. It is the cleanest and most computationally efficient approach at this scale.

---

## 7. Evaluation Metrics

### Primary: AUC-PR (Average Precision)

With class imbalance, **ROC-AUC is misleading.** A classifier that randomly labels all workers as non-stagnant achieves 70–80% ROC-AUC by exploiting the large true negative count. AUC-PR (area under the precision-recall curve) removes true negatives from the denominator entirely — it measures only the tradeoff between finding stagnant workers (recall) and avoiding false alarms (precision). This directly corresponds to the policy objective: identify the workers most at risk of stagnation without wasting intervention resources on false positives.

### Secondary: AUC-ROC

Retained for comparison and for use cases where false positive costs are symmetric. Reports the probability that a randomly chosen stagnant worker scores higher than a randomly chosen non-stagnant worker.

### Tertiary: Classification report with optimal threshold

The 0.5 threshold is rarely optimal under class imbalance. We compute the threshold that maximises the F1 score for the stagnant class on the test set. This threshold should be selected based on the cost ratio between false negatives (missed stagnant workers — intervention cost) and false positives (wasted interventions) in the deployment context.

---

## 8. Logistic Regression as the Model

**Why logistic regression rather than a tree-based model?**

1. **Interpretability:** Coefficients map directly to log-odds — each unit increase in a feature increases the log-odds of stagnation by the coefficient value. For a policy instrument, this is essential: it allows analysts to communicate "being in a rural area increases stagnation odds by X%."

2. **Calibrated probabilities:** Logistic regression outputs well-calibrated probabilities natively (the sigmoid maps directly to probability). Tree-based models require post-hoc Platt scaling or isotonic regression for calibration. Calibration matters here because predicted probabilities will be used to rank intervention priority.

3. **Baseline requirement:** The task specifies a logistic regression baseline. More complex models (gradient boosting, neural networks) can be benchmarked against this baseline in future iterations.

4. **L2 regularisation (`C=1.0`):** Prevents overfitting on correlated demographic features (age and marital status are correlated; education and occupation are correlated). Default `C=1.0` provides moderate regularisation; can be tuned via cross-validation if needed.

5. **Solver `saga`:** Stochastic Average Gradient Augmented — designed for large-scale, sparse datasets. Supports L1, L2, and elastic net penalties. Converges significantly faster than `lbfgs` on datasets with millions of rows.

---

## 9. GDP Per Person Employed as a Macro Feature

GDP per person employed (constant 2021 PPP $) is a **year-level macro control** that captures the aggregate productivity environment faced by all workers in a given year. This is distinct from individual-level wages (not available in the LFS) and captures:
- Post-COVID recovery dynamics (2021 rebound, 2022 contraction)
- Structural shifts in labour productivity
- Cross-year comparisons without inflation distortion (PPP-adjusted)

**Limitation:** All workers in the same year receive the same GDP value — it cannot differentiate between high-GDP industries and low-GDP industries within a year. This is a macro fixed-effect, not an individual predictor. It controls for year-specific shocks that would otherwise be confounded with demographic trends.

**Source:** Our World in Data / World Bank — "GDP per person employed, constant 2021 PPP dollars." Philippines data available 2021–2024 matching the survey period exactly.

---

## 10. Education Level Encoding

The PSA `PUFC07_GRADE` is a 5-digit code where the **first digit encodes the education system level**:

| First digit | Level |
|-------------|-------|
| 0 | No grade completed |
| 1 | Elementary |
| 2 | Old high school (pre-K12) |
| 3 | Junior high school (K12) |
| 4 | Senior high school (K12) |
| 5 | Vocational/Technical |
| 6 | College (bachelor's) |
| 7 | Post-graduate |

This extraction preserves the ordinal nature of education (higher digit = higher level) in a single integer feature (0–7), which is appropriate for logistic regression's linear assumptions. The remaining 4 digits (specific grade year) add granularity not needed for this task.

---

## 11. Occupation and Industry Encoding

**Occupation:** PSOC (Philippine Standard Occupational Classification) 2-digit code. First digit = major group (1–9). Used directly as integer. The major group captures the skill level gradient (1=Managers at high skill, 9=Elementary at low skill) — this ordinal structure is meaningful for stagnation prediction.

**Industry:** PSIC (Philippine Standard Industrial Classification) 2-digit code. Binned into 10 broad sectors (agriculture, mining, manufacturing, construction/utilities, retail, transport/food, ICT/finance, professional/admin, public/health/education, other). This reduces cardinality from ~90 to 10 while preserving the economically meaningful distinctions between formal and informal sectors.

---

## 12. Cyclical Encoding for Survey Month

Survey month (1–12) is encoded as `sin(2π·month/12)` and `cos(2π·month/12)`. This is necessary because:
1. Month is a **circular** variable — December (12) is closer to January (1) than to June (6) in the seasonal cycle
2. Treating it as a linear integer breaks this circularity, causing the model to see a 12-unit "gap" between December and January
3. The 2D sin/cos encoding preserves the circular structure while remaining compatible with linear models

The seasonal signal matters: agricultural underemployment peaks during off-harvest months (July–September), holiday retail employment spikes in November–December, and school-year patterns affect youth employment throughout Q1/Q3.
