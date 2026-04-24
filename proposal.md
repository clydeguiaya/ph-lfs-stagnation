# Predicting Labour Market Stagnation in the Philippines
## A Machine Learning Approach Using PSA Labour Force Survey Data (2021–2024)

**Prepared by:** NARIZ, SIEGA, TACATA

---

## Research Question

**Can machine learning classifiers trained on the PSA Labour Force Survey (2021–2024) reliably identify Filipino workers at risk of labour market stagnation using only demographic and structural features — and which features carry the strongest predictive weight?**

Supporting questions:
- How prevalent is multi-dimensional stagnation (simultaneous underemployment, precarious employment, and education-occupation mismatch) among employed Filipinos aged 15 and above?
- Does aggregate economic productivity (GDP per person employed) predict individual stagnation risk beyond what demographic features alone explain?
- Which sociodemographic profiles — by age, sex, region, and industry sector — are most associated with stagnation risk?

---

## Background

The Philippine economy recorded GDP growth of 5.6% in 2024, up from 5.5% in 2023. Yet headline growth conceals a persistent and structurally entrenched labour quality problem. Household final consumption expenditure — the primary driver of 70–75% of GDP — has slowed for 11 consecutive quarters, from 10% growth in 2021 Q1 down to 4.7% in 2024 Q4 (IBON Foundation, 2025a). Foreign direct investment fell from approximately US$12 billion in 2021 to US$9.1 billion in 2023 (IBON Foundation, 2025a). The number of poorly-paid Filipinos increased by over 2 million between 2021 and 2024 (IBON Foundation, 2025b).

PSA Labour Force Survey data show that 14–17% of employed workers are visibly underemployed — they hold a job but work fewer hours than they want and cannot find additional work (PSA LFS, 2023). Many more are in temporary, casual, or unpaid family arrangements, or hold positions far below their educational qualifications. A worker counted as "employed" in the national accounts may be earning a poverty wage, in a job with no security, and with no realistic path to advancement.

No official Philippine statistic directly measures labour market stagnation. This study constructs and predicts that condition.

---

## What Is Labour Market Stagnation?

Labour market stagnation is a state in which an employed worker is trapped in a low-quality job with no realistic path to upward mobility. It is not unemployment — the worker is in the labour force and working — but neither is it decent employment. Three observable conditions signal it:

1. **Visible underemployment** — the worker wants more hours or a better position but cannot obtain one; their time and capacity are wasted.
2. **Precarious employment** — the worker is in a temporary, casual, or unpaid family arrangement, with no employment security, benefits, or career structure.
3. **Education-occupation mismatch (overeducation)** — the worker holds a college or postgraduate qualification but is employed in an elementary occupation (domestic helper, farmhand, market vendor), meaning their human capital cannot be applied.

A worker experiencing two or three of these simultaneously, by any reasonable standard, is stagnant.

---

## Alignment with Sustainable Development Goals

### SDG 8 — Decent Work and Economic Growth

This study directly addresses SDG 8, which calls on governments to "promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all."

Relevant targets:

- **Target 8.5:** "By 2030, achieve full and productive employment and decent work for all women and men, including for young people and persons with disabilities, and equal pay for work of equal value." The Philippine employment statistics that inform SDG 8 tracking count workers by employment status, not job quality. A worker employed 10 hours per week in an unpaid family enterprise is counted identically to a full-time professional. This study constructs a quality-adjusted indicator that reveals who is not achieving productive employment despite being nominally employed.

- **Target 8.6:** "By 2030, substantially reduce the proportion of youth not in employment, education or training." Education-occupation mismatch — a criterion in this study — is particularly pronounced among young workers. A graduate forced into an elementary occupation represents both a human capital loss and a failure to achieve 8.6's intent.

### SDG 10 — Reduced Inequalities

Stagnation is not uniformly distributed. The World Bank Philippines Growth and Jobs Report (2025) finds that productivity gains have not reached workers in agriculture, informal trade, and elementary occupations — the sectors where stagnation risk is highest. The regional and sectoral dimensions of this study directly map to the inequality gradients SDG 10 seeks to address.

---

## Why GDP Per Person Employed Belongs in the Model

GDP per person employed (constant 2021 PPP dollars) is included as a year-level macro control. This variable captures the aggregate productivity environment faced by all workers in a given year — post-COVID recovery in 2021, contraction signals in 2022, and the structural slowdown in 2023–2024.

Three pieces of evidence justify its inclusion:

**1. Productivity gains are not reaching workers.** IBON Foundation (2023) documents that Filipino workers' productivity has increased over the review period, but real wages have simultaneously declined. This means GDP per employed person has risen while individual wellbeing has not — the GDP feature captures the macro environment; the stagnation label captures whether that macro growth translates to job quality at the individual level.

**2. The World Bank confirms low-productivity job creation.** The World Bank Philippines Growth and Jobs Report (2025) finds that Philippine growth remains input-driven rather than productivity-driven. New employment is disproportionately concentrated in low-productivity sectors. Year-on-year variation in GDP per employed captures whether conditions for decent work are expanding or contracting, which cannot be read from demographic variables alone.

**3. GDP is a year-level fixed effect, not an individual predictor.** All workers in a given year receive the same GDP value. Its role is to control for year-specific shocks (pandemic rebound, monetary tightening, fiscal contraction) that would otherwise be confounded with demographic trends across the 2021–2024 period.

---

## Why This Study Matters

### The measurement gap

The PSA Labour Force Survey is the primary instrument for tracking Philippine employment. It records whether a person has a job — not whether that job is decent, growing, or paying fairly. Between the two extremes of unemployment and stable formal employment lies a large population of workers who are neither jobless nor well-employed. They are invisible to standard statistics.

### The macro context

IBON Foundation (2025a) argues that the Philippines is entering a period of economic stagnation characterised by weakening household consumption, falling investment, and debt-driven government spending. In this environment, identifying which workers are most structurally exposed to stagnation risk is both a labour market and a macroeconomic question. The IBON Foundation (2025c) further documents that reported improvements in underemployment statistics have not translated into real improvements in worker welfare — wages remain low, hours remain insufficient, and job security remains fragile.

### The policy application

A model that reliably predicts stagnation risk from observable demographic and structural features — without requiring self-reported wellbeing data or longitudinal tracking — enables targeted intervention. Policymakers can identify which regions, industries, age groups, and household structures are disproportionately at risk before conditions deteriorate further.

---

## Data

**Philippine Statistics Authority — Labour Force Survey Public Use Files (PUF), January 2021 – December 2024**

- 48 monthly cross-sectional surveys
- 6,554,055 total respondents across the full period
- 2,740,496 employed working-age respondents (the analytical sample; age ≥ 15, classified as employed)
- Surveys cover employment status, occupation code (PSOC), industry code (PSIC), educational attainment, contracted and actual hours worked, household size, region, and urban/rural classification

**World Bank / Our World in Data — GDP per person employed (constant 2021 PPP $)**

- Philippines annual series matching the 2021–2024 survey period
- Merged at the year level as a macro-structural control variable

---

## Target Construction

An employed worker (age ≥ 15) is labelled **stagnant** if at least **2 of 3** of the following PSA-grounded criteria are met simultaneously:

| Criterion | PSA Variable | Condition |
|-----------|-------------|-----------|
| C1 — Visible underemployment | `PUFC20_PWMORE` | = 1 (wants additional work or hours) |
| C2 — Precarious employment | `PUFC17_NATEM`, `PUFC23_PCLASS` | Temporary or casual job, OR unpaid family worker |
| C3 — Education-occupation mismatch | `PUFC07_GRADE`, `PUFC14_PROCC` | College or higher education AND PSOC major group 9 (elementary occupations) |

The 2-of-3 composite threshold reduces false positives from any single noisy indicator. Voluntary part-time workers (C1 alone), high-earning contract workers (C2 alone), and new graduates in transitional roles (C3 alone) are not labelled stagnant unless a second condition corroborates the first. The resulting stagnation rate across the employed sample is **8.1%**, representing approximately 222,000 stagnant workers identified in the dataset.

All criterion variables are excluded from the feature matrix to prevent the model from reconstructing the labelling rule.

---

## Methodology

Three classifiers are trained on 2021–2023 data and evaluated on 2024 data:

- **Logistic Regression** — interpretable log-odds coefficients; calibrated probability output; baseline for comparison
- **K-Nearest Neighbors** — non-parametric; captures local interaction patterns without assuming linearity
- **Linear Support Vector Machine** — maximum-margin decision boundary; robust under moderate class imbalance

The train/test split is **temporal** (not random): training on 2021–2023 data and evaluating on 2024 data reflects real deployment conditions, where a model predicts on future, unseen survey data.

Primary evaluation metric: **AUC-PR (Area Under the Precision-Recall Curve)**, appropriate under the observed class imbalance (8.1% stagnant). ROC-AUC inflates in imbalanced settings by crediting large true-negative counts; AUC-PR removes true negatives from the denominator and measures only the quality of stagnant-worker identification.

---

## Limitations

1. **Cross-sectional, not longitudinal.** The LFS does not track the same individuals across months. Stagnation is inferred from a single snapshot per respondent — it cannot capture transitions into or out of stagnation over time.
2. **Operationalised label, not self-reported.** The stagnation measure is constructed from observable proxies. Different threshold choices (any 1-of-3 or all 3-of-3) would produce different prevalence rates and label compositions.
3. **No wage data.** PSA basic pay records have high missingness and inconsistent reporting; wage-based stagnation criteria were excluded.
4. **Imputed urban/rural for 2024.** The 2024 questionnaire dropped the urban/rural classification. All 2024 test records inherit a mode-imputed value, introducing systematic measurement error for that feature.
5. **Linear decision boundaries for LR and SVM.** Interaction effects (e.g., age × industry sector) cannot be captured by linear models. KNN can approximate these locally but is constrained to a 50,000-record balanced subsample for computational tractability.

---

## References

1. Philippine Statistics Authority. (2021–2024). *Labour Force Survey Public Use Files (PUF), January 2021 – December 2024.* https://psa.gov.ph

2. IBON Foundation. (2025a). *Falling employment, higher underemployment signs of unsolved PH jobs crisis.* https://www.ibon.org/mr-march-2025-lfs/

3. IBON Foundation. (2025b). *Over 2M increase in poorly paid Filipinos.* https://www.ibon.org/over-2m-increase-in-poorly-paid-filipinos/

4. IBON Foundation. (2025c). *Filipino workers struggling despite reported drop in underemployment.* https://www.ibon.org/filipino-workers-struggling-nov-2024-lfs/

5. IBON Foundation. (2023). *Filipino workers' productivity increasing but real wages falling.* https://www.ibon.org/filipino-workers-productivity-increasing-but-real-wages-falling-ibon/

6. World Bank. (2025). *Philippines Growth and Jobs Report 2025.* https://www.worldbank.org/en/country/philippines/publication/philippines-growth-and-jobs-report-2025

7. Our World in Data / World Bank. (2024). *GDP per person employed, constant 2021 PPP dollars — Philippines.* https://ourworldindata.org/grapher/gdp-per-person-employed-constant-ppp
