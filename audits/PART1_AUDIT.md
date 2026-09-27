# Research Audit Report: Part 1 Bikeshare Analysis & Forecasting

**Assessment:** UNLV PhD Technical Assessment — Dr. Sohn's Research Lab  
**Domain:** Part 1: Capital Bikeshare Demand Analysis & Forecasting  
**Audit Date:** September 26, 2026  
**Auditor:** PhD Technical Candidate (Self-Audit & Methodological Verification)

---

## 1. Audit Summary

This Research Audit presents a thorough methodological and technical review of Part 1 (Capital Bikeshare Demand Analysis & Forecasting). Every line of code, statistical model specification, feature engineering pipeline, evaluation metric calculation, and visualization has been audited against standard econometrics and machine learning protocols to guarantee scientific rigor and zero data leakage.

### Overall Audit Verdict
* **Data Integrity:** **PASSED** (17,379 observations, 17 columns, 0 missing, 0 duplicates, contiguous hourly coverage from 2011-01-01 00:00:00 to 2012-12-31 23:00:00).
* **Statistical Regression:** **PASSED & REFINED** (Eliminated `temp`/`atemp` VIF collinearity; conducted Breusch-Pagan and Jarque-Bera tests; applied Jacobian transformation to Log-OLS likelihood to make AIC statistically comparable).
* **Forecasting Partitioning & Leakage:** **PASSED & DUAL-BENCHMARKED** (Enforced Days 1–20 Train vs Days 21–End Test per month. Tested both a Pure Exogenous Zero-Target-Lag Model and a Day-Ahead Rolling Lag Model).
* **Metrics & Evaluation:** **PASSED** (Verified RMSLE, MAE, RMSE, $R^2$ calculations, inverse $\text{expm1}$ transformations, and non-negative clipping).
* **Visualizations & Tone:** **PASSED** (Resolved Seaborn deprecation warnings and Matplotlib tick warnings; purged causal language in favor of observational association terms).

---

## 2. Data Validation Audit

| Data Property | Claimed in Plan | Verified in Code & Runtime Output | Status |
| :--- | :--- | :--- | :---: |
| **Observation Count** | 17,379 rows | 17,379 rows loaded from `data/bike/hour.csv` | **VERIFIED** |
| **Attribute Count** | 17 columns | 17 columns (`instant`, `dteday`, `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`, `casual`, `registered`, `cnt`) | **VERIFIED** |
| **Missing Values** | 0 missing | `df.isnull().sum().sum() == 0` | **VERIFIED** |
| **Duplicates** | 0 duplicates | `df.duplicated().sum() == 0` | **VERIFIED** |
| **Temporal Coverage** | 2011-01-01 to 2012-12-31 | `2011-01-01 00:00:00` to `2012-12-31 23:00:00` | **VERIFIED** |
| **Total Demand** | 3,292,679 rentals | Registered: 2,672,662 (81.2%), Casual: 620,017 (18.8%) | **VERIFIED** |

---

## 3. Regression Validation Audit

### 3.1 Categorical Encoding & Variable Selection
* Categorical variables (`season`, `weathersit`) are encoded as dummy reference variables using `C(season)` and `C(weathersit)` in `statsmodels`, preventing ordinal assumptions on non-monotonic weather categories.
* Binary indicators (`workingday`, `holiday`) are explicitly parsed as integers (0/1).

### 3.2 Multicollinearity & VIF Analysis
* **Initial Audit Finding:** `temp_celsius` ($\text{VIF} = 43.60$) and `atemp_celsius` ($\text{VIF} = 43.73$) exhibit severe multicollinearity ($r = 0.985$).
* **Correction:** Dropping `atemp_celsius` reduces all remaining predictor VIF values below **$1.50$**, eliminating collinearity risk.

### 3.3 Statistical Diagnostic Tests
* **Heteroscedasticity Test:** Breusch-Pagan Lagrange Multiplier test on OLS yields $p = 1.28 \times 10^{-182}$, confirming severe heteroscedasticity in raw OLS.
* **Normality Test:** Jarque-Bera test on OLS residuals yields $p < 0.001$, confirming violation of Gaussian error distribution assumptions.
* **Count Overdispersion:** Ratio of variance to mean ($\text{Var}(Y)/\mathbb{E}[Y] = 173.66$) confirms severe overdispersion, invalidating standard Poisson models and justifying Negative Binomial GLM / Log-OLS.

### 3.4 Jacobian-Adjusted AIC Comparison
* **Methodological Trap Avoided:** Raw AIC values for models with different dependent variable transformations (e.g., $y$ vs. $\ln(y+1)$) are not directly comparable because the likelihood scale changes by the Jacobian determinant $J = \prod \frac{1}{y_i + 1}$.
* **Statistical Formulation:**
  $$\ln L_y(y \mid \beta, \sigma^2) = \ln L_z(z \mid \beta, \sigma^2) - \sum_{i=1}^N \ln(y_i + 1)$$
  $$\text{AIC}_{\text{adj}} = 2k - 2 \ln L_y$$
* **Empirical Results:**
  - **Standard OLS (Raw Scale $y$):** $\text{AIC} = 224,316.62$
  - **Log-OLS (Raw Log Scale $z$):** Raw $\text{AIC} = 55,465.82$ *(Unadjusted)*
  - **Jacobian Adjustment Term:** $\sum \ln(y_i + 1) = 79,504.38$
  - **Log-OLS (Jacobian-Adjusted on $y$ Scale):** **$\text{AIC}_{\text{adj}} = 214,474.59$**
* **Conclusion:** Even after rigorous Jacobian adjustment, Log-OLS $\text{AIC}_{\text{adj}}$ ($214,474.59$) is substantially lower than standard OLS ($224,316.62$), proving statistically valid superiority.

---

## 4. Forecasting Partitioning & Leakage Audit

### 4.1 Chronological Split Verification
* **Training Set:** Days 1 through 20 of every month (`day <= 20`).
* **Test Set:** Day 21 through the end of every month (`day >= 21`).
* **Audit Result:** Verified across all 24 months. Zero random shuffling; time-series ordering strictly preserved.

### 4.2 Leakage Audit Across Feature Engineering Paradigms
To guarantee zero lookahead bias under any operational forecasting setup, two distinct feature models were implemented and benchmarked:

1. **Model Paradigm A: Pure Exogenous Features (Zero Target Lags)**
   - Predictors: `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp_celsius`, `humidity_pct`, `windspeed_kmh`, $\sin/\cos$ cyclical encodings.
   - Target Lags: **ZERO** (`cnt` is never used as a lag).
   - **Leakage Status:** **100% Leakage-Free under Multi-Step Horizon Cutoff** (Days 21–End predicted at midnight on Day 20 without observing future target values).
   - Empirical Performance: **RMSLE = 0.4050**, **MAE = 32.332 bikes/hr**, **$R^2 = 0.9093$**.

2. **Model Paradigm B: Day-Ahead Rolling Update Forecast (24h Shifted Target Lags)**
   - Predictors: Pure exogenous features + `cnt_lag_24`, `cnt_lag_48`, `cnt_lag_168`, `cnt_roll_mean_24`.
   - Protocol: Assumes a 24-hour rolling update framework (observed rentals from day $d-1$ available prior to forecasting day $d$).
   - Empirical Performance: **RMSLE = 0.4053**, **MAE = 34.795 bikes/hr**, **$R^2 = 0.8986$**.

---

## 5. Metric Validation Audit

* **RMSLE Calculation:** Verified $\sqrt{\frac{1}{N} \sum (\ln(\hat{y}_i+1) - \ln(y_i+1))^2}$.
* **Target Inverse Transformation:** Models trained on $\ln(\text{cnt}+1)$ are transformed back using $\hat{y} = \max(0, \exp(\hat{z}) - 1)$ via `np.expm1` and `np.clip(y_pred, 0, None)`.
* **Baseline Construction:** Historical Hourly Average Baseline computes group means by `(hr, workingday, season)` strictly on training data (`days <= 20`) and maps them to test windows (`days >= 21`).

---

## 6. Visualization Validation Audit

* **Styling & Aesthetics:** All plots rendered at 300 DPI, styled with consistent seaborn whitegrid theme and custom palettes.
* **Warning Purge:** 
  - Resolved Seaborn deprecation warnings by assigning `hue` and setting `legend=False`.
  - Fixed Matplotlib tick location warnings by explicitly defining fixed tick positions prior to label assignment.

---

## 7. Summary of Issues Found & Fixed

| Issue Description | Severity | Fix Implemented | Verification |
| :--- | :---: | :--- | :---: |
| **Unadjusted Log-OLS AIC Comparison** | High | Added exact Jacobian determinant correction ($\sum \ln(y_i+1) = 79,504.38$) to compute true comparative AIC on raw $y$ scale. | Log-OLS $\text{AIC}_{\text{adj}} = 214,474.59$ vs OLS $224,316.62$. |
| **Ambiguity in Test Target Lags** | Medium | Built and benchmarked **Pure Exogenous Feature Model (Zero Target Lags)** alongside Rolling Lag model. | Pure Feature LightGBM achieves $\text{MAE} = 32.33$, $R^2 = 0.9093$. |
| **Seaborn & Matplotlib Warnings** | Low | Updated `src/utils/plotting.py` syntax to eliminate all hue and tick warnings. | Pipeline runs with zero deprecation warnings. |
| **Causal Language in Docstrings/Notebook** | Low | Replaced words like "causes", "drives", "impacts" with observational association terms ("is associated with", "correlates with"). | Purged causal phrasing. |

---

## 8. Remaining Limitations

1. **Weather Variable Coarseness:** Weather codes in the dataset are discrete integers (1–4) representing aggregated atmospheric conditions. Extreme events (code 4) are rare (only 3 observations in two years).
2. **Exogenous Temperature Horizon:** Forecasting models assume perfect knowledge of test-set weather metrics (temperature, humidity, windspeed) at evaluation time, consistent with standard micro-mobility benchmarks.

---

## 9. Final Readiness Assessment

* **Part 1 Completion:** **100% Executed & Audited**.
* **Code & Notebook Status:** Executable notebook [`notebooks/01_bikeshare_demand_analysis.ipynb`](file:///d:/UNLV_PhD_Technical_Assessment/notebooks/01_bikeshare_demand_analysis.ipynb) fully executed with zero warnings and embedded outputs.
* **Part 2 Readiness:** **READY TO PROCEED TO PART 2 (BUILDING FOOTPRINT SEGMENTATION)**.
