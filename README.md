# Obesity Metrics and Stroke Risk Assessment Framework

This repository provides a comprehensive analysis pipeline for evaluating the association between anthropometric obesity indicators and stroke risk, as well as post-stroke mortality outcomes in both population-based and hospital-based cohorts.

**Study Title:** Multi-dimensional Obesity Assessment and Stroke Risk Stratification: A Machine Learning-Enhanced Scoring System

---

## Study Overview

Obesity is a well-established risk factor for stroke, but the relative importance of different anthropometric measurements and their optimal thresholds remain unclear. This project develops data-driven risk scoring systems by:

- Systematically evaluating 11 obesity-related anthropometric indicators
- Identifying non-linear dose-response relationships using restricted cubic splines (RCS)
- Leveraging machine learning (Gradient Boosting) to quantify variable importance via SHAP values
- Constructing interpretable risk scores for clinical translation
- Validating findings across population-based (UK Biobank) and hospital-based (local stroke registry) cohorts

The pipeline supports both **primary prevention** (pre-stroke risk prediction) and **secondary prevention** (post-stroke mortality/CVD event prediction).

---

## Project Structure
```txt
Stroke-Obesity-Risk-Assessment/
├── Correlation_Analysis/
│   ├── correlation_matrix_analysis.R
│   └── multicollinearity_check.R
│
├── Incident_Stroke_Analysis/          # Incident stroke analysis
│   ├── univariate_logistic_and_RCS_logistic.py
│   ├── quintile_logistic.py
│   ├── GBM_SHAP_analysis.py
│   ├── grouped_logistic_validation.py
│   └── requirements.txt
│
├── Post_Stroke_mortality_Analysis/             # Post-stroke mortality analysis
│   ├── univariate_cox_and_RCS_cox.py
│   ├── quintile_cox.py
│   ├── GBS_SHAP_analysis.py
│   ├── grouped_cox_validation.py
│   └── requirements.txt
│
└── README.md
```

---

## Core Components

### 1. Incident Stroke Risk Analysis

**Location:** `Incident_Stroke_Analysis/`

**Study Population:** General population from UK Biobank  
**Outcomes:** 
- Stroke
- Hemorrhagic stroke  
- Ischemic stroke

**Analysis Pipeline:**

#### Step 1: Univariate Logistic Regression
- Evaluate crude associations between 11 obesity indicators and stroke incidence
- Adjusted for age and sex
- Outcomes: OR (95% CI), p-values

#### Step 2: Restricted Cubic Spline (RCS) Analysis
- Model non-linear dose-response relationships
- 3 knots at 10th, 50th, 90th percentiles
- Reference value: median
- Identify optimal risk ranges for each indicator

#### Step 3: Quintile-Based Logistic Regression
- Categorize each indicator into quintiles (Q1-Q5)
- Reference group: Q1
- Assess trend across quintiles (p for trend)

#### Step 4: Machine Learning Feature Importance
- **Model:** Gradient Boosting Classifier (GBM)
- **Method:** SHAP (SHapley Additive exPlanations) values
- **Cross-validation:** 10-fold CV
- **Output:** Ranked variable importance scores

#### Step 5: Risk Scoring System Development
- Select top 5 variables by SHAP importance
- Define low-risk ranges based on RCS curves and quintile analysis
- **Scoring rule:** 
  - Within optimal range = 0 points
  - Outside optimal range = 1 point
- **Total score range:** 0-5 points

#### Step 6: Score-Based Risk Stratification
- Categorize participants into risk groups:
  - Low risk: 0-1 points
  - Moderate risk: 2-3 points
  - High risk: 4-5 points
- Validate using grouped logistic regression
- Reference: Low-risk group
- Outcomes: OR (95% CI) for moderate and high-risk groups

---

### 2. Post Stroke mortality Analysis

**Location:** `Post_Stroke_mortality_Analysis/`

**Study Population:** Stroke patients from local hospital cohort  
**Outcomes:**
- All-cause mortality
- Cardiovascular disease (CVD) mortality
- Cardiovascular disease

**Analysis Pipeline:**

#### Step 1: Univariate Cox Regression
- Evaluate crude associations between obesity indicators and survival outcomes
- Adjusted for age and sex
- Outcomes: HR (95% CI), p-values

#### Step 2: Restricted Cubic Spline (RCS) Cox Models
- Model non-linear associations with time-to-event outcomes
- 3 knots at 10th, 50th, 90th percentiles
- Reference value: median
- Visualize HR curves with 95% CI

#### Step 3: Quintile-Based Cox Regression
- Categorize indicators into quintiles
- Reference: Q1
- Test proportional hazards assumption

#### Step 4: Machine Learning Survival Analysis
- **Model:** Gradient Boosting Survival Analysis (GBS)
- **Method:** SHAP values for survival models
- **Cross-validation:** 10-fold CV
- **Performance metric:** Concordance index (C-index)

#### Step 5: Prognostic Scoring System
- Select top 5 prognostic indicators by SHAP importance
- Define low-risk ranges from RCS and quintile analysis
- **Scoring:** Same as incident stroke (0-5 points)

#### Step 6: Score-Based Prognostic Validation
- Risk groups: 0-2, 3, 4-5 points
- Grouped Cox regression
- Kaplan-Meier survival curves
- Log-rank test for survival differences

---

## Statistical Methods

### Conventional Statistical Modeling

#### Logistic Regression (Incident Stroke)
```
logit(P(Stroke)) = β₀ + β₁·Obesity_Indicator + β₂·Age + β₃·Sex
```

#### Cox Proportional Hazards Model (Post-Stroke Mortality)
```
h(t) = h₀(t) × exp(β₁·Obesity_Indicator + β₂·Age + β₃·Sex)
```

#### Restricted Cubic Splines (RCS)
- **Knots:** in Cox 3 (at 10th, 50th, 90th percentiles)，in LR 2 (at 33th,66th percentiles)
- **Reference value:** Median
- **Implementation:** 
  - Logistic: `patsy` + `statsmodels`
  - Cox: `patsy` + `lifelines`
- **Non-linearity test:** Wald test

### Machine Learning Models

#### Gradient Boosting Classifier (GBM)
**Hyperparameters:**
- n_estimators: 100
- learning_rate: 0.05
- max_depth: 3
- subsample: 0.8
- random_state: 42

#### Gradient Boosting Survival Analysis (GBS)
**Same hyperparameters as GBM**
- Uses Cox loss function
- C-index for model evaluation

#### SHAP Value Analysis
- **Tree models:** TreeExplainer (exact algorithm)
- **Metric:** Mean absolute SHAP value across 10-fold CV
- **Output:** Variable importance ranking + standard deviation

### Risk Scoring System

**Score Calculation:**
```python
For each individual:
    Score = 0
    For each of top 5 variables:
        If value NOT in optimal range:
            Score += 1
    Final Score: 0-5 points
```


**Validation:**
- Logistic regression (incident stroke)
- Cox regression (post-stroke mortality)
---

## Obesity Indicators (n=11)

All indicators calculated from measured height, weight, waist circumference (WC), and hip circumference (HC):

1. **BMI** - Body Mass Index: Weight(kg) / Height(m)²
2. **WHR** - Waist-to-Hip Ratio: WC / HC
3. **WHtR** - Waist-to-Height Ratio: WC / Height
4. **HHtR** - Hip-to-Height Ratio: HC / Height
5. **ABSI** - A Body Shape Index
6. **BRI** - Body Roundness Index
7. **BF_percent** - Body Fat Percentage (estimated)
8. **Height** - Standing height (cm)
9. **Weight** - Body weight (kg)
10. **WC** - Waist circumference (cm)
11. **HC** - Hip circumference (cm)

---

## Data Requirements

### Incident Stroke Analysis
**Required variables:**
- Anthropometric measurements (baseline)
- Incident stroke outcomes (ICD-10: I60-I64)
- Stroke subtypes (hemorrhagic vs. ischemic)
- Demographic covariates (age, sex)

### Post-Stroke Prognosis
**Required variables:**
- Anthropometric measurements (admission/baseline)
- Survival outcomes (all-cause death, CVD death, CVD events)
- Follow-up time
- Demographic covariates (age, sex)

> **Note:** Individual-level data are not included in this repository due to privacy regulations. Users must obtain data access through appropriate institutional channels.

---

## Installation

### Python Environment
```bash
# Create conda environment
conda create -n stroke_risk python=3.8
conda activate stroke_risk

# Install dependencies
pip install -r requirements.txt
```

**Key packages:**
- `pandas >= 1.3.0`
- `numpy >= 1.20.0`
- `scikit-learn >= 1.0.0`
- `scikit-survival >= 0.17.0`
- `shap >= 0.41.0`
- `lifelines >= 0.27.0`
- `statsmodels >= 0.13.0`
- `patsy >= 0.5.0`
- `matplotlib >= 3.4.0`
- `seaborn >= 0.11.0`

### R Environment
```r
# Install required packages
install.packages(c("tidyverse", "corrplot", "ggcorrplot", "psych", "car", "GGally"))
```

---
## Validation and Scope

- All analyses adjusted for age and sex as baseline confounders
- Internal validation: 10-fold cross-validation for machine learning models
- Score performance evaluated using discrimination (AUC, C-index) and calibration metrics
- **Limitations:**
  - Observational design; causal inference not established
  - Single-center data for prognosis analysis
  - External validation in independent cohorts recommended

---

## License

This project is released under the **Apache-2.0 License**.

---

## Disclaimer

The risk scoring system is intended for **research purposes only** and has not been clinically validated.  
Clinical decisions should not be made based on these models without prospective validation and regulatory approval.
