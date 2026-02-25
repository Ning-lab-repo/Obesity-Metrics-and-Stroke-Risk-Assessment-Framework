import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ===============================
# 1. Load data
# ===============================
threads_per_task = 6
os.environ.update({
    "OMP_NUM_THREADS": str(threads_per_task),
    "OPENBLAS_NUM_THREADS": str(threads_per_task),
    "MKL_NUM_THREADS": str(threads_per_task),
    "VECLIB_MAXIMUM_THREADS": str(threads_per_task),
    "NUMEXPR_NUM_THREADS": str(threads_per_task)
})
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
data_path = "/2467_UKB.csv"
df = pd.read_csv(data_path)
df = df.replace(r'^\s*$', np.nan, regex=True)
date_cols = ['入组时间', 'death_date', '终止时间',"cvdevent_data","cvddeath_data"]
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# ===============================
# Calculate the time to onset (unit: days)
# ===============================
df['death_time'] = (
    df['death_date']
    .fillna(df['终止时间'])
    - df['入组时间']
).dt.days

df['cvddeath_time'] = (
    df['cvddeath_data']
    .fillna(df['death_date'])
    .fillna(df['终止时间'])
    - df['入组时间']
).dt.days
df['cvdevent_time'] = (
    df['cvdevent_data']
    .fillna(df['death_date'])
    .fillna(df['终止时间'])
    - df['入组时间']
).dt.days
# ===============================
# 6. Outcome
# ===============================
outcome_cols = ['death', 'cvddeath', 'cvdevent']

for col in outcome_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
required_cols = (
    outcome_cols +
    ['death_time', 'cvddeath_time', 'cvdevent_time']
)
df_clean = df.dropna(subset=required_cols)
df_clean = df_clean[
    (df_clean['death_time'] >= 0) &
    (df_clean['cvddeath_time'] >= 0) &
    (df_clean['cvdevent_time'] >= 0)
]
# ===============================
# 3. Define the variable
# ===============================
main_vars = ['BMI', 'WHR', 'Height', 'Weight', 'WC', 'HC', 'WHtR', 'HHtR', 'ABSI', 'BRI', 'BF_percent']
covariates = ['Age','Sex']

all_vars = main_vars + covariates + ['death', 'cvddeath', 'cvdevent', 'death_time', 'cvddeath_time', 'cvdevent_time']
for col in all_vars:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

score_rules = pd.DataFrame([
    # ========== death | rule_based ==========
    ["death", "GBM", "SHAP", "ABSI", None, 0.80],
    ["death", "GBM", "SHAP", "WHR", 0.75, 0.94],
    ["death", "GBM", "SHAP","BF_percent", 37.25, 43.14],
    ["death", "GBM", "SHAP","Weight", 72.90, 90.05],
    ["death", "GBM", "SHAP", "Height", 177.50, None],

    # ========== cvdevent | rule_based ==========
    ["cvdevent", "GBM", "SHAP",  "WHR", None, 0.94],
    ["cvdevent", "GBM", "SHAP", "Weight", 68.50, 85.20],
    ["cvdevent", "GBM", "SHAP", "WC", None, 99.00],
    ["cvdevent", "GBM", "SHAP", "BRI", None, 6.16],
    ["cvdevent", "GBM", "SHAP", "WHtR", None, 0.58],

    # ========== cvddeath | rule_based ==========
    ["cvddeath", "GBM", "SHAP", "WHR", None, 0.94],
    ["cvddeath", "GBM", "SHAP", "ABSI", None, 0.80],
    ["cvddeath", "GBM", "SHAP", "Weight", 72.90, 90.05],
    ["cvddeath", "GBM", "SHAP", "WC", None, 100.00],
    ["cvddeath", "GBM", "SHAP", "Height", None, 161.50],
    ["cvddeath", "GBM", "SHAP", "Height", 177.50, None],

], columns=["outcome", "model", "method", "variable", "low", "high"])

def compute_risk_scores(df, rules):
    df = df.copy()

    for (outcome, model, method), sub in rules.groupby(
        ["outcome", "model", "method"]
    ):
        score_col = f"{outcome}_{model}_{method}"
        df[score_col] = 0

        for _, r in sub.iterrows():
            x = df[r["variable"]]
            low, high = r["low"], r["high"]

            if pd.isna(low):
                flag = x >= high
            elif pd.isna(high):
                flag = x <= low
            else:
                flag = ~x.between(low, high, inclusive="both")

            df.loc[flag, score_col] += 1

    return df


df = compute_risk_scores(df_clean, score_rules)


def score_to_group(s):
    return pd.cut(
        s,
        bins=[-0.1, 2, 4, 5],
        labels=["0-2", "3-4", "5"]
    )


score_cols = [
    f"{o}_{m}_{me}"
    for o in ["death", "cvddeath","cvdevent"]
    for m in ["GBM"]
    for me in ["SHAP"]
]

for col in score_cols:
    df[f"{col}_group"] = score_to_group(df[col])

print(df[score_cols].describe())

for col in score_cols:
    df[f"{col}_group"] = score_to_group(df[col])
score_group_vars = [f"{c}_group" for c in score_cols]



def Grouped_Cox_analysis(data, group_var, time_col, event_col, outcome_type, adjust_vars, reference_group=None):
    base_var = group_var.replace('_group', '')
    cols_to_use = [group_var, base_var, time_col, event_col] + adjust_vars
    model_data = data[cols_to_use].dropna().reset_index(drop=True)

    if model_data.shape[0] < 50 or model_data[event_col].sum() < 5:
        return None
    continuous_adjust = [v for v in adjust_vars if model_data[v].nunique() > 2]
    if continuous_adjust:
        scaler = StandardScaler()
        model_data[continuous_adjust] = scaler.fit_transform(model_data[continuous_adjust])
    groups = [g for g in model_data[group_var].unique() if pd.notna(g)]
    if reference_group not in groups:
        reference_group = groups[0]
    dummy_df = pd.get_dummies(model_data[group_var], prefix=group_var)
    ref_col = f"{group_var}_{reference_group}"
    dummy_cols = [c for c in dummy_df.columns if c != ref_col]
    
    final_model_data = pd.concat([model_data[[time_col, event_col] + adjust_vars], dummy_df[dummy_cols]], axis=1)

    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(final_model_data, duration_col=time_col, event_col=event_col)
        summary = cph.summary
        
        results = []
        for col in dummy_cols:
            group_name = col.replace(f"{group_var}_", "")
            mask = model_data[group_var] == group_name
            
            results.append({
                "variable": group_var,
                "group": group_name,
                "reference": reference_group,
                "hr": np.exp(summary.loc[col, 'coef']),
                "lci": np.exp(summary.loc[col, 'coef'] - 1.96 * summary.loc[col, 'se(coef)']),
                "uci": np.exp(summary.loc[col, 'coef'] + 1.96 * summary.loc[col, 'se(coef)']),
                "p_val": summary.loc[col, 'p'],
                "n": int(mask.sum()),
                "events": int(model_data.loc[mask, event_col].sum()),
                "range": f"{model_data.loc[mask, base_var].min():.2f}-{model_data.loc[mask, base_var].max():.2f}",
                "outcome": outcome_type
            })
        ref_mask = model_data[group_var] == reference_group
        results.append({
            "variable": group_var, "group": reference_group, "reference": reference_group,
            "hr": 1.0, "lci": 1.0, "uci": 1.0, "p_val": np.nan,
            "n": int(ref_mask.sum()), "events": int(model_data.loc[ref_mask, event_col].sum()),
            "range": f"{model_data.loc[ref_mask, base_var].min():.2f}-{model_data.loc[ref_mask, base_var].max():.2f}",
            "outcome": outcome_type
        })
        
        return pd.DataFrame(results)
    except Exception as e:
        print(f"  ⚠ error： ({group_var}): {e}")
        return None

outcome_config = {
    "death": {"event": "death", "time": "death_time"},
    "cvddeath": {"event": "cvddeath", "time": "cvddeath_time"},
    "cvdevent": {"event": "cvdevent", "time": "cvdevent_time"}
}
all_out_results = []
for out_name, config in outcome_config.items():
    for g_var in score_group_vars:
        if not g_var.startswith(out_name):
            continue

        res = Grouped_Cox_analysis(
            data=df,
            group_var=g_var,
            time_col=config["time"],
            event_col=config["event"],
            outcome_type=out_name,
            adjust_vars=covariates,
            reference_group="0-2"
        )
        if res is not None:
            all_out_results.append(res)

    if all_out_results:
        final_res = pd.concat(all_out_results, ignore_index=True)

        valid_p = final_res['p_val'].dropna()
        if not valid_p.empty:
            _, padj = fdrcorrection(valid_p)
            final_res.loc[final_res['p_val'].notna(), 'p_adj'] = padj

        save_dir = f"/{out_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/grouped_cox_results.csv"
        final_res.to_csv(save_path, index=False)
        print(f"✅ {out_name} Save: {save_path}")

