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
# Load data
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
var_file = "/2467_UKB.csv"
var_data = pd.read_csv(var_file)

df = var_data.applymap(
    lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
)
df = df.replace(r'^\s*$', np.nan, regex=True)
# ===============================
# Delete unnecessary columns
# ===============================
drop_cols = [
    'data完整', 'dataset完整临床',
    'subject', 'subtype', '发病日期','thrombosis', 'lacunar', 'ICH'
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ===============================
# Convert the date column to datetime format
# ===============================
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
# Outcome
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

# =====================================================
# Unified cleaning of covariates (numericalization + removal of invalid values)
# =====================================================
merged_df = df_clean.applymap(
    lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
)

base_vars = ['Height', 'Weight', 'WC', 'HC']

for col in base_vars:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
# =====================================================
# Define the variable
# =====================================================
main_vars = [
    'BMI', 'WHR', 'Height', 'Weight', 'WC', 'HC',
    'WHtR', 'HHtR', 'ABSI', 'BRI', 'BF_percent'
]

covariates_1 = ['Age', 'Sex']

all_vars = main_vars + covariates_1 + ['death', 'cvddeath', 'cvdevent', 'death_time', 'cvddeath_time', 'cvdevent_time']
for col in all_vars:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ===============================
# Quintile grouping function
# ===============================
def create_groups(data, vars_to_group):
    d = data.copy()
    for var in vars_to_group:
        if var not in d.columns:
            d[f'{var}_group'] = np.nan
            continue
        
        try:
            d[f'{var}_group'] = pd.qcut(
                d[var],
                q=5,
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
            )
        except Exception:
            d[f'{var}_group'] = np.nan
    
    return d

df = create_groups(df, main_vars)

# ===============================
# Core Cox analysis function
# ===============================
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

        # 2. 添加参照组行
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

# ===============================
# Execute the fully automated analysis loop
# ===============================
outcome_config = {
    "death": {"event": "death", "time": "death_time"},
    "cvddeath": {"event": "cvddeath", "time": "cvddeath_time"},
    "cvdevent": {"event": "cvdevent", "time": "cvdevent_time"}
}

body_comp_groups = [c for c in df.columns if c.endswith('_group')]

for out_name, config in outcome_config.items():
    event_col = config['event']
    time_col = config['time']

    analysis_df = df.dropna(subset=[event_col, time_col])
    analysis_df = analysis_df[analysis_df[time_col] > 0]
    print(
    f"{out_name}: N={analysis_df.shape[0]}, "
    f"events={analysis_df[event_col].sum()}"
)
    all_out_results = []
    
    for g_var in tqdm(body_comp_groups, desc=f"Analyzing {out_name}"):
        ref = 'Q1'
        res = Grouped_Cox_analysis(
            data=analysis_df,
            group_var=g_var,
            time_col=time_col,
            event_col=event_col,
            outcome_type=out_name,
            adjust_vars=covariates_1,
            reference_group=ref
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
        print(f"✅ {out_name} save: {save_path}")
