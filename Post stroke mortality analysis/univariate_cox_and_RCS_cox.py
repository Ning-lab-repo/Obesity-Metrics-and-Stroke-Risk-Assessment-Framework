import os
threads_per_task = 6
os.environ["OMP_NUM_THREADS"] = str(threads_per_task)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_task)
os.environ["MKL_NUM_THREADS"] = str(threads_per_task)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads_per_task)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_task)
from lifelines import CoxPHFitter
from patsy import dmatrix, build_design_matrices
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from statsmodels.api import Logit, MNLogit, add_constant
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from statsmodels.api import Logit, MNLogit, add_constant
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import chi2
# ===============================
# 1. Load data
# ===============================

var_file = "/2467_UKB.csv"
var_data = pd.read_csv(var_file)


df = var_data.applymap(
    lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
)

# ===============================
# 2. Delete unnecessary columns
# ===============================
drop_cols = [
    'data完整', 'dataset完整临床',
    'subject', 'subtype', '发病日期','thrombosis', 'lacunar', 'ICH'
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ===============================
# 3. Convert the date column to datetime format
# ===============================
date_cols = ['入组时间', 'death_date', '终止时间',"cvdevent_data","cvddeath_data"]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# ===============================
# 4. Calculate the time to onset (unit: days)
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
# 5. Outcome
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

merged_df = df.applymap(
    lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
)

base_vars = ['Height', 'Weight', 'WC', 'HC']

for col in base_vars:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
# =====================================================
# 6. Define the variable
# =====================================================
main_vars = [
    'BMI', 'WHR', 'Height', 'Weight', 'WC', 'HC',
    'WHtR', 'HHtR', 'ABSI', 'BRI', 'BF_percent'
]

covariates_1 = ['Age', 'Sex']
all_model_vars = list(set(main_vars + covariates_1 + covariates_2))

# =====================================================
# 7. Unified cleaning of covariates (numericalization + removal of invalid values)
# =====================================================
for col in all_model_vars:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
merged_df[all_model_vars] = merged_df[all_model_vars].applymap(
    lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
)
binary_vars = []
continuous_vars = []

for col in all_model_vars:
    if col in merged_df.columns:
        unique_vals = merged_df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}) or set(unique_vals).issubset({1, 2}):
            binary_vars.append(col)
        else:
            continuous_vars.append(col)


def rcs_analysis_with_viz_data(
    data,
    var_name,
    time_col,
    event_col,
    adjust_vars=None,
    knot_percentiles=(10, 50, 90),
    penalizer=0.1,
    n_points=100
):

    
    if adjust_vars is None:
        adjust_vars = []


    use_cols = [var_name, time_col, event_col] + adjust_vars
    rcs_data = data[use_cols].dropna().reset_index(drop=True)
    if rcs_data.shape[0] < 50:
        raise ValueError("RCS analysis suspended: insufficient sample size")

    x = rcs_data[var_name].values
    knots = np.percentile(x, knot_percentiles)
    knots_list = [float(k) for k in knots]

    safe_var = f"Q('{var_name}')"
    safe_adjust = [f"Q('{v}')" for v in adjust_vars]
    rcs_formula = f"cc({safe_var}, knots={knots_list})"
    if safe_adjust:
        rcs_formula += " + " + " + ".join(safe_adjust)

    X_rcs = dmatrix(rcs_formula, rcs_data, return_type="dataframe")
    design_info = X_rcs.design_info
    X_rcs = X_rcs.drop(columns="Intercept")
    X_rcs[time_col] = rcs_data[time_col].values
    X_rcs[event_col] = rcs_data[event_col].values

    cph_rcs = CoxPHFitter(penalizer=penalizer)
    cph_rcs.fit(X_rcs, duration_col=time_col, event_col=event_col)

    x_min, x_max = np.percentile(x, [2.5, 97.5])
    x_range = np.linspace(x_min, x_max, n_points)

    predict_df = pd.DataFrame({var_name: x_range})
    for v in adjust_vars:
        predict_df[v] = rcs_data[v].mean()

    X_pred = build_design_matrices([design_info], predict_df)[0]
    X_pred = pd.DataFrame(X_pred, columns=design_info.column_names)
    X_pred = X_pred.drop(columns="Intercept")

    preds = cph_rcs.predict_partial_hazard(X_pred).values

    ref_value = float(np.median(x))
    ref_df = pd.DataFrame({var_name: [ref_value]})
    for v in adjust_vars:
        ref_df[v] = rcs_data[v].mean()

    X_ref = build_design_matrices([design_info], ref_df)[0]
    X_ref = pd.DataFrame(X_ref, columns=design_info.column_names)
    X_ref = X_ref.drop(columns="Intercept")
    ref_pred = cph_rcs.predict_partial_hazard(X_ref).values[0]

    relative_hr = preds / ref_pred
    log_hr = np.log(relative_hr)

    spline_cols = [col for col in X_pred.columns if 'cc(' in col]
    coef_spline = cph_rcs.params_[spline_cols].values
    vcov_spline = cph_rcs.variance_matrix_.loc[spline_cols, spline_cols].values

    X_spline = X_pred[spline_cols].values
    X_ref_spline = X_ref[spline_cols].values
    
    se_log_hr = []
    for i in range(len(X_spline)):
        diff = X_spline[i] - X_ref_spline[0]
        var = diff @ vcov_spline @ diff.T
        se_log_hr.append(np.sqrt(max(var, 0)))
    
    se_log_hr = np.array(se_log_hr)

    hr_lower = np.exp(log_hr - 1.96 * se_log_hr)
    hr_upper = np.exp(log_hr + 1.96 * se_log_hr)

    plot_df = pd.DataFrame({
        "value": x_range,
        "HR": relative_hr,
        "HR_lower": hr_lower,
        "HR_upper": hr_upper,
        "log_HR": log_hr,
        "se_log_HR": se_log_hr
    })

    n_bins = 30
    hist_bins = np.linspace(x_min, x_max, n_bins + 1)

    histogram_data = []
    for i in range(n_bins):
        bin_mask = (rcs_data[var_name] >= hist_bins[i]) & (rcs_data[var_name] < hist_bins[i+1])
        if i == n_bins - 1:  # 最后一个bin包含右边界
            bin_mask = (rcs_data[var_name] >= hist_bins[i]) & (rcs_data[var_name] <= hist_bins[i+1])
        
        bin_center = (hist_bins[i] + hist_bins[i+1]) / 2
        total_n = bin_mask.sum()
        event_n = rcs_data.loc[bin_mask, event_col].sum()
        
        histogram_data.append({
            'bin_left': hist_bins[i],
            'bin_right': hist_bins[i+1],
            'bin_center': bin_center,
            'total_n': total_n,
            'event_n': event_n,
            'event_rate': event_n / total_n if total_n > 0 else 0
        })
    
    histogram_df = pd.DataFrame(histogram_data)

    summary_info = {
        'knots': knots_list,
        'ref_value': ref_value,
        'x_min': float(x_min),
        'x_max': float(x_max),
        'total_n': len(rcs_data),
        'total_events': int(rcs_data[event_col].sum()),
        'event_rate': float(rcs_data[event_col].mean())
    }

    return plot_df, histogram_df, cph_rcs, summary_info

def run_rcs_analysis(model_df,var,covariates, outcome_type):

    df = model_df.copy()
    required_cols = [var, "follow_up_time", "event"]+covariates
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ {var}Skip: missing columns{missing_cols}")
        return
    
    try:
        rcs_plot_data, rcs_histogram_data, cph_rcs, rcs_summary = rcs_analysis_with_viz_data(
            df, var,
            "follow_up_time",
            "event",
            covariates,
            knot_percentiles=[10, 50, 90],
            n_points=100
        )
        
        output_dir = f'/{outcome_type}/{var}/'
        os.makedirs(output_dir, exist_ok=True)
        
        rcs_plot_data.to_csv(f'{output_dir}rcs_hr_curve.csv', index=False)
        rcs_histogram_data.to_csv(f'{output_dir}rcs_histogram.csv', index=False)
        print(rcs_plot_data.shape)
        pd.DataFrame([rcs_summary]).to_csv(f'{output_dir}rcs_summary.csv', index=False)
        
        print(f"✓ {outcome_type} - {var} 完成")
        
    except Exception as e:
        print(f"❌ {outcome_type} - {var} 失败: {str(e)}")

def Cox_analysis(data,
    var_name,
    time_col,
    event_col,
    outcome_type,
    adjust_vars=None):
    use_cols = [var_name, time_col, event_col] + adjust_vars
    model_data4 = data[use_cols].dropna().reset_index(drop=True)
    if model_data4.shape[0] < 50:
        raise ValueError("RCS analysis suspended: insufficient sample size")
    scaler = StandardScaler()
    fei_cvar = ["Sex","Smoke","Alcohol","Summed_minutes_activity","HTNhis","CHDhis","RENhis","hyperlipid","Diabhis","hypertension","spirin","antihyper"]
    ac_vars = [v for v in adjust_vars if v not in fei_cvar]
    b_cols = [var_name]+ac_vars
    model_data4[b_cols] = scaler.fit_transform(model_data4[b_cols])

    model_data4[event_col] = pd.to_numeric(model_data4[event_col], errors='coerce')

    dead_count4 = (model_data4[event_col] == 1).sum()
    cph4 = CoxPHFitter(penalizer=0.1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        result_p4 = []
        try:
            cph4.fit(model_data4, duration_col=time_col, event_col=event_col)
            coef4 = cph4.summary
            hr4 = coef4['coef'].apply(np.exp)
            lci4 = (coef4['coef'] - 1.96*coef4['se(coef)']).apply(np.exp)
            uci4 = (coef4['coef'] + 1.96*coef4['se(coef)']).apply(np.exp)
            p_val4 = coef4['p']

            result_p4.append(pd.DataFrame({
                "variable": [var],
                "hr": [hr4[var]],
                "lci": [lci4[var]],
                "uci": [uci4[var]],
                "p_val": [p_val4[var]],
                "model": ["Full Model-BMI"],
                "se": [(np.log(uci4[var]) - np.log(hr4[var]))/1.96],
                "n": [model_data4.shape[0]],
                "dead": [int(dead_count4)]
            }))
        except Exception as e:
            print(f"error: {e}")
    def adjust_and_concat(results_list):
        if results_list:
            df = pd.concat(results_list, ignore_index=True)
            reject, padj = fdrcorrection(df["p_val"].values)
            df["padj"] = padj
            return df
        else:
            return pd.DataFrame()
    final_p4 = adjust_and_concat(result_p4)
    result_all = pd.concat([final_p4], ignore_index=True)

    if not result_all.empty:
        return result_all
    else:
        print("No results to save")

outcome_map = {
    "death": {
        "event": "death",
        "time": "death_time"
    },
    "cvddeath": {
        "event": "cvddeath",
        "time": "cvddeath_time"
    },
    "cvdevent": {
        "event": "cvdevent",
        "time": "cvdevent_time"
    }
}


all_cox_results = []

for outcome_type, omap in outcome_map.items():

    event_col = omap["event"]
    time_col = omap["time"]

    for var in main_vars:

        for model_name, covariates in {
            "Model1_age_sex": covariates_1,
        }.items():

            print(f"\n▶ {outcome_type} | {var} | {model_name}")

            model_df = merged_df.copy()

            model_df = model_df.rename(columns={
                time_col: "follow_up_time",
                event_col: "event"
            })

            use_cols = [var, "follow_up_time", "event"] + covariates
            model_df = model_df[use_cols].dropna()

            if model_df.shape[0] < 50:
                print("⚠️ Insufficient sample size, skipped.")
                continue

            # ===============================
            # Cox analysis
            # ===============================
            try:
                cox_res = Cox_analysis(
                    data=model_df,
                    var_name=var,
                    time_col="follow_up_time",
                    event_col="event",
                    outcome_type=outcome_type,
                    adjust_vars=covariates
                )

                if cox_res is not None and not cox_res.empty:
                    cox_res["outcome"] = outcome_type
                    cox_res["model"] = model_name
                    all_cox_results.append(cox_res)

            except Exception as e:
                print(f"❌ Cox error: {e}")

            # ===============================
            # RCS analysis
            # ===============================
            try:
                run_rcs_analysis(
                    model_df=model_df,
                    var=var,
                    covariates=covariates,
                    outcome_type=f"{outcome_type}_{model_name}"
                )

            except Exception as e:
                print(f"❌ RCS error: {e}")
if all_cox_results:
    final_cox_df = pd.concat(all_cox_results, ignore_index=True)

    output_dir = "/Cox_all.csv"
    final_cox_df.to_csv(output_dir, index=False)
else:
    print("⚠️ No Cox results generated.")