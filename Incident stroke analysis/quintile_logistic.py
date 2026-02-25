"""
Quintile Logistic Regression Analysis
Each primary variable was grouped by quintiles, with Q1 as the reference, adjusted for age and gender.
"""
import os
threads_per_task = 6
os.environ["OMP_NUM_THREADS"] = str(threads_per_task)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_task)
os.environ["MKL_NUM_THREADS"] = str(threads_per_task)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads_per_task)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_task)

import pandas as pd
import numpy as np
from statsmodels.api import Logit, add_constant
from statsmodels.stats.multitest import fdrcorrection
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. Load data
# ===============================

var_file = "/3839.csv"
df = pd.read_csv(var_file)

df = df.applymap(
    lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
)

# ===============================
# 2. Define variables
# ===============================
main_vars = [
    'BMI', 'WHR', 'Height', 'Weight', 'WC', 'HC',
    'WHtR', 'HHtR', 'ABSI', 'BRI', 'BF_percent'
]
covariates = ['age', 'sex']
outcomes = ["subject",'Ischemic_Stroke', 'Hemorrhagic_Stroke']

for col in outcomes:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# ===============================
# 3. Quintile grouping function
# ===============================
def create_quintiles(data, var, outcome, adjust_vars):

    cols = [outcome, var] + adjust_vars

    df_model = data[cols].apply(pd.to_numeric, errors='coerce').dropna()

    if df_model.shape[0] < 100:
        return None, None
    
    if df_model[outcome].nunique() < 2:
        return None, None

    quintiles = df_model[var].quantile([0.2, 0.4, 0.6, 0.8]).values
    

    df_model[f'{var}_quintile'] = pd.cut(
        df_model[var],
        bins=[-np.inf] + quintiles.tolist() + [np.inf],
        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
        include_lowest=True
    )

    quintile_stats = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_data = df_model[df_model[f'{var}_quintile'] == q]
        quintile_stats.append({
            'quintile': q,
            'n': len(q_data),
            'events': int(q_data[outcome].sum()),
            'event_rate': q_data[outcome].mean(),
            'mean': q_data[var].mean(),
            'median': q_data[var].median(),
            'min': q_data[var].min(),
            'max': q_data[var].max()
        })
    
    quintile_stats_df = pd.DataFrame(quintile_stats)
    
    return df_model, quintile_stats_df

# ===============================
# 4. Quintile Logistic Regression
# ===============================
def run_quintile_logistic(data, var, outcome, adjust_vars):

    df_model, quintile_stats = create_quintiles(data, var, outcome, adjust_vars)
    
    if df_model is None:
        return None, None, None
    

    quintile_col = f'{var}_quintile'
    

    df_model_dummies = pd.get_dummies(
        df_model[quintile_col],
        prefix=var,
        drop_first=True
    ).astype(float)


    X = pd.concat([df_model_dummies, df_model[adjust_vars]], axis=1)
    X = add_constant(X)
    y = df_model[outcome]
    
    try:
        model = Logit(y, X).fit(disp=False, maxiter=300)

        results = []

        q1_stats = quintile_stats[quintile_stats['quintile'] == 'Q1'].iloc[0]
        results.append({
            'variable': var,
            'outcome': outcome,
            'quintile': 'Q1',
            'n': q1_stats['n'],
            'events': q1_stats['events'],
            'event_rate': q1_stats['event_rate'],
            'mean': q1_stats['mean'],
            'median': q1_stats['median'],
            'range': f"{q1_stats['min']:.2f}-{q1_stats['max']:.2f}",
            'OR': 1.0,
            'OR_LCI': np.nan,
            'OR_UCI': np.nan,
            'p_value': np.nan,
            'reference': 'Ref'
        })
        
        # Q2-Q5
        for q_num, quintile in enumerate(['Q2', 'Q3', 'Q4', 'Q5'], start=2):
            coef_name = f'{var}_{quintile}'
            
            if coef_name in model.params.index:
                coef = model.params[coef_name]
                se = model.bse[coef_name]
                pval = model.pvalues[coef_name]
                
                or_value = np.exp(coef)
                or_lci = np.exp(coef - 1.96 * se)
                or_uci = np.exp(coef + 1.96 * se)
                
                q_stats = quintile_stats[quintile_stats['quintile'] == quintile].iloc[0]
                
                results.append({
                    'variable': var,
                    'outcome': outcome,
                    'quintile': quintile,
                    'n': q_stats['n'],
                    'events': q_stats['events'],
                    'event_rate': q_stats['event_rate'],
                    'mean': q_stats['mean'],
                    'median': q_stats['median'],
                    'range': f"{q_stats['min']:.2f}-{q_stats['max']:.2f}",
                    'OR': or_value,
                    'OR_LCI': or_lci,
                    'OR_UCI': or_uci,
                    'p_value': pval,
                    'reference': ''
                })
        df_trend = df_model.copy()
        df_trend['quintile_num'] = df_trend[quintile_col].map({
            'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4, 'Q5': 5
        })
        
        X_trend = df_trend[['quintile_num'] + adjust_vars]
        X_trend = add_constant(X_trend)
        y_trend = df_trend[outcome]
        
        model_trend = Logit(y_trend, X_trend).fit(disp=False, maxiter=300)
        p_trend = model_trend.pvalues['quintile_num']
        
        results_df = pd.DataFrame(results)
        
        return results_df, quintile_stats, p_trend
        
    except Exception as e:
        print(f"  ✗ {var} - {outcome} error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ===============================
# 5. Main analysis workflow
# ===============================
all_results = []
all_quintile_stats = []
trend_test_results = []

output_dir = "/Quintile_Analysis"
os.makedirs(output_dir, exist_ok=True)

for outcome in outcomes:
    outcome_results = []
    
    for var in main_vars:
        if var not in df.columns:
            continue
        
        results_df, quintile_stats, p_trend = run_quintile_logistic(
            df, var, outcome, covariates
        )
        
        if results_df is not None:
            results_df['p_trend'] = p_trend
            
            all_results.append(results_df)
            outcome_results.append(results_df)
            quintile_stats['variable'] = var
            quintile_stats['outcome'] = outcome
            all_quintile_stats.append(quintile_stats)

            trend_test_results.append({
                'variable': var,
                'outcome': outcome,
                'p_trend': p_trend
            })

            print(f"\n  {var}:")
            print(f"    Q1 (reference): n={results_df.iloc[0]['n']}, "
                  f"events={results_df.iloc[0]['events']}, "
                  f"range={results_df.iloc[0]['range']}")
            
            for idx in range(1, len(results_df)):
                row = results_df.iloc[idx]
                print(f"    {row['quintile']}: "
                      f"OR={row['OR']:.3f} ({row['OR_LCI']:.3f}-{row['OR_UCI']:.3f}), "
                      f"p={row['p_value']:.4f}")

    if outcome_results:
        outcome_combined = pd.concat(outcome_results, ignore_index=True)
        outcome_file = os.path.join(output_dir, f"Quintile_{outcome}.csv")
        outcome_combined.to_csv(outcome_file, index=False)
        print(f"\n  ✅ {outcome}save: {outcome_file}")

# ===============================
# 6. Merge and save all results
# ===============================
if all_results:

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_file = os.path.join(output_dir, "Quintile_All_Results.csv")
    all_results_df.to_csv(all_results_file, index=False)

    if all_quintile_stats:
        quintile_stats_df = pd.concat(all_quintile_stats, ignore_index=True)
        stats_file = os.path.join(output_dir, "Quintile_Statistics.csv")
        quintile_stats_df.to_csv(stats_file, index=False)

    if trend_test_results:
        trend_df = pd.DataFrame(trend_test_results)

        for outcome in outcomes:
            mask = trend_df['outcome'] == outcome
            if mask.sum() > 0:
                _, padj = fdrcorrection(trend_df.loc[mask, 'p_trend'].values)
                trend_df.loc[mask, 'p_trend_adj'] = padj
        
        trend_file = os.path.join(output_dir, "Quintile_Trend_Tests.csv")
        trend_df.to_csv(trend_file, index=False)

