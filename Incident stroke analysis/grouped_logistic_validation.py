"""
Classification Prediction Model - Scoring System Based on Variable Ranges and Grouped Logistic Regression Analysis
Scores are constructed using the top 5 variables ranked by importance, followed by grouped analysis.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. Load data
# ===============================

DATA_PATH = "/3839.csv"
OUTPUT_DIR = "/ML_Classification_3839"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Main vars
MAIN_VARS = [
    'BMI', 'WHR', 'Height', 'Weight', 'WC', 'HC',
    'WHtR', 'HHtR', 'ABSI', 'BRI', 'BF_percent'
]

# Covariate
COVARIATES = ['age', 'sex']

BINARY_VARS = ['sex', 'Smoke', 'Alcohol', 'DM']
CONTINUOUS_VARS = [v for v in MAIN_VARS + COVARIATES if v not in BINARY_VARS]

# Outcomes
OUTCOMES = {
    'subject': 'subject',
    'Hemorrhagic_Stroke': 'Hemorrhagic_Stroke',
    'Ischemic_Stroke': 'Ischemic_Stroke'
}

# ===============================
# 2. Scoring rule definition
# ===============================

# Scoring rule: {Outcome: {Method: {Model: [(Variable, Range)]}}}
SCORING_RULES = {
    'Hemorrhagic_Stroke': {
        'Importance': {
            'GBM': [
                ('HC', 91.00, 105.00),
                ('HHtR', 0.54, 0.64),
                ('WHR', 0.79, 0.90),
                ('Weight', 60.00, 71.5),
                ('BRI', 3.18, 4.87)
            ],
            'Logistic': [
                ('WHtR', 0.49, 0.57),
                ('WC', 86.00, 95.00),
                ('BF_percent', 18.34, 26.89),
                ('WHR', 0.79, 0.90),
                ('BRI', 3.18, 4.87)
            ]
        },
        'SHAP': {
            'GBM': [
                ('HHtR', 0.54, 0.64),
                ('HC', 91.00, 105.00),
                ('WHR', 0.79, 0.90),
                ('BMI', 22.49, 25.82),
                ('Height', None, 167)  # <167
            ],
            'Logistic': [
                ('WHtR', 0.49, 0.57),
                ('BRI', 3.18, 4.87),
                ('WC', 86.00, 95.00),
                ('BMI', 22.49, 25.82),
                ('BF_percent', 18.34, 26.89)
            ]
        }
    },
    'Ischemic_Stroke': {
        'Importance': {
            'GBM': [
                ('WHR', 0.82, 1.88),
                ('Weight', None, 62.20),  # <62.20
                ('HC', 91.00, 98.00),
                ('BRI', 3.18, 4.09),
                ('WC', None, 83.00)  # <83.00
            ],
            'Logistic': [
                ('BRI', 3.18, 4.09),
                ('WHR', 0.82, 1.88),
                ('HHtR', 0.58, 0.62),
                ('HC', 91.00, 98.00),
                ('WHtR', 0.49, 0.54)
            ]
        },
        'SHAP': {
            'GBM': [
                ('WHR', 0.82, 1.88),
                ('HHtR', 0.58, 0.62),
                ('Weight', None, 62.20),  # <62.20
                ('HC', 91.00, 98.00),
                ('Height', None, 162)  # <162
            ],
            'Logistic': [
                ('BRI', 3.18, 4.09),
                ('HC', 91.00, 98.00),
                ('HHtR', 0.58, 0.62),
                ('WHtR', 0.49, 0.54),
                ('WC', None, 83.00)  # <83.00
            ]
        }
    },
    'subject': {
        'Importance': {
            'GBM': [
                ('HC', 91.00, 100.00),
                ('HHtR', 0.58, 0.64),
                ('WHR', None, 0.87),  # <0.87
                ('Weight', 55.70, 65.00),
                ('BRI', 3.48, 4.46)
            ],
            'Logistic': [
                ('WHtR', 0.47, 0.57),
                ('WC', 80.00, 91.00),
                ('BF_percent', None, 23.39),  # <23.39
                ('WHR', None, 0.87),  # <0.87
                ('BRI', 3.48, 4.46)
            ]
        },
        'SHAP': {
            'GBM': [
                ('WHR', None, 0.87),  # <0.87
                ('HC', 91.00, 100.00),
                ('HHtR', 0.58, 0.64),
                ('Weight', 55.70, 65.00),
                ('Height', None, 160)  # <160
            ],
            'Logistic': [
                ('BRI', 3.48, 4.46),
                ('WHtR', 0.47, 0.57),
                ('HC', 91.00, 100.00),
                ('WC', 80.00, 91.00),
                ('HHtR', 0.58, 0.64)
            ]
        }
    }
}

# ===============================
# 3. Data processing function
# ===============================

def load_and_clean_data(data_path):
    print("=" * 80)
    print("数据读取和基础清理")
    print("=" * 80)
    
    df = pd.read_csv(data_path)
    print(f"\n原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
    
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    return df

def impute_missing_values(df, continuous_vars, binary_vars):
    df_imputed = df.copy()
    n_before = len(df_imputed)
    
    for var in binary_vars:
        if var in df_imputed.columns:
            n_missing = df_imputed[var].isna().sum()
            if n_missing > 0:
                df_imputed = df_imputed[df_imputed[var].notna()]
    
    n_after = len(df_imputed)
    if n_before > n_after:
        print(f"\n  Total deleted: {n_before - n_after} Row")
        print(f"  Remaining samples: {n_after} Row")

    continuous_to_impute = [v for v in continuous_vars if v in df_imputed.columns]
    
    if continuous_to_impute:
        imputer = SimpleImputer(strategy='median')
        df_imputed[continuous_to_impute] = imputer.fit_transform(
            df_imputed[continuous_to_impute]
        )

    return df_imputed

# ===============================
# 4. Scoring calculation function
# ===============================

def calculate_score(value, lower, upper):

    if pd.isna(value):
        return np.nan
    

    if lower is None and upper is not None:
        return 0 if value < upper else 1
    

    elif lower is not None and upper is not None:
        return 0 if lower <= value <= upper else 1
    

    elif lower is not None and upper is None:
        return 0 if value > lower else 1
    
    return np.nan

def create_scores(df, scoring_rules):
    df_scores = df.copy()
    score_columns = []
    
    for outcome in scoring_rules.keys():
        for method in scoring_rules[outcome].keys():
            for model in scoring_rules[outcome][method].keys():
                
                score_name = f"{outcome}_{method}_{model}_score"
                score_columns.append(score_name)
                
                rules = scoring_rules[outcome][method][model]
                

                df_scores[score_name] = 0
                

                for var, lower, upper in rules:
                    if var in df_scores.columns:
                        var_scores = df_scores[var].apply(
                            lambda x: calculate_score(x, lower, upper)
                        )
                        df_scores[score_name] += var_scores
                

                print(f"\n{score_name}:")
                print(df_scores[score_name].value_counts().sort_index())
    
    return df_scores, score_columns

def create_score_groups(df, score_columns):
    df_grouped = df.copy()
    group_columns = []
    
    for score_col in score_columns:
        group_col = score_col.replace('_score', '_group')
        group_columns.append(group_col)
        

        df_grouped[group_col] = pd.cut(
            df_grouped[score_col],
            bins=[-0.5, 1.5, 3.5, 5.5],
            labels=['0-1', '2-3', '4-5'],
            include_lowest=True
        )
        

        print(f"\n{group_col}:")
        print(df_grouped[group_col].value_counts().sort_index())
    
    return df_grouped, group_columns

# ===============================
# 5. Grouped logistic regression analysis
# ===============================
def perform_grouped_logistic_regression(df, outcome_col, group_col, covariates):

    df_analysis = df[[outcome_col, group_col] + covariates].dropna()


    df_analysis[group_col] = df_analysis[group_col].astype(str)


    dummies = pd.get_dummies(df_analysis[group_col], prefix='group')
    if 'group_0-1' in dummies.columns:
        dummies = dummies.drop(columns=['group_0-1'])


    X = pd.concat([dummies, df_analysis[covariates]], axis=1)
    X = add_constant(X)
    X = X.astype(float)
    y = df_analysis[outcome_col].astype(float)

    try:
        model = Logit(y, X)
        res = model.fit(disp=0)

        results = {}


        mask_ref = df_analysis[group_col] == '0-1'
        results['group_0-1'] = {
            'OR': 1.0,
            'CI_lower': 1.0,
            'CI_upper': 1.0,
            'p_value': np.nan,
            'n_cases': int(df_analysis.loc[mask_ref, outcome_col].sum()),
            'n_total': int(mask_ref.sum())
        }


        for term in ['group_2-3', 'group_4-5']:
            if term in res.params.index:
                coef = res.params[term]
                se = res.bse[term]
                OR = np.exp(coef)
                CI_lower = np.exp(coef - 1.96 * se)
                CI_upper = np.exp(coef + 1.96 * se)
                p_value = res.pvalues[term]

                # 统计该组病例数和总数
                group_name = term.split('_')[1]  # '2-3' 或 '4-5'
                mask = df_analysis[group_col] == group_name
                n_total_group = mask.sum()
                n_cases_group = df_analysis.loc[mask, outcome_col].sum()

                results[term] = {
                    'OR': OR,
                    'CI_lower': CI_lower,
                    'CI_upper': CI_upper,
                    'p_value': p_value,
                    'n_cases': int(n_cases_group),
                    'n_total': int(n_total_group)
                }

        return results

    except Exception as e:
        print(f"  ⚠️ Logit 拟合失败: {e}")
        return None

def analyze_all_scores(df, score_columns, group_columns, outcomes, covariates):
    all_results = {}
    
    for score_col, group_col in zip(score_columns, group_columns):
        parts = score_col.split('_')
        outcome_name = parts[0]
        method = parts[1]
        model = parts[2]
        
        if outcome_name not in outcomes:
            continue
        
        outcome_col = outcomes[outcome_name]
        
        results = perform_grouped_logistic_regression(
            df, outcome_col, group_col, covariates
        )
        
        if results:
            all_results[score_col] = {
                'outcome': outcome_name,
                'method': method,
                'model': model,
                'results': results
            }
            
            # 打印结果
            for group in ['group_0-1', 'group_2-3', 'group_4-5']:
                r = results[group]
                if pd.isna(r['p_value']):
                    print(f"    {group}: OR=1.00 (reference)")
                else:
                    print(f"    {group}: OR={r['OR']:.2f}, 95%CI=[{r['CI_lower']:.2f}, {r['CI_upper']:.2f}], p={r['p_value']:.4f}")
    
    return all_results

# ===============================
# 6. Save results
# ===============================

def save_results(all_results, output_dir):

    results_list = []
    
    for score_name, result_dict in all_results.items():
        outcome = result_dict['outcome']
        method = result_dict['method']
        model = result_dict['model']
        results = result_dict['results']
        
        for group, r in results.items():
            results_list.append({
                'Score': score_name,
                'Outcome': outcome,
                'Method': method,
                'Model': model,
                'Group': group,
                'OR': r['OR'],
                'CI_lower': r['CI_lower'],
                'CI_upper': r['CI_upper'],
                'p_value': r['p_value'],
                'n_cases': r['n_cases'],
                'n_total': r['n_total']
            })
    
    results_df = pd.DataFrame(results_list)
    
    save_path = os.path.join(output_dir, 'all_logistic_regression_results.csv')
    results_df.to_csv(save_path, index=False)


# ===============================
# 8. Main workflow
# ===============================

if __name__ == "__main__":
    df = load_and_clean_data(DATA_PATH)
    df_imputed = impute_missing_values(df, CONTINUOUS_VARS, BINARY_VARS)

    df_scores, score_columns = create_scores(df_imputed, SCORING_RULES)

    df_grouped, group_columns = create_score_groups(df_scores, score_columns)

    score_save_path = os.path.join(OUTPUT_DIR, 'scored_data.csv')
    df_grouped.to_csv(score_save_path, index=False)

    all_results = analyze_all_scores(
        df_grouped, score_columns, group_columns, 
        OUTCOMES, COVARIATES
    )

    save_results(all_results, OUTPUT_DIR)