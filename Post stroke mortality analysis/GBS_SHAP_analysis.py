"""
Survival Analysis Machine Learning - Variable Importance Analysis (SHAP Version)Model: GBSVariable importance calculated using SHAP values
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import shap
import warnings
warnings.filterwarnings('ignore')

# ===============================
# Load data
# ===============================

DATA_PATH = "/2467_UKB.csv"
OUTPUT_DIR = "/ML_Survival_2467_SHAP"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Main vars
MAIN_VARS = [
    'BMI', 'WHR', 'Height', 'Weight', 'WC', 'HC',
    'WHtR', 'HHtR', 'ABSI', 'BRI', 'BF_percent'
]

# covariates
COVARIATES = ['Age', 'Sex']

BINARY_VARS = ['Sex', 'Smoke', 'Alcohol', 'Diabhis']
CONTINUOUS_VARS = [v for v in MAIN_VARS + COVARIATES if v not in BINARY_VARS]

OUTCOMES = {
    'death': {'time': 'death_time', 'event': 'death'},
    'cvddeath': {'time': 'cvddeath_time', 'event': 'cvddeath'},
    'cvdevent': {'time': 'cvdevent_time', 'event': 'cvdevent'}
}

# GBS Model Parameter Configuration
GBS_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 3,
    'subsample': 0.8,
    'random_state': 42
}

N_REPEATS = 10
RANDOM_SEED = 42

# ===============================
# Data processing function
# ===============================

def load_and_clean_data(data_path):
    df = pd.read_csv(data_path)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    df = df.applymap(
    lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
    )

    drop_cols = [
        'data完整', 'dataset完整临床',
        'subject', 'subtype', '发病日期','thrombosis', 'lacunar', 'ICH'
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    date_cols = ['入组时间', 'death_date', '终止时间',"cvdevent_data","cvddeath_data"]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
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
        
    return df_clean
    

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

def prepare_survival_data(df, outcome_name, outcome_config, 
                         main_vars, covariates):

    time_col = outcome_config['time']
    event_col = outcome_config['event']
    
    required_cols = [time_col, event_col] + main_vars + covariates
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️ The following columns are missing: {missing_cols}")
        return None, None, None, None, None
    
    cols_to_use = [time_col, event_col] + main_vars + covariates
    df_model = df[cols_to_use].copy()
    
    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    
    df_model = df_model.dropna(subset=[time_col, event_col])
    
    if (df_model[time_col] <= 0).any():
        n_invalid = (df_model[time_col] <= 0).sum()
        df_model = df_model[df_model[time_col] > 0]

    if len(df_model) < 100 or df_model[event_col].sum() < 20:
        print(f"⚠️ Insufficient sample size or number of events")
        return None, None, None, None, None
    
    feature_names = main_vars + covariates
    X = df_model[feature_names].copy()
    
    scaler = StandardScaler()
    continuous_vars_in_features = [v for v in feature_names if v not in BINARY_VARS]
    
    if continuous_vars_in_features:
        X[continuous_vars_in_features] = scaler.fit_transform(
            X[continuous_vars_in_features]
        )
    
    y_surv = Surv.from_dataframe(event=event_col, time=time_col, data=df_model)
    y_binary = df_model[event_col].values
    
    return X, y_surv, y_binary, feature_names, event_col

# ===============================
# 3. Calculate variable importance using SHAP values
# ===============================

def calculate_shap_importance_cv_gbs(
    X, y, feature_names, main_vars, event_col, time_col,
    n_folds=10, random_state=42
):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    shap_by_fold = {var: [] for var in main_vars}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")

        X_train = X.iloc[train_idx].copy()
        X_val   = X.iloc[val_idx].copy()
        y_train = y[train_idx]

        model = GradientBoostingSurvivalAnalysis(**GBS_PARAMS)
        model.fit(X_train, y_train)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
            for var in main_vars:
                var_idx = feature_names.index(var)
                mean_abs_shap = np.mean(np.abs(shap_values[:, var_idx]))
                shap_by_fold[var].append(mean_abs_shap)
                
        except Exception as e:
            n_samples = min(100, len(X_train))
            X_summary = shap.sample(X_train, n_samples, random_state=random_state)
            
            def model_predict(data):
                return model.predict(pd.DataFrame(data, columns=X_train.columns))
            
            explainer = shap.KernelExplainer(model_predict, X_summary)
            shap_values = explainer.shap_values(X_val)
            
            for var in main_vars:
                var_idx = feature_names.index(var)
                mean_abs_shap = np.mean(np.abs(shap_values[:, var_idx]))
                shap_by_fold[var].append(mean_abs_shap)

    results = []
    for var in main_vars:
        fold_values = shap_by_fold[var]
        results.append({
            'variable': var,
            'shap_importance': np.mean(fold_values),
            'shap_importance_std': np.std(fold_values)
        })

    importance_df = pd.DataFrame(results)
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    return importance_df


def calculate_shap_importance_cv_logistic(
    X, y_binary, feature_names, main_vars,
    n_folds=10, random_state=42
):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    shap_by_fold = {var: [] for var in main_vars}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")

        X_train = X.iloc[train_idx].copy()
        X_val   = X.iloc[val_idx].copy()
        y_train = y_binary[train_idx]

        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        try:
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_val)
            
            # 计算每个主要变量的平均绝对SHAP值
            for var in main_vars:
                var_idx = feature_names.index(var)
                mean_abs_shap = np.mean(np.abs(shap_values[:, var_idx]))
                shap_by_fold[var].append(mean_abs_shap)
                
        except Exception as e:
            print(f"  ⚠️ Error in SHAP calculation: {e}")
            continue
    results = []
    for var in main_vars:
        vals = shap_by_fold[var]
        if len(vals) > 0:
            results.append({
                "variable": var,
                "shap_importance": np.mean(vals),
                "shap_importance_std": np.std(vals)
            })

    df_imp = pd.DataFrame(results).sort_values("shap_importance", ascending=False)
    df_imp["rank"] = range(1, len(df_imp) + 1)
    return df_imp


def calculate_shap_importance_cv_lasso_logistic(
    X, y_binary, feature_names, main_vars,
    base_vars=["age", "sex"],
    n_folds=10,
    random_state=42
):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    beta_by_fold = {v: [] for v in main_vars}
    shap_by_fold = {v: [] for v in main_vars}
    auc_by_fold = []

    if isinstance(y_binary, pd.Series):
        y_binary = y_binary.values
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y_binary[train_idx]
        y_val = y_binary[val_idx]
        model = LogisticRegressionCV(
            Cs=20,
            cv=5,
            penalty="l1",
            solver="saga",
            scoring="roc_auc",
            max_iter=5000,
            n_jobs=-1,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)

        coef = model.coef_.ravel()
        for var in main_vars:
            var_idx = feature_names.index(var)
            beta_by_fold[var].append(coef[var_idx])

        y_pred_baseline = model.predict_proba(X_val)[:, 1]
        baseline_auc = roc_auc_score(y_val, y_pred_baseline)
        auc_by_fold.append(baseline_auc)
        try:
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_val)
            
            for var in main_vars:
                var_idx = feature_names.index(var)
                mean_abs_shap = np.mean(np.abs(shap_values[:, var_idx]))
                shap_by_fold[var].append(mean_abs_shap)
                
        except Exception as e:
            print(f"  ⚠️ Error in SHAP calculation: {e}")
            for var in main_vars:
                shap_by_fold[var].append(np.nan)
    results = []
    for var in main_vars:
        betas = np.array(beta_by_fold[var])
        shap_vals = np.array([s for s in shap_by_fold[var] if not np.isnan(s)])
        
        results.append({
            "variable": var,
            "beta_mean": betas.mean(),
            "beta_std": betas.std(),
            "beta_abs_mean": np.abs(betas).mean(),
            "selection_freq": np.mean(betas != 0),
            "shap_importance": shap_vals.mean() if len(shap_vals) > 0 else 0,
            "shap_importance_std": shap_vals.std() if len(shap_vals) > 0 else 0
        })
    
    importance_df = pd.DataFrame(results)

    importance_df = importance_df.sort_values("shap_importance", ascending=False)
    importance_df["rank"] = range(1, len(importance_df) + 1)
    
    return importance_df
# ===============================
# Main process
# ===============================

def run_analysis(outcome_name, outcome_config):
    outcome_dir = os.path.join(OUTPUT_DIR, outcome_name)
    os.makedirs(outcome_dir, exist_ok=True)
    
    time_col = outcome_config['time']
    event_col = outcome_config['event']
    result = prepare_survival_data(
        df_imputed, outcome_name, outcome_config, 
        MAIN_VARS, COVARIATES
    )
    
    if result[0] is None:
        return None
    
    X, y_surv, y_binary, feature_names, event_col = result

    
    all_results = {}

    gbs_imp = calculate_shap_importance_cv_gbs(
        X, y_surv, feature_names, MAIN_VARS,
        event_col, time_col,
        n_folds=10,
        random_state=RANDOM_SEED
    )
    
    save_path = os.path.join(outcome_dir, f'{outcome_name}_GBS_shap_importance.csv')
    gbs_imp.to_csv(save_path, index=False)
    all_results['GBS'] = gbs_imp
    return all_results

if __name__ == "__main__":

    df = load_and_clean_data(DATA_PATH)
    df_imputed = impute_missing_values(df, CONTINUOUS_VARS, BINARY_VARS)

    all_results = {}
    
    for outcome_name, outcome_config in OUTCOMES.items():
        try:
            results = run_analysis(outcome_name, outcome_config)
            
            if results is not None:
                all_results[outcome_name] = results
                
        except Exception as e:
            print(f"\n❌ {outcome_name} error: {e}")
            import traceback
            traceback.print_exc()

    for outcome_name, results_dict in all_results.items():
        print(f"\n{outcome_name}:")
        for model_name in results_dict.keys():
            print(f"  - {model_name}: ✅")