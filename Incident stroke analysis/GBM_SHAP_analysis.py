"""
Classification Predictive Model: SHAP-Based Variable Importance Analysis
Model: Gradient Boosting Machine (GBM)
Method: Variable importance was computed using SHapley Additive exPlanations (SHAP) values.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import shap
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. Configuration Parameters
# ===============================

DATA_PATH = "{data_dir}/脑卒中_3839.csv"
OUTPUT_DIR = "{data_dir}/ML_Classification_3839_SHAP"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Main variables
MAIN_VARS = [
    'BMI', 'WHR', 'Height', 'Weight', 'WC', 'HC',
    'WHtR', 'HHtR', 'ABSI', 'BRI', 'BF_percent'
]

# Covariate
COVARIATES = ['age', 'sex']

BINARY_VARS = ['sex', 'Smoke', 'Alcohol', 'DM']
CONTINUOUS_VARS = [v for v in MAIN_VARS + COVARIATES if v not in BINARY_VARS]

# Outcome
OUTCOMES = {
    'subject': 'subject',
    'Hemorrhagic_Stroke': 'Hemorrhagic_Stroke',
    'Ischemic_Stroke': 'Ischemic_Stroke'
}

N_FOLDS = 10
RANDOM_SEED = 42

# ===============================
# 2. Data processing function
# ===============================

def load_and_clean_data(data_path):
    df = pd.read_csv(data_path)
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
        print(f"\n  Total deleted: {n_before - n_after} 行")
        print(f"  Remaining samples: {n_after} 行")

    continuous_to_impute = [v for v in continuous_vars if v in df_imputed.columns]
    
    if continuous_to_impute:
        imputer = SimpleImputer(strategy='median')
        df_imputed[continuous_to_impute] = imputer.fit_transform(
            df_imputed[continuous_to_impute]
        )
    
    return df_imputed

def prepare_classification_data(df, outcome_name, outcome_col, 
                                main_vars, covariates):
    
    required_cols = [outcome_col] + main_vars + covariates
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️ The following columns are missing: {missing_cols}")
        return None, None, None
    
    cols_to_use = [outcome_col] + main_vars + covariates
    df_model = df[cols_to_use].copy()
    
    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    
    df_model = df_model.dropna(subset=[outcome_col])
    
    if len(df_model) < 100 or df_model[outcome_col].sum() < 20:
        print(f"⚠️ Sample size or number of cases is insufficient.")
        return None, None, None
    
    feature_names = main_vars + covariates
    X = df_model[feature_names].copy()
    y = df_model[outcome_col].copy()
    
    # 标准化
    scaler = StandardScaler()
    continuous_vars_in_features = [v for v in feature_names if v not in BINARY_VARS]
    
    if continuous_vars_in_features:
        X[continuous_vars_in_features] = scaler.fit_transform(
            X[continuous_vars_in_features]
        )
    
    return X, y, feature_names

# ===============================
# 3. Compute SHAP importance (10-fold cross-validation)
# ===============================

def calculate_shap_importance_cv_gbm(
    X, y, feature_names, main_vars,
    n_folds=10, random_state=42
):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    shap_by_fold = {var: [] for var in main_vars}
    auc_by_fold = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")

        X_train = X.iloc[train_idx].copy()
        X_val   = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx]
        y_val   = y.iloc[val_idx]

        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        pred_val = model.predict_proba(X_val)[:, 1]
        baseline_auc = roc_auc_score(y_val, pred_val)
        auc_by_fold.append(baseline_auc)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            for var in main_vars:
                var_idx = feature_names.index(var)
                mean_abs_shap = np.mean(np.abs(shap_values[:, var_idx]))
                shap_by_fold[var].append(mean_abs_shap)
                
        except Exception as e:
            print(f"  ⚠️ SHAP calculation error: {e}")
            for var in main_vars:
                shap_by_fold[var].append(np.nan)

    # 汇总结果
    results = []
    for var in main_vars:
        fold_values = [v for v in shap_by_fold[var] if not np.isnan(v)]
        if len(fold_values) > 0:
            results.append({
                'variable': var,
                'shap_importance': np.mean(fold_values),
                'shap_importance_std': np.std(fold_values)
            })

    importance_df = pd.DataFrame(results)
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)

    print(importance_df[['rank', 'variable', 'shap_importance', 'shap_importance_std']].to_string(index=False))

    return importance_df

# ===============================
# 4. Visualization
# ===============================

def plot_shap_importance(importance_df, outcome_name, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))

    plot_df = importance_df.sort_values('shap_importance', ascending=True)

    ax.barh(
        plot_df['variable'],
        plot_df['shap_importance'],
        xerr=plot_df.get('shap_importance_std', 0),
        color='coral',
        alpha=0.8,
        capsize=5
    )

    ax.set_xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=12, fontweight='bold')
    ax.set_title(
        f'{outcome_name} - {model_name}\nSHAP Importance (10-Fold CV)',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        f'{outcome_name}_{model_name}_shap_importance.pdf'
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ===============================
# 5. Main process
# ===============================

def run_analysis(outcome_name, outcome_col):
    outcome_dir = os.path.join(OUTPUT_DIR, outcome_name)
    os.makedirs(outcome_dir, exist_ok=True)

    X, y, feature_names = prepare_classification_data(
        df_imputed, outcome_name, outcome_col, 
        MAIN_VARS, COVARIATES
    )
    
    if X is None:
        return None
    all_results = {}
    # ===== 1. GBM =====
    
    gbm_imp = calculate_shap_importance_cv_gbm(
        X, y, feature_names, MAIN_VARS,
        n_folds=N_FOLDS,
        random_state=RANDOM_SEED
    )
    
    save_path = os.path.join(outcome_dir, f'{outcome_name}_GBM_shap_importance.csv')
    gbm_imp.to_csv(save_path, index=False)
    plot_shap_importance(gbm_imp, outcome_name, "GBM", outcome_dir)
    all_results['GBM'] = gbm_imp
    
    # ===== 2. Logistic Regression =====
    
    logistic_imp = calculate_shap_importance_cv_logistic(
        X, y, feature_names, MAIN_VARS,
        n_folds=N_FOLDS,
        random_state=RANDOM_SEED
    )
    
    save_path = os.path.join(outcome_dir, f'{outcome_name}_Logistic_shap_importance.csv')
    logistic_imp.to_csv(save_path, index=False)
    plot_shap_importance(logistic_imp, outcome_name, "Logistic", outcome_dir)
    all_results['Logistic'] = logistic_imp
    
    # ===== 3. LASSO Logistic =====
    lasso_imp = calculate_shap_importance_cv_lasso_logistic(
        X, y, feature_names, MAIN_VARS,
        base_vars=["age", "sex"],
        n_folds=N_FOLDS,
        random_state=RANDOM_SEED
    )
    
    save_path = os.path.join(outcome_dir, f"{outcome_name}_LASSO_Logistic_shap_importance.csv")
    lasso_imp.to_csv(save_path, index=False)
    
    plot_shap_importance(lasso_imp, outcome_name, "LASSO_Logistic", outcome_dir)
    all_results['LASSO_Logistic'] = lasso_imp
    
    return all_results

# ===============================
# 6. Execute
# ===============================

if __name__ == "__main__":
    df = load_and_clean_data(DATA_PATH)
    df_imputed = impute_missing_values(df, CONTINUOUS_VARS, BINARY_VARS)
    
    all_results = {}
    
    for outcome_name, outcome_col in OUTCOMES.items():
        try:
            results = run_analysis(outcome_name, outcome_col)
            
            if results is not None:
                all_results[outcome_name] = results
                
        except Exception as e:
            print(f"\n❌ {outcome_name} error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("All completed!")
    print(f"{'='*80}")
    # 打印总结
    for outcome_name, results_dict in all_results.items():
        print(f"\n{outcome_name}:")
        for model_name in results_dict.keys():
            print(f"  - {model_name}: ✅")