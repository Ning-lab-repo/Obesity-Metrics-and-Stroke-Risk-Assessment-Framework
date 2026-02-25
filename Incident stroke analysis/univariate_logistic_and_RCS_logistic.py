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
from patsy import dmatrix
import scipy.stats as stats
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
# ===============================
# 1. Read data
# ===============================
var_file = "/3839.csv"
var_data = pd.read_csv(var_file)

# 2. Uniformly convert empty strings to NaN
df = var_data.applymap(
    lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
)

# ===============================
# 3. Define variables
# ===============================
main_vars = [
    'BMI', 'WHR', 'Height', 'Weight', 'WC', 'HC',
    'WHtR', 'HHtR', 'ABSI', 'BRI', 'BF_percent'
]

covariates = ['age', 'sex']
outcomes = ["subject",'Ischemic_Stroke', 'Hemorrhagic_Stroke']

# Ensure that the outcome variable is numerical
for col in outcomes:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ===============================
# 4. Univariate Logistic Regression Function
# ===============================
def run_univariate_logistic(data, var, outcome, adjust_vars):
    """
    单变量Logistic回归（调整协变量）
    """
    cols = [outcome, var] + adjust_vars
    
    # 数值化并删除缺失
    df_model = data[cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    if df_model.shape[0] < 50:
        return None
    
    if df_model[outcome].nunique() < 2:
        return None
    scaler = StandardScaler()
    fei_cvar = ["sex","Smoke","Alcohol","Summed_minutes_activity","HTNhis","CHDhis","RENhis","hyperlipid","DM","hypertension","spirin","antihyper"]
    ac_vars = [v for v in adjust_vars if v not in fei_cvar]
    b_cols = [var]+ac_vars
    df_model[b_cols] = scaler.fit_transform(df_model[b_cols])
    try:
        # 准备数据
        X = df_model[[var] + adjust_vars]
        X = add_constant(X)
        y = df_model[outcome]
        
        # 拟合模型
        model = Logit(y, X).fit(disp=False, maxiter=300)
        
        # 提取主变量的结果
        coef = model.params[var]
        se = model.bse[var]
        pval = model.pvalues[var]
        
        or_value = np.exp(coef)
        or_lci = np.exp(coef - 1.96 * se)
        or_uci = np.exp(coef + 1.96 * se)
        
        return {
            'variable': var,
            'outcome': outcome,
            'n': df_model.shape[0],
            'events': int(y.sum()),
            'OR': or_value,
            'OR_LCI': or_lci,
            'OR_UCI': or_uci,
            'p_value': pval,
            'coef': coef,
            'se': se
        }
    except Exception as e:
        print(f"  ✗ {var} - {outcome} LR失败: {e}")
        return None

# ===============================
# 5. RCS analysis function
# ===============================
def run_logistic_rcs(data, var, outcome, adjust_vars, knots=4):
    """
    Logistic RCS分析
    """
    cols = [outcome, var] + adjust_vars
    
    # 数值化并删除缺失
    df_model = data[cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    if df_model.shape[0] < 100:
        return None
    
    if df_model[outcome].nunique() < 2:
        return None
    
    try:
        # 生成RCS基函数
        formula = f"cr({var}, df={knots-1})"
        spline_basis = dmatrix(formula, df_model, return_type="dataframe")
        
        # 准备协变量
        X = pd.concat([spline_basis, df_model[adjust_vars]], axis=1)
        X = add_constant(X)
        y = df_model[outcome]
        
        # 拟合模型
        model = Logit(y, X).fit(disp=False, maxiter=300)
        
        # 非线性检验（Wald test）
        spline_cols = [c for c in spline_basis.columns if 'Intercept' not in c]
        beta = model.params[spline_cols]
        cov = model.cov_params().loc[spline_cols, spline_cols]
        wald_chi2 = float(beta.T @ np.linalg.inv(cov) @ beta)
        p_nonlinear = 1 - stats.chi2.cdf(wald_chi2, len(beta))
        
        # 生成绘图数据
        test_range = np.linspace(
            df_model[var].quantile(0.05), 
            df_model[var].quantile(0.95), 
            100
        )
        
        # 参照点：中位数
        ref_val = df_model[var].median()
        
        # 构建预测数据
        plot_df = pd.DataFrame({var: np.append(test_range, ref_val)})
        for cov_var in adjust_vars:
            if df_model[cov_var].nunique() == 2:  # 二分类变量用众数
                plot_df[cov_var] = df_model[cov_var].mode()[0]
            else:  # 连续变量用均值
                plot_df[cov_var] = df_model[cov_var].mean()
        
        # 计算预测矩阵
        X_plot_basis = dmatrix(formula, plot_df, return_type="dataframe")
        X_plot = pd.concat([X_plot_basis, plot_df[adjust_vars]], axis=1)
        X_plot = add_constant(X_plot)
        
        # 计算Log-Odds及其标准误
        predictions = np.dot(X_plot, model.params)
        cov_mat = model.cov_params().values
        se_predictions = np.sqrt(np.diag(X_plot.values @ cov_mat @ X_plot.values.T))
        
        # 相对于参照点的OR
        ref_log_odds = predictions[-1]
        
        res_plot = pd.DataFrame({
            "value": test_range,
            "OR": np.exp(predictions[:-1] - ref_log_odds),
            "OR_lower": np.exp((predictions[:-1] - ref_log_odds) - 1.96 * se_predictions[:-1]),
            "OR_upper": np.exp((predictions[:-1] - ref_log_odds) + 1.96 * se_predictions[:-1]),
            "log_OR": predictions[:-1] - ref_log_odds,
            "se_log_OR": se_predictions[:-1]
        })
        
        return {
            "summary": {
                "variable": var,
                "outcome": outcome,
                "n": df_model.shape[0],
                "events": int(y.sum()),
                "reference_value": ref_val,
                "p_nonlinear": p_nonlinear,
                "wald_chi2": wald_chi2,
                "df": len(beta)
            },
            "plot_data": res_plot
        }
    except Exception as e:
        print(f"  ✗ {var} - {outcome} RCS失败: {e}")
        return None

# ===============================
# 6. Main analysis workflow
# ===============================

lr_results = []

for outcome in outcomes:
    print(f"\n分析结局: {outcome}")
    print(f"  事件数: {df[outcome].sum()}")
    
    for var in main_vars:
        if var not in df.columns:
            print(f"  ⚠️ 变量不存在: {var}")
            continue
        
        result = run_univariate_logistic(df, var, outcome, covariates)
        
        if result:
            lr_results.append(result)
            print(f"  ✓ {var}: OR={result['OR']:.3f} ({result['OR_LCI']:.3f}-{result['OR_UCI']:.3f}), p={result['p_value']:.4f}")

if lr_results:
    lr_df = pd.DataFrame(lr_results)
    
    # FDR校正（按结局分组）
    for outcome in outcomes:
        mask = lr_df['outcome'] == outcome
        if mask.sum() > 0:
            _, padj = fdrcorrection(lr_df.loc[mask, 'p_value'].values)
            lr_df.loc[mask, 'p_adj'] = padj
    
    # 保存
    lr_output = "/home/data/wangshikai/脑卒中/LR/Univariate_LR_Results.csv"
    os.makedirs(os.path.dirname(lr_output), exist_ok=True)
    lr_df.to_csv(lr_output, index=False)
    print(f"\n✅ 单变量LR结果已保存: {lr_output}")
    print(f"   共{len(lr_results)}个有效分析")

# ===============================
# 7. RCS analysis
# ===============================

rcs_summaries = []
output_dir = "/LR/RCS"
os.makedirs(output_dir, exist_ok=True)

for outcome in outcomes:
    print(f"\n分析结局: {outcome}")
    
    for var in main_vars:
        if var not in df.columns:
            continue
        
        result = run_logistic_rcs(df, var, outcome, covariates, knots=4)
        
        if result:
            rcs_summaries.append(result["summary"])
            
            # 保存曲线数据
            safe_var_name = var.replace("/", "_per_")
            plot_file = f"{output_dir}/RCS_{outcome}_{safe_var_name}.csv"
            result["plot_data"].to_csv(plot_file, index=False)
            
            print(f"  ✓ {var}: p_nonlinear={result['summary']['p_nonlinear']:.4f}, ref={result['summary']['reference_value']:.2f}")

if rcs_summaries:
    rcs_df = pd.DataFrame(rcs_summaries)
    
    # FDR校正
    for outcome in outcomes:
        mask = rcs_df['outcome'] == outcome
        if mask.sum() > 0:
            _, padj = fdrcorrection(rcs_df.loc[mask, 'p_nonlinear'].values)
            rcs_df.loc[mask, 'p_adj'] = padj
    
    # 保存
    rcs_summary_file = "/home/data/wangshikai/脑卒中/LR/RCS_Summary_Table.csv"
    rcs_df.to_csv(rcs_summary_file, index=False)
    print(f"\n✅ RCS汇总结果已保存: {rcs_summary_file}")
    print(f"   共{len(rcs_summaries)}个有效分析")
    print(f"📈 RCS曲线数据保存目录: {output_dir}")

# ===============================
# 8. Generate a comprehensive report
# ===============================


if lr_results and rcs_summaries:
    for outcome in outcomes:
        lr_count = sum(1 for r in lr_results if r['outcome'] == outcome)
        rcs_count = sum(1 for r in rcs_summaries if r['outcome'] == outcome)
        
        print(f"\n{outcome}:")
        print(f"  - 单变量LR: {lr_count}个变量")
        print(f"  - RCS分析: {rcs_count}个变量")
        
        # 显著结果统计
        if lr_count > 0:
            sig_lr = lr_df[(lr_df['outcome'] == outcome) & (lr_df['p_value'] < 0.05)]
            print(f"  - LR显著变量 (p<0.05): {len(sig_lr)}个")
            if len(sig_lr) > 0:
                print(f"    {', '.join(sig_lr['variable'].tolist())}")
        
        if rcs_count > 0:
            sig_rcs = rcs_df[(rcs_df['outcome'] == outcome) & (rcs_df['p_nonlinear'] < 0.05)]
            print(f"  - RCS非线性显著 (p<0.05): {len(sig_rcs)}个")
            if len(sig_rcs) > 0:
                print(f"    {', '.join(sig_rcs['variable'].tolist())}")

print("\n" + "=" * 80)
print("All analyses completed！")
print("=" * 80)