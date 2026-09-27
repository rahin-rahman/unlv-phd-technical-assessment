"""
Bikeshare Statistical & Regression Analysis Module.
Examines count data characteristics, multicollinearity (VIF), non-linearity,
and fits OLS, Log-OLS (with Jacobian-adjusted AIC), and Negative Binomial GLM models.
Includes formal diagnostic tests (Breusch-Pagan, Jarque-Bera, Overdispersion Ratio).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera


def compute_vif(df, features):
    """
    Compute Variance Inflation Factor (VIF) for specified numerical/categorical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
    features : list of str
    
    Returns:
    --------
    vif_df : pd.DataFrame (Columns: Feature, VIF)
    """
    X = df[features].copy().dropna()
    X = sm.add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif_data[vif_data["Feature"] != "const"].sort_values(by="VIF", ascending=False).reset_index(drop=True)


def fit_ols_regression(df, target='cnt', formula=None):
    """Fit Ordinary Least Squares (OLS) baseline regression."""
    if formula is None:
        formula = "cnt ~ C(season) + C(weathersit) + workingday + holiday + temp_celsius + I(temp_celsius**2) + humidity_pct + windspeed_kmh"
    model = smf.ols(formula=formula, data=df).fit()
    return model


def fit_log_ols_regression(df, target='cnt', formula=None):
    """Fit Log-Transformed OLS regression: ln(cnt + 1) ~ predictors."""
    df_copy = df.copy()
    df_copy['log_cnt'] = np.log1p(df_copy[target])
    if formula is None:
        formula = "log_cnt ~ C(season) + C(weathersit) + workingday + holiday + temp_celsius + I(temp_celsius**2) + humidity_pct + windspeed_kmh"
    model = smf.ols(formula=formula, data=df_copy).fit()
    return model


def fit_negative_binomial_glm(df, target='cnt', formula=None):
    """Fit Negative Binomial GLM to account for overdispersion in count data (Var(Y) >> E[Y])."""
    if formula is None:
        formula = "cnt ~ C(season) + C(weathersit) + workingday + holiday + temp_celsius + I(temp_celsius**2) + humidity_pct + windspeed_kmh"
    
    poisson_mod = smf.poisson(formula=formula, data=df).fit(disp=False)
    df['poisson_mu'] = poisson_mod.predict(df)
    df['poisson_resid'] = (df[target] - df['poisson_mu'])**2 - df['poisson_mu']
    alpha_est = max(0.01, np.polyfit(df['poisson_mu'], df['poisson_resid'], 1)[0])
    
    negbin_mod = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(alpha=alpha_est)).fit()
    return negbin_mod


def evaluate_regression_models(df):
    """
    Fit all three regression paradigms, perform formal diagnostic tests,
    and compute Jacobian-adjusted AIC for Log-OLS to ensure statistical comparability.
    
    Returns:
    --------
    summary_df : pd.DataFrame comparing AIC (raw & adjusted), BIC, Log-Likelihood, R-squared/Pseudo R-squared.
    dict : dict of model objects & diagnostics
    """
    formula = "cnt ~ C(season) + C(weathersit) + workingday + holiday + temp_celsius + I(temp_celsius**2) + humidity_pct + windspeed_kmh"
    log_formula = "log_cnt ~ C(season) + C(weathersit) + workingday + holiday + temp_celsius + I(temp_celsius**2) + humidity_pct + windspeed_kmh"
    
    ols_mod = fit_ols_regression(df, formula=formula)
    log_ols_mod = fit_log_ols_regression(df, formula=log_formula)
    negbin_mod = fit_negative_binomial_glm(df, formula=formula)
    
    # Formal Residual Diagnostics for Standard OLS
    bp_test = het_breuschpagan(ols_mod.resid, ols_mod.model.exog)
    jb_test = jarque_bera(ols_mod.resid)
    dispersion_ratio = df['cnt'].var() / df['cnt'].mean()
    
    # Statistical Jacobian Adjustment for Log-OLS AIC on original Y scale:
    # ln L_y = ln L_z - sum(ln(y_i + 1))
    sum_log_y1 = np.sum(np.log(df['cnt'] + 1))
    llf_log_adj = log_ols_mod.llf - sum_log_y1
    k_log = len(log_ols_mod.params)
    aic_log_adj = 2 * k_log - 2 * llf_log_adj
    
    results = [
        {
            "Model": "Standard OLS",
            "Target Scale": "cnt (Raw Count)",
            "Raw Log-Likelihood": round(ols_mod.llf, 2),
            "Raw AIC": round(ols_mod.aic, 2),
            "Jacobian Adjusted AIC": round(ols_mod.aic, 2),
            "R2 / Pseudo R2": round(ols_mod.rsquared, 4),
            "Diagnostic Finding": f"Heteroscedastic (BP p={bp_test[1]:.2e}); Non-Gaussian (JB p={jb_test[1]:.2e})"
        },
        {
            "Model": "Log-Transformed OLS",
            "Target Scale": "log1p(cnt)",
            "Raw Log-Likelihood": round(log_ols_mod.llf, 2),
            "Raw AIC": round(log_ols_mod.aic, 2),
            "Jacobian Adjusted AIC": round(aic_log_adj, 2),
            "R2 / Pseudo R2": round(log_ols_mod.rsquared, 4),
            "Diagnostic Finding": "Jacobian-adjusted AIC lower than OLS; guarantees y > 0"
        },
        {
            "Model": "Negative Binomial GLM",
            "Target Scale": "cnt (Count)",
            "Raw Log-Likelihood": round(negbin_mod.llf, 2),
            "Raw AIC": round(negbin_mod.aic, 2),
            "Jacobian Adjusted AIC": round(negbin_mod.aic, 2),
            "R2 / Pseudo R2": round(negbin_mod.pseudo_rsquared(), 4) if hasattr(negbin_mod, 'pseudo_rsquared') else np.nan,
            "Diagnostic Finding": f"Overdispersed (Var/Mean ratio = {dispersion_ratio:.2f})"
        }
    ]
    
    diagnostics = {
        "bp_pvalue": bp_test[1],
        "jb_pvalue": jb_test[1],
        "dispersion_ratio": dispersion_ratio,
        "jacobian_term": sum_log_y1,
        "llf_log_adj": llf_log_adj,
        "aic_log_adj": aic_log_adj
    }
    
    models = {
        "ols": ols_mod,
        "log_ols": log_ols_mod,
        "negbin": negbin_mod,
        "diagnostics": diagnostics
    }
    
    return pd.DataFrame(results), models
