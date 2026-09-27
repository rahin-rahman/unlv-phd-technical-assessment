"""
Bikeshare Demand Forecasting Module.
Implements chronological month-based partitioning (Days 1-20 Train, Days 21+ Test),
dual feature engineering paradigms (Pure Exogenous vs Day-Ahead Rolling Lags),
baseline & GBDT modeling, and evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


def build_forecasting_features(df):
    """
    Construct features for time-series forecasting.
    
    Features built:
    - Calendar & temporal features (hour, month, day, dayofweek, year, workingday, holiday, season)
    - Cyclical transformations (sin/cos for hour, month, dayofweek)
    - Weather features (temp_celsius, atemp_celsius, humidity_pct, windspeed_kmh, weathersit)
    - Shifted lag features (24h, 48h, 168h) for day-ahead rolling forecast protocol
    """
    df_feat = df.copy().sort_values('datetime').reset_index(drop=True)
    
    # 1. Cyclical Encodings
    df_feat['hr_sin'] = np.sin(2 * np.pi * df_feat['hr'] / 24.0)
    df_feat['hr_cos'] = np.cos(2 * np.pi * df_feat['hr'] / 24.0)
    df_feat['mnth_sin'] = np.sin(2 * np.pi * df_feat['mnth'] / 12.0)
    df_feat['mnth_cos'] = np.cos(2 * np.pi * df_feat['mnth'] / 12.0)
    df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['dayofweek'] / 7.0)
    df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['dayofweek'] / 7.0)
    
    # 2. Temperature Difference
    df_feat['temp_diff'] = df_feat['atemp_celsius'] - df_feat['temp_celsius']
    
    # 3. Shifted Lag Features (24h = same hour previous day, 168h = same hour previous week)
    df_feat['cnt_lag_24'] = df_feat['cnt'].shift(24)
    df_feat['cnt_lag_48'] = df_feat['cnt'].shift(48)
    df_feat['cnt_lag_168'] = df_feat['cnt'].shift(168)
    
    # Rolling statistics (shifted by 24h)
    df_feat['cnt_roll_mean_24'] = df_feat['cnt'].shift(24).rolling(window=24, min_periods=1).mean()
    df_feat['cnt_roll_std_24'] = df_feat['cnt'].shift(24).rolling(window=24, min_periods=1).std().fillna(0)
    
    return df_feat


def partition_by_day_of_month(df):
    """
    Partition dataset strictly according to assessment requirements:
    - Train set: First 20 days of each month (days 1 through 20)
    - Test set: From the 21st through the end of each month
    """
    train_mask = df['day'] <= 20
    test_mask = df['day'] >= 21
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    return train_df, test_df


def compute_metrics(y_true, y_pred):
    """Calculate evaluation metrics: MAE, RMSE, RMSLE, R2."""
    y_pred_clipped = np.clip(y_pred, 0, None)
    
    mae = mean_absolute_error(y_true, y_pred_clipped)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_clipped)))
    r2 = r2_score(y_true, y_pred_clipped)
    
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "RMSLE": round(rmsle, 4),
        "R2": round(r2, 4)
    }


class HistoricalAverageBaseline:
    """Historical Hourly Average Baseline grouped by (hr, workingday, season)."""
    def __init__(self):
        self.lookup_ = None
        self.global_mean_ = 0
        
    def fit(self, X_train, y_train):
        train_df = X_train.copy()
        train_df['target'] = y_train
        self.global_mean_ = y_train.mean()
        self.lookup_ = train_df.groupby(['hr', 'workingday', 'season'])['target'].mean().reset_index()
        
    def predict(self, X_test):
        merged = pd.merge(X_test[['hr', 'workingday', 'season']], self.lookup_, on=['hr', 'workingday', 'season'], how='left')
        preds = merged['target'].fillna(self.global_mean_).values
        return preds


def train_and_evaluate_forecasting(df):
    """
    Complete Forecasting Pipeline evaluating two leakage-proof feature protocols:
    Protocol A: Pure Exogenous Features (Zero Target Lags - 10-day multi-step cutoff)
    Protocol B: Day-Ahead Rolling Forecast (24-hour shifted target lags)
    """
    df_feat = build_forecasting_features(df)
    df_clean = df_feat.dropna(subset=['cnt_lag_168']).copy()
    
    train_df, test_df = partition_by_day_of_month(df_clean)
    
    pure_feature_cols = [
        'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit',
        'temp_celsius', 'humidity_pct', 'windspeed_kmh',
        'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos', 'dow_sin', 'dow_cos', 'temp_diff'
    ]
    
    lag_feature_cols = pure_feature_cols + [
        'cnt_lag_24', 'cnt_lag_48', 'cnt_lag_168', 'cnt_roll_mean_24', 'cnt_roll_std_24'
    ]
    
    y_train = train_df['cnt']
    y_train_log = np.log1p(y_train)
    y_test = test_df['cnt']
    
    # 1. Baseline Model
    baseline = HistoricalAverageBaseline()
    baseline.fit(train_df, y_train)
    y_pred_base = baseline.predict(test_df)
    metrics_base = compute_metrics(y_test, y_pred_base)
    metrics_base["Model"] = "Historical Hourly Average Baseline"
    metrics_base["Protocol"] = "Zero Lags (Lookup)"

    # 2. LightGBM (Pure Exogenous Features - Zero Target Lags)
    lgb_pure = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
    lgb_pure.fit(train_df[pure_feature_cols], y_train_log)
    y_pred_lgb_pure = np.expm1(lgb_pure.predict(test_df[pure_feature_cols]))
    metrics_lgb_pure = compute_metrics(y_test, y_pred_lgb_pure)
    metrics_lgb_pure["Model"] = "LightGBM (Pure Calendar & Weather)"
    metrics_lgb_pure["Protocol"] = "Zero Target Lags (Multi-Step Horizon)"

    # 3. LightGBM (Day-Ahead Rolling 24h Lags)
    lgb_lag = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
    lgb_lag.fit(train_df[lag_feature_cols], y_train_log)
    y_pred_lgb_lag = np.expm1(lgb_lag.predict(test_df[lag_feature_cols]))
    metrics_lgb_lag = compute_metrics(y_test, y_pred_lgb_lag)
    metrics_lgb_lag["Model"] = "LightGBM (Day-Ahead Rolling Lags)"
    metrics_lgb_lag["Protocol"] = "24h Shifted Lags (Rolling Update)"

    # 4. XGBoost Regressor (Pure Exogenous Features)
    xgb_pure = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    xgb_pure.fit(train_df[pure_feature_cols], y_train_log)
    y_pred_xgb_pure = np.expm1(xgb_pure.predict(test_df[pure_feature_cols]))
    metrics_xgb_pure = compute_metrics(y_test, y_pred_xgb_pure)
    metrics_xgb_pure["Model"] = "XGBoost (Pure Calendar & Weather)"
    metrics_xgb_pure["Protocol"] = "Zero Target Lags (Multi-Step Horizon)"

    # 5. XGBoost Regressor (Day-Ahead Rolling 24h Lags)
    xgb_lag = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    xgb_lag.fit(train_df[lag_feature_cols], y_train_log)
    y_pred_xgb_lag = np.expm1(xgb_lag.predict(test_df[lag_feature_cols]))
    metrics_xgb_lag = compute_metrics(y_test, y_pred_xgb_lag)
    metrics_xgb_lag["Model"] = "XGBoost (Day-Ahead Rolling Lags)"
    metrics_xgb_lag["Protocol"] = "24h Shifted Lags (Rolling Update)"

    # Compile Overall Comparison
    overall_results = pd.DataFrame([metrics_base, metrics_lgb_pure, metrics_xgb_pure, metrics_lgb_lag, metrics_xgb_lag])
    cols_order = ["Model", "Protocol", "RMSLE", "MAE", "RMSE", "R2"]
    overall_results = overall_results[cols_order].sort_values(by="RMSLE").reset_index(drop=True)
    
    # Monthly Breakdown for LightGBM Pure Feature Model
    test_df_eval = test_df.copy()
    test_df_eval['pred_lgb_pure'] = y_pred_lgb_pure
    
    monthly_records = []
    for (yr, mnth), group in test_df_eval.groupby(['yr', 'mnth']):
        yr_label = 2011 if yr == 0 else 2012
        m_metrics = compute_metrics(group['cnt'], group['pred_lgb_pure'])
        monthly_records.append({
            "Year": yr_label,
            "Month": mnth,
            "Evaluation Window": f"{yr_label}-{mnth:02d} (Days 21-End)",
            "Observed Hours": len(group),
            "RMSLE": m_metrics["RMSLE"],
            "MAE": m_metrics["MAE"],
            "RMSE": m_metrics["RMSE"],
            "R2": m_metrics["R2"]
        })
    monthly_eval_df = pd.DataFrame(monthly_records)

    # Feature Importance for LightGBM Pure Feature Model
    importance_df = pd.DataFrame({
        "Feature": pure_feature_cols,
        "Importance": lgb_pure.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    best_info = {
        "y_test": y_test.values,
        "y_pred_lgb": y_pred_lgb_pure,
        "test_df": test_df,
        "importance_df": importance_df,
        "lgb_model": lgb_pure,
        "feature_cols": pure_feature_cols
    }
    
    return overall_results, monthly_eval_df, best_info
