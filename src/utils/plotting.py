"""
Visualization Utilities Module.
Provides publication-quality styling and reusable plot functions for Part 1 Bikeshare analysis.
Fixed all Seaborn palette/hue deprecations and Matplotlib tick warnings.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def set_publication_style():
    """Configure matplotlib and seaborn parameters for publication-ready aesthetics."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False
    })

PRIMARY_COLOR = "#1f77b4"
ACCENT_CASUAL = "#ff7f0e"
ACCENT_REGISTERED = "#2ca02c"


def save_figure(fig, filename, output_dir="outputs/figures"):
    """Save figure with high resolution."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath


def plot_temporal_demand_patterns(df, output_dir="outputs/figures"):
    """
    Plot 4-panel temporal demand patterns:
    1. Hourly rental demand by User Type (Casual vs Registered)
    2. Hourly demand by Workingday vs Weekend
    3. Monthly rental demand across Seasons
    4. Day of Week demand breakdown
    """
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Hourly demand by User Type
    hourly_user = df.groupby('hr')[['casual', 'registered', 'cnt']].mean().reset_index()
    axes[0, 0].plot(hourly_user['hr'], hourly_user['registered'], marker='o', color=ACCENT_REGISTERED, label='Registered Users', linewidth=2)
    axes[0, 0].plot(hourly_user['hr'], hourly_user['casual'], marker='s', color=ACCENT_CASUAL, label='Casual Users', linewidth=2)
    axes[0, 0].set_title("A: Mean Hourly Demand by User Type", fontweight='bold')
    axes[0, 0].set_xlabel("Hour of Day (0–23)")
    axes[0, 0].set_ylabel("Average Rental Count (bikes/hr)")
    axes[0, 0].set_xticks(range(0, 24, 2))
    axes[0, 0].legend()
    
    # 2. Hourly demand by Workingday vs Non-Workingday
    hourly_work = df.groupby(['hr', 'workingday'])['cnt'].mean().unstack()
    axes[0, 1].plot(hourly_work.index, hourly_work[1], marker='o', color='#1f77b4', label='Working Day (Mon-Fri)', linewidth=2)
    axes[0, 1].plot(hourly_work.index, hourly_work[0], marker='^', color='#d62728', label='Non-Working Day (Weekend/Holiday)', linewidth=2)
    axes[0, 1].set_title("B: Diurnal Demand: Working Day vs. Non-Working Day", fontweight='bold')
    axes[0, 1].set_xlabel("Hour of Day (0–23)")
    axes[0, 1].set_ylabel("Average Rental Count (bikes/hr)")
    axes[0, 1].set_xticks(range(0, 24, 2))
    axes[0, 1].legend()

    # 3. Monthly Demand Dynamics
    monthly = df.groupby(['mnth', 'yr'])['cnt'].sum().unstack()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[1, 0].plot(months, monthly[0], marker='o', label='Year 2011', color='#3498db', linewidth=2)
    axes[1, 0].plot(months, monthly[1], marker='s', label='Year 2012', color='#2ecc71', linewidth=2)
    axes[1, 0].set_title("C: Monthly Rental Demand Trajectory (2011 vs 2012)", fontweight='bold')
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Total Monthly Rentals")
    axes[1, 0].legend()

    # 4. Day of Week Breakdown
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_df = df.groupby('weekday_label')[['casual', 'registered']].mean().reindex(day_order)
    dow_df.plot(kind='bar', stacked=True, ax=axes[1, 1], color=[ACCENT_CASUAL, ACCENT_REGISTERED], alpha=0.85)
    axes[1, 1].set_title("D: Average Daily Demand Breakdown by Day of Week", fontweight='bold')
    axes[1, 1].set_xlabel("Day of Week")
    axes[1, 1].set_ylabel("Average Hourly Rentals")
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(['Casual', 'Registered'])

    plt.tight_layout()
    filepath = save_figure(fig, "fig1_temporal_demand_patterns.png", output_dir)
    return fig, filepath


def plot_weather_impact_analysis(df, output_dir="outputs/figures"):
    """
    Plot weather and environmental demand relationships:
    1. Demand across Weather Categories (Boxplot)
    2. Temperature vs Demand (Scatter + Lowess)
    3. Humidity vs Demand
    4. Windspeed vs Demand
    """
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Weather category impact
    sns.boxplot(data=df, x='weathersit', y='cnt', hue='weathersit', ax=axes[0, 0], palette="Blues_r", legend=False)
    axes[0, 0].set_title("A: Rental Count Distribution across Weather Situations", fontweight='bold')
    axes[0, 0].set_xlabel("Weather Severity (1: Clear -> 4: Extreme Rain/Snow)")
    axes[0, 0].set_ylabel("Hourly Rental Count")
    axes[0, 0].set_xticks([0, 1, 2, 3])
    axes[0, 0].set_xticklabels(['1: Clear', '2: Mist/Cloudy', '3: Light Rain/Snow', '4: Heavy Rain/Snow'])

    # 2. Temperature vs Demand (Non-linear relationship)
    sns.regplot(data=df.sample(2000, random_state=42), x='temp_celsius', y='cnt', ax=axes[0, 1],
                scatter_kws={'alpha': 0.2, 'color': '#2980b9'}, line_kws={'color': '#e74c3c', 'linewidth': 2}, lowess=True)
    axes[0, 1].set_title("B: Non-Linear Temperature Response (LOWESS Curve)", fontweight='bold')
    axes[0, 1].set_xlabel("Temperature (°C)")
    axes[0, 1].set_ylabel("Hourly Rental Count")

    # 3. Humidity vs Demand
    sns.regplot(data=df.sample(2000, random_state=42), x='humidity_pct', y='cnt', ax=axes[1, 0],
                scatter_kws={'alpha': 0.2, 'color': '#27ae60'}, line_kws={'color': '#c0392b', 'linewidth': 2}, lowess=True)
    axes[1, 0].set_title("C: Relative Humidity Impact on Demand", fontweight='bold')
    axes[1, 0].set_xlabel("Relative Humidity (%)")
    axes[1, 0].set_ylabel("Hourly Rental Count")

    # 4. Windspeed vs Demand
    sns.regplot(data=df.sample(2000, random_state=42), x='windspeed_kmh', y='cnt', ax=axes[1, 1],
                scatter_kws={'alpha': 0.2, 'color': '#8e44ad'}, line_kws={'color': '#d35400', 'linewidth': 2}, lowess=True)
    axes[1, 1].set_title("D: Wind Speed Impact on Demand", fontweight='bold')
    axes[1, 1].set_xlabel("Wind Speed (km/h)")
    axes[1, 1].set_ylabel("Hourly Rental Count")

    plt.tight_layout()
    filepath = save_figure(fig, "fig2_weather_impact_analysis.png", output_dir)
    return fig, filepath


def plot_forecasting_results(y_true, y_pred, df_test, output_dir="outputs/figures"):
    """Plot actual vs predicted demand and residual diagnostic visualizations."""
    set_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    sample_df = pd.DataFrame({'datetime': df_test['datetime'], 'Actual': y_true, 'Predicted': y_pred}).iloc[:336]

    # 1. Actual vs Predicted Demand Time Series
    axes[0].plot(sample_df['datetime'], sample_df['Actual'], label='Actual Demand', color='#2c3e50', linewidth=1.8)
    axes[0].plot(sample_df['datetime'], sample_df['Predicted'], label='Forecasted Demand (LightGBM)', color='#e74c3c', linestyle='--', linewidth=1.5)
    axes[0].set_title("A: Actual vs. Forecasted Bikeshare Demand (14-Day Evaluation Window Snapshot)", fontweight='bold')
    axes[0].set_xlabel("Timestamp")
    axes[0].set_ylabel("Hourly Rental Count")
    axes[0].legend(loc='upper right')

    # 2. Parity Scatter Plot
    axes[1].scatter(y_true, y_pred, alpha=0.3, color='#2980b9', edgecolors='none', s=15)
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([0, max_val], [0, max_val], 'r--', label='Perfect 1:1 Prediction Line')
    axes[1].set_title("B: Parity Plot: Forecasted vs. Actual Hourly Demand", fontweight='bold')
    axes[1].set_xlabel("Actual Demand Count")
    axes[1].set_ylabel("Forecasted Demand Count")
    axes[1].legend()

    plt.tight_layout()
    filepath = save_figure(fig, "fig3_forecasting_performance.png", output_dir)
    return fig, filepath
