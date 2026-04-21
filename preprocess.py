"""
Data preprocessing module for MGNREGA Fraud Detection.

This module handles data cleaning, feature engineering, and fraud scoring
using a hybrid approach combining rule-based flags and machine learning
(Isolation Forest) anomaly detection.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def safe_divide(numerator, denominator, default=0):
    """
    Safely divide two values, returning a default if division by zero would occur.

    Args:
        numerator: The numerator value or Series
        denominator: The denominator value or Series
        default: Default value to return when division is not possible

    Returns:
        Result of division or default value
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(denominator != 0, numerator / denominator, default)
    return result


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process MGNREGA data to compute fraud risk scores.

    This function performs:
    1. Data cleaning (handle inf values, fill NaN)
    2. Feature engineering (compute ratios and metrics)
    3. Rule-based flagging (create binary flags for anomalies)
    4. ML-based anomaly detection (Isolation Forest)
    5. Hybrid score computation (60% ML + 40% rule-based)

    Args:
        df: Raw DataFrame with MGNREGA district-level data

    Returns:
        DataFrame with added fraud risk scores and probabilities

    Raises:
        ValueError: If required columns are missing
        RuntimeError: If model training fails
    """
    df = df.copy()

    # ========== Group by District ==========
    # If a district appears multiple times, aggregate by taking the mean of numeric columns
    if "district_name" in df.columns:
        duplicate_count = df.duplicated(subset=["district_name"]).sum()
        if duplicate_count > 0:
            # Get numeric columns for aggregation
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            # Group by district_name and aggregate numeric columns by mean
            df = df.groupby("district_name")[numeric_cols].mean().reset_index()

    # Replace infinite values with NaN for clean processing
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ========== Feature Engineering ==========

    # Worker to job card ratio - high values suggest fake workers
    df["worker_to_jobcard_ratio"] = safe_divide(
        df["Total_No_of_Workers"],
        df["Total_No_of_JobCards_issued"]
    )

    # Actual wage per day and deviation from average
    df["actual_wage_per_day"] = safe_divide(
        df["Wages"],
        df["Persondays_of_Central_Liability_so_far"]
    )
    df["wage_deviation_ratio"] = safe_divide(
        df["actual_wage_per_day"],
        df["Average_Wage_rate_per_day_per_person"]
    )

    # Completion ratio - households completing 100 days
    df["completion_ratio"] = safe_divide(
        df["Total_No_of_HHs_completed_100_Days_of_Wage_Employment"],
        df["Total_Households_Worked"]
    )

    # Women participation ratio
    df["women_ratio"] = safe_divide(
        df["Women_Persondays"],
        df["Persondays_of_Central_Liability_so_far"]
    )

    # Cost per work - high values may indicate inflated costs
    df["cost_per_work"] = safe_divide(
        df["Total_Exp"],
        df["Number_of_Completed_Works"]
    )

    # Fill remaining NaN values with 0
    df.fillna(0, inplace=True)

    # ========== Rule-Based Flagging ==========

    # Flag 1: Worker ratio > 1.5 (more workers than job cards)
    df["flag_worker"] = (df["worker_to_jobcard_ratio"] > 1.5).astype(int)

    # Flag 2: Wage deviation (too high or too low compared to average)
    df["flag_wage"] = (
        (df["wage_deviation_ratio"] > 1.5) |
        (df["wage_deviation_ratio"] < 0.5)
    ).astype(int)

    # Flag 3: Unrealistically high completion ratio
    df["flag_completion"] = (df["completion_ratio"] > 0.8).astype(int)

    # Flag 4: Women ratio outside normal range (should be around 0.33-0.5)
    df["flag_women"] = (
        (df["women_ratio"] < 0.2) |
        (df["women_ratio"] > 0.8)
    ).astype(int)

    # Flag 5: Cost per work above 95th percentile
    cost_threshold = df["cost_per_work"].quantile(0.95)
    df["flag_cost"] = (df["cost_per_work"] > cost_threshold).astype(int)

    # Flag 6: Low percentage of payments within 15 days (delayed payments)
    df["flag_delay"] = (df["percentage_payments_gererated_within_15_days"] < 50).astype(int)

    # Sum all flags to get composite fraud score
    flags = ["flag_worker", "flag_wage", "flag_completion", "flag_women", "flag_cost", "flag_delay"]
    df["fraud_score"] = df[flags].sum(axis=1)

    # ========== Machine Learning - Isolation Forest ==========

    features = [
        "worker_to_jobcard_ratio",
        "wage_deviation_ratio",
        "completion_ratio",
        "women_ratio",
        "cost_per_work",
        "percentage_payments_gererated_within_15_days"
    ]

    X = df[features]

    # Standardize features for Isolation Forest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest model (5% contamination rate)
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)

    # Get anomaly scores (more negative = more anomalous)
    scores = model.decision_function(X_scaled)

    # Convert to 0-1 probability scale (invert so anomalies have higher scores)
    prob_scaler = MinMaxScaler()
    df["model_probability"] = 1 - prob_scaler.fit_transform(scores.reshape(-1, 1))

    # ========== Hybrid Score Computation ==========

    # Normalize rule-based score to 0-1 range
    df["rule_score_norm"] = df["fraud_score"] / df["fraud_score"].max()

    # Combine ML and rule-based scores (60% ML, 40% rules)
    df["final_probability"] = (
        0.6 * df["model_probability"] +
        0.4 * df["rule_score_norm"]
    )

    return df
