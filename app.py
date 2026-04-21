"""
MGNREGA Fraud Detection System - Main Application

A Streamlit-based dashboard for detecting potential fraud in MGNREGA data
using a hybrid approach combining rule-based scoring and machine learning
(Isolation Forest) anomaly detection.
"""
import streamlit as st
import pandas as pd
from preprocess import process_data
from utils import bar_chart, risk_distribution_chart


# Required columns that must be present in the uploaded CSV
REQUIRED_COLUMNS = [
    "district_name",
    "Total_No_of_Workers",
    "Total_No_of_JobCards_issued",
    "Wages",
    "Persondays_of_Central_Liability_so_far",
    "Average_Wage_rate_per_day_per_person",
    "Total_No_of_HHs_completed_100_Days_of_Wage_Employment",
    "Total_Households_Worked",
    "Total_Exp",
    "Number_of_Completed_Works",
    "Women_Persondays",
    "percentage_payments_gererated_within_15_days"
]


def validate_dataframe(df: pd.DataFrame) -> tuple:
    """
    Validate that the uploaded DataFrame has all required columns.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"

    if df.empty:
        return False, "Uploaded file is empty"

    return True, ""


def main():
    """Main application function."""
    st.set_page_config(
        page_title="MGNREGA Fraud Detection",
        page_icon=":shield:",
        layout="wide"
    )

    # Custom CSS for modern styling
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d9ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #00d9ff;
    }

    div[data-testid="stMetricLabel"] {
        color: #a0a0a0;
        font-size: 0.9rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        border-left: 4px solid #00d9ff;
        padding-left: 15px;
        margin: 30px 0 20px 0;
    }

    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d9ff, #00ff88);
        color: #1a1a2e;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,217,255,0.4);
    }

    /* File uploader */
    .stFileUploader {
        border-radius: 10px;
        border: 2px dashed rgba(0,217,255,0.3);
    }

    /* Info boxes */
    .info-box {
        background: rgba(0,217,255,0.1);
        border-left: 4px solid #00d9ff;
        padding: 15px 20px;
        border-radius: 0 10px 10px 0;
        margin: 20px 0;
    }

    /* Warning boxes */
    .warning-box {
        background: rgba(245,158,11,0.1);
        border-left: 4px solid #f59e0b;
        padding: 15px 20px;
        border-radius: 0 10px 10px 0;
    }

    /* Error boxes */
    .error-box {
        background: rgba(220,38,38,0.1);
        border-left: 4px solid #dc2626;
        padding: 15px 20px;
        border-radius: 0 10px 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-title">MGNREGA Fraud Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered anomaly detection for MGNREGA data analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>How it works:</strong> This system uses a hybrid approach combining
    <strong>Rule-based scoring</strong> (expert-defined flags) and
    <strong>Machine Learning</strong> (Isolation Forest) to detect potential fraud.
    Final risk score = 60% ML model + 40% rule-based scoring.
    </div>
    """, unsafe_allow_html=True)

    # File uploader
    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file is None:
        st.markdown("""
        <div style="text-align: center; padding: 50px; color: #a0a0a0;">
            <p style="font-size: 1.2rem;">Upload a CSV file containing MGNREGA district data to begin analysis</p>
            <p>Required columns: district_name, Total_No_of_Workers, Total_No_of_JobCards_issued, Wages, etc.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        st.markdown('<div class="error-box"><strong>Error:</strong> The uploaded file is empty</div>', unsafe_allow_html=True)
        return
    except Exception as e:
        st.markdown(f'<div class="error-box"><strong>Error:</strong> {str(e)}</div>', unsafe_allow_html=True)
        return

    # Validate the uploaded data
    is_valid, error_msg = validate_dataframe(df)
    if not is_valid:
        st.markdown(f'<div class="error-box"><strong>Invalid Data:</strong> {error_msg}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="warning-box"><strong>Required columns:</strong> {", ".join(REQUIRED_COLUMNS)}</div>', unsafe_allow_html=True)
        return

    try:
        # Process the data
        df = process_data(df)
    except Exception as e:
        st.markdown(f'<div class="error-box"><strong>Processing Error:</strong> {str(e)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="warning-box">Please ensure all numeric columns contain valid numeric values</div>', unsafe_allow_html=True)
        return

    # Sort by final probability (highest risk first)
    df_sorted = df.sort_values(by="final_probability", ascending=False)

    # Display metrics with custom styling
    st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    total_records = len(df)
    high_risk_count = len(df[df["final_probability"] > 0.7])
    avg_risk = df["final_probability"].mean()

    with col1:
        st.metric(
            label="Total Districts",
            value=f"{total_records:,}",
            delta=None
        )
    with col2:
        st.metric(
            label="High Risk (>0.7)",
            value=f"{high_risk_count:,}",
            delta=f"{(high_risk_count/total_records*100):.1f}% of total" if total_records > 0 else None
        )
    with col3:
        st.metric(
            label="Average Risk Score",
            value=f"{avg_risk:.3f}",
            delta="Low" if avg_risk < 0.4 else "Medium" if avg_risk < 0.7 else "High"
        )

    st.divider()

    # Top 20 high-risk districts table
    st.markdown('<div class="section-header">High Risk Districts</div>', unsafe_allow_html=True)
    st.markdown("Top 20 districts ranked by fraud probability score")

    top20_df = df_sorted.head(20)[[
        "district_name",
        "final_probability",
        "fraud_score"
    ]].copy()
    top20_df.columns = ["District", "Risk Score", "Fraud Flags"]

    st.dataframe(
        top20_df.style.format({
            "Risk Score": "{:.3f}",
            "Fraud Flags": "{:.0f}"
        }).background_gradient(
            subset=["Risk Score"],
            cmap="Reds"
        ),
        use_container_width=True,
        height=400
    )

    st.divider()

    # Visualizations
    st.markdown('<div class="section-header">Visual Analysis</div>', unsafe_allow_html=True)

    st.plotly_chart(
        bar_chart(df_sorted),
        use_container_width=True,
        key="bar_chart_unique"
    )

    st.plotly_chart(
        risk_distribution_chart(df),
        use_container_width=True,
        key="risk_dist_unique"
    )

    st.divider()

    # High-risk cases detailed view
    high_fraud_df = df[df["final_probability"] > 0.7]

    if not high_fraud_df.empty:
        st.markdown(f'<div class="section-header">High Risk Cases ({len(high_fraud_df)} districts)</div>', unsafe_allow_html=True)

        high_risk_df = high_fraud_df[[
            "district_name",
            "final_probability",
            "fraud_score"
        ]].copy()
        high_risk_df.columns = ["District", "Risk Score", "Fraud Flags"]

        st.dataframe(
            high_risk_df.style.format({
                "Risk Score": "{:.3f}",
                "Fraud Flags": "{:.0f}"
            }).background_gradient(
                subset=["Risk Score"],
                cmap="Reds"
            ),
            use_container_width=True,
            height=300
        )
    else:
        st.markdown('<div class="info-box">No high-risk districts detected (threshold > 0.7)</div>', unsafe_allow_html=True)

    # Download section
    st.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)

    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download Complete Results (CSV)",
        data=csv_data,
        file_name="mgnrega_fraud_analysis_results.csv",
        mime="text/csv",
        help="Download the complete analysis with all risk scores and flags"
    )


if __name__ == "__main__":
    main()
