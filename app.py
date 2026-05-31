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

    # ── Professional CSS ──────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #0d1117;
        background-image:
            radial-gradient(ellipse 80% 50% at 20% -10%, rgba(14, 165, 233, 0.08) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 110%, rgba(20, 184, 166, 0.06) 0%, transparent 60%);
    }

    /* ── Typography ── */
    .page-eyebrow {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #0ea5e9;
        margin: 0 0 0.5rem 0;
    }

    .page-title {
        font-size: 2.1rem;
        font-weight: 600;
        color: #f1f5f9;
        line-height: 1.2;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
    }

    .page-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        font-weight: 300;
        margin: 0 0 2.5rem 0;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        position: relative;
        overflow: hidden;
    }

    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #0ea5e9, #14b8a6);
    }

    div[data-testid="stMetricLabel"] > div {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #475569 !important;
    }

    div[data-testid="stMetricValue"] > div {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        letter-spacing: -0.03em !important;
        font-family: 'DM Mono', monospace !important;
    }

    div[data-testid="stMetricDelta"] > div {
        font-size: 0.78rem !important;
        font-weight: 400 !important;
        color: #475569 !important;
    }

    /* ── Section headers ── */
    .section-label {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 2.5rem 0 1rem 0;
    }

    .section-label-line {
        flex: 1;
        height: 1px;
        background: #21262d;
    }

    .section-label-text {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #475569;
        white-space: nowrap;
    }

    /* ── Info / warning / error banners ── */
    .banner {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 1.2rem 0;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .banner-info {
        background: rgba(14, 165, 233, 0.06);
        border: 1px solid rgba(14, 165, 233, 0.18);
        color: #7dd3fc;
    }

    .banner-warning {
        background: rgba(245, 158, 11, 0.06);
        border: 1px solid rgba(245, 158, 11, 0.2);
        color: #fcd34d;
    }

    .banner-error {
        background: rgba(239, 68, 68, 0.06);
        border: 1px solid rgba(239, 68, 68, 0.2);
        color: #fca5a5;
    }

    .banner-icon {
        font-size: 1rem;
        margin-top: 1px;
        flex-shrink: 0;
    }

    .banner strong {
        font-weight: 600;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
        color: #334155;
    }

    .empty-state-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        opacity: 0.4;
    }

    .empty-state h3 {
        font-size: 1.1rem;
        font-weight: 500;
        color: #475569;
        margin: 0 0 0.5rem 0;
    }

    .empty-state p {
        font-size: 0.85rem;
        color: #334155;
        margin: 0;
        max-width: 480px;
        margin: 0 auto;
        line-height: 1.7;
    }

    /* ── File uploader ── */
    .stFileUploader > div {
        border: 1px dashed #21262d !important;
        border-radius: 10px !important;
        background: #161b22 !important;
        transition: border-color 0.2s;
    }

    .stFileUploader > div:hover {
        border-color: #0ea5e9 !important;
    }

    /* ── Dataframe ── */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        border: 1px solid #21262d;
        overflow: hidden;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: transparent !important;
        border: 1px solid #21262d !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.2rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease !important;
    }

    .stDownloadButton > button:hover {
        border-color: #0ea5e9 !important;
        color: #0ea5e9 !important;
        background: rgba(14, 165, 233, 0.06) !important;
    }

    /* ── Divider ── */
    hr {
        border: none !important;
        border-top: 1px solid #21262d !important;
        margin: 2rem 0 !important;
    }

    /* ── Plotly chart containers ── */
    .stPlotlyChart {
        border: 1px solid #21262d;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown('<p class="page-eyebrow">Ministry of Rural Development · Analytics</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">MGNREGA Fraud Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Hybrid anomaly detection — 60% Isolation Forest · 40% rule-based scoring</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="banner banner-info">
        <span class="banner-icon">ℹ</span>
        <span>
            <strong>How it works:</strong> Upload a district-level CSV to surface anomalies.
            Rule-based flags capture domain heuristics; the ML model catches statistical outliers.
            Final risk score blends both signals.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────────
    file = st.file_uploader("Upload CSV File", type=["csv"], label_visibility="collapsed")

    if file is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📂</div>
            <h3>No file uploaded yet</h3>
            <p>
                Drop a CSV containing MGNREGA district data above to begin analysis.<br>
                Required fields include <code>district_name</code>, <code>Total_No_of_Workers</code>,
                <code>Wages</code>, and 9 other columns.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Read CSV ──────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        st.markdown('<div class="banner banner-error"><span class="banner-icon">✕</span><span><strong>Error:</strong> The uploaded file is empty.</span></div>', unsafe_allow_html=True)
        return
    except Exception as e:
        st.markdown(f'<div class="banner banner-error"><span class="banner-icon">✕</span><span><strong>Error:</strong> {str(e)}</span></div>', unsafe_allow_html=True)
        return

    # ── Validate ──────────────────────────────────────────────────────────────
    is_valid, error_msg = validate_dataframe(df)
    if not is_valid:
        st.markdown(f'<div class="banner banner-error"><span class="banner-icon">✕</span><span><strong>Invalid data:</strong> {error_msg}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="banner banner-warning"><span class="banner-icon">⚠</span><span><strong>Required columns:</strong> {", ".join(REQUIRED_COLUMNS)}</span></div>', unsafe_allow_html=True)
        return

    # ── Process ───────────────────────────────────────────────────────────────
    try:
        df = process_data(df)
    except Exception as e:
        st.markdown(f'<div class="banner banner-error"><span class="banner-icon">✕</span><span><strong>Processing error:</strong> {str(e)}</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="banner banner-warning"><span class="banner-icon">⚠</span><span>Ensure all numeric columns contain valid numeric values.</span></div>', unsafe_allow_html=True)
        return

    df_sorted = df.sort_values(by="final_probability", ascending=False)

    # ── Overview metrics ──────────────────────────────────────────────────────
    _section("Overview")

    total_records   = len(df)
    high_risk_count = len(df[df["final_probability"] > 0.7])
    avg_risk        = df["final_probability"].mean()
    risk_label      = "Low" if avg_risk < 0.4 else ("Medium" if avg_risk < 0.7 else "High")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Districts", f"{total_records:,}")
    with col2:
        st.metric(
            "High Risk  (> 0.7)",
            f"{high_risk_count:,}",
            delta=f"{(high_risk_count / total_records * 100):.1f}% of total" if total_records > 0 else None,
        )
    with col3:
        st.metric("Average Risk Score", f"{avg_risk:.3f}", delta=risk_label)

    st.divider()

    # ── Top 20 table ──────────────────────────────────────────────────────────
    _section(f"All Districts by Risk Score  ·  {len(df_sorted)} records")

    top20_df = (
        df_sorted[["district_name", "final_probability", "fraud_score"]]
        .copy()
        .rename(columns={"district_name": "District", "final_probability": "Risk Score", "fraud_score": "Fraud Flags"})
    )

    st.dataframe(
        top20_df.style
            .format({"Risk Score": "{:.3f}", "Fraud Flags": "{:.0f}"})
            .background_gradient(subset=["Risk Score"], cmap="Reds"),
        use_container_width=True,
        height=400,
    )

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    _section("Visual Analysis")

    st.plotly_chart(bar_chart(df_sorted), use_container_width=True, key="bar_chart_unique")
    st.plotly_chart(risk_distribution_chart(df), use_container_width=True, key="risk_dist_unique")

    st.divider()

    # ── High-risk cases ───────────────────────────────────────────────────────
    high_fraud_df = df[df["final_probability"] > 0.7]

    if not high_fraud_df.empty:
        _section(f"High Risk Cases  ·  {len(high_fraud_df)} districts")

        high_risk_df = (
            high_fraud_df[["district_name", "final_probability", "fraud_score"]]
            .copy()
            .rename(columns={"district_name": "District", "final_probability": "Risk Score", "fraud_score": "Fraud Flags"})
        )

        st.dataframe(
            high_risk_df.style
                .format({"Risk Score": "{:.3f}", "Fraud Flags": "{:.0f}"})
                .background_gradient(subset=["Risk Score"], cmap="Reds"),
            use_container_width=True,
            height=300,
        )
    else:
        st.markdown('<div class="banner banner-info"><span class="banner-icon">ℹ</span><span>No high-risk districts detected (threshold&nbsp;&gt;&nbsp;0.7).</span></div>', unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────
    _section("Export")

    st.download_button(
        label="⬇  Download complete results (CSV)",
        data=df.to_csv(index=False),
        file_name="mgnrega_fraud_analysis_results.csv",
        mime="text/csv",
        help="Download the full analysis with all risk scores and flags",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(label: str) -> None:
    """Render a subtle ruled section label."""
    st.markdown(
        f'<div class="section-label">'
        f'<span class="section-label-line"></span>'
        f'<span class="section-label-text">{label}</span>'
        f'<span class="section-label-line"></span>'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
