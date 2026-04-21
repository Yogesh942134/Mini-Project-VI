"""
Utility functions for data visualization.
Contains Plotly chart generation functions for the fraud detection dashboard.
"""
import pandas as pd
import plotly.express as px


def bar_chart(df):
    """
    Create a horizontal bar chart showing top 10 high-risk districts.

    Args:
        df: DataFrame with 'final_probability' and 'district_name' columns

    Returns:
        Plotly Figure object
    """
    top10 = df.head(10).copy()

    fig = px.bar(
        top10,
        x="final_probability",
        y="district_name",
        orientation="h",
        color="final_probability",
        color_continuous_scale=[[0, "#1a1a2e"], [0.5, "#ff6b6b"], [1, "#dc2626"]],
        title="Top 10 High Risk Districts"
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        title_font_size=20,
        font=dict(family="Inter, sans-serif", size=12),
        xaxis_title="Fraud Probability Score",
        yaxis_title="District",
        coloraxis_colorbar=dict(title="Risk Score"),
        plot_bgcolor="rgba(0,0,0,0.1)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")

    return fig


def risk_distribution_chart(df):
    """
    Create a donut chart showing the distribution of risk levels.

    Risk levels:
    - High Risk: final_probability > 0.7
    - Medium Risk: 0.4 < final_probability <= 0.7
    - Low Risk: final_probability <= 0.4

    Args:
        df: DataFrame with 'final_probability' and 'district_name' columns

    Returns:
        Plotly Figure object
    """
    def categorize_risk(prob):
        if prob > 0.7:
            return "High Risk"
        elif prob > 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    df_risk = df.copy()
    df_risk["risk_category"] = df_risk["final_probability"].apply(categorize_risk)

    risk_counts = df_risk.groupby("risk_category")["district_name"].count().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]

    risk_order = ["High Risk", "Medium Risk", "Low Risk"]
    risk_counts = risk_counts[risk_counts["Risk Level"].isin(risk_order)]
    risk_counts["Risk Level"] = risk_counts["Risk Level"].astype(
        pd.CategoricalDtype(categories=risk_order, ordered=True)
    )
    risk_counts = risk_counts.sort_values("Risk Level")

    fig = px.pie(
        risk_counts,
        values="Count",
        names="Risk Level",
        title="District Distribution by Risk Level",
        color="Risk Level",
        color_discrete_map={
            "High Risk": "#dc2626",
            "Medium Risk": "#f59e0b",
            "Low Risk": "#10b981"
        },
        hole=0.5
    )

    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        title_font_size=20,
        font=dict(family="Inter, sans-serif", size=12),
        plot_bgcolor="rgba(0,0,0,0.1)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(size=14, color="#ffffff")
    )

    return fig
