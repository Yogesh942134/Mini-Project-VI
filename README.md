# MGNREGA Fraud Detection System

A Streamlit-based dashboard for detecting potential fraud in MGNREGA (Mahatma Gandhi National Rural Employment Guarantee Act) data using a hybrid approach combining rule-based scoring and machine learning anomaly detection.

## Features

- **Hybrid Fraud Detection**: Combines rule-based expert system with Isolation Forest ML model
- **Modern Dark UI**: Sleek gradient-based interface with custom styling
- **Interactive Visualizations**: Plotly charts for risk analysis
- **Risk Scoring**: Each district receives a composite fraud probability score (0-1)
- **Data Export**: Download processed results with fraud scores

## How It Works

### Rule-Based Flags

The system flags anomalies in six key areas:

| Flag | Condition | Rationale |
|------|-----------|-----------|
| Worker Ratio | Workers > 1.5x Job Cards | Possible fake/ghost workers |
| Wage Deviation | Actual wage >1.5x or <0.5x average | Wage manipulation |
| Completion Ratio | >80% households complete 100 days | Unrealistic targets |
| Women Ratio | <20% or >80% | Deviation from mandated 33% participation |
| Cost per Work | Above 95th percentile | Inflated project costs |
| Payment Delay | <50% payments within 15 days | Violation of MGNREGA guidelines |

### Machine Learning Model

- **Algorithm**: Isolation Forest (unsupervised anomaly detection)
- **Contamination Rate**: 5% (expects ~5% of data to be anomalies)
- **Features**: 6 engineered ratios/metrics from the raw data

### Final Score Calculation

```
final_probability = 0.6 * ML_score + 0.4 * rule_based_score
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. Clone or download this repository:
   ```bash
   git clone <repository-url>
   cd mini-project-vi
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   ```bash
   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to `http://localhost:8501`

3. Upload a CSV file containing MGNREGA district data

4. View the fraud analysis results and download the processed data

### Required CSV Columns

The input CSV must contain these columns:

| Column | Description |
|--------|-------------|
| district_name | Name of the district |
| Total_No_of_Workers | Total workers employed |
| Total_No_of_JobCards_issued | Job cards issued |
| Wages | Total wages paid |
| Persondays_of_Central_Liability_so_far | Person-days generated |
| Average_Wage_rate_per_day_per_person | Average wage rate |
| Total_No_of_HHs_completed_100_Days_of_Wage_Employment | Households completing 100 days |
| Total_Households_Worked | Total households worked |
| Total_Exp | Total expenditure |
| Number_of_Completed_Works | Completed works count |
| Women_Persondays | Women person-days |
| percentage_payments_gererated_within_15_days | % payments within 15 days |

## Project Structure

```
mini-project-vi/
├── app.py              # Main Streamlit application
├── preprocess.py       # Data processing and ML model
├── utils.py            # Visualization functions
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Output Metrics

The system generates three key outputs for each district:

1. **fraud_score**: Sum of triggered rule-based flags (0-6)
2. **model_probability**: ML model anomaly score (0-1)
3. **final_probability**: Combined risk score (0-1)

## Visualizations

- **Top 10 High Risk Districts**: Horizontal bar chart showing highest risk districts
- **Risk Distribution**: Donut chart showing High/Medium/Low risk category distribution

## Risk Categories

| Category | Score Range | Color |
|----------|-------------|-------|
| High Risk | > 0.7 | Red |
| Medium Risk | 0.4 - 0.7 | Orange |
| Low Risk | <= 0.4 | Green |

## License

This project is for educational purposes.

## Contributing

Feel free to submit issues and enhancement requests.
