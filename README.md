# WattWise: Household Electricity Consumption Analysis & Full-Stack Deployment
### Foundations of Data Science (FODS) — Review 2 Case Study

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-Integrated-yellow.svg)](https://powerbi.microsoft.com/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

---

## 📌 Executive Summary

**WattWise** is an end-to-end Foundations of Data Science case study investigating longitudinal electric power consumption across **2,075,259 minute-level measurements** (December 2006 – November 2010) from the **UCI Machine Learning Repository**.

The project rigorously addresses all **10 Review 2 Rubric Criteria**:
1. **Real Dataset:** UCI Individual Household Electric Power Consumption (2.07M observations, 9 attributes).
2. **Preprocessing & EDA:** Missing-value time-series forward interpolation, temporal feature creation, distribution fitting, and scaling comparisons.
3. **Data Insights Learned:** High right-skewness (Mean: 1.090 kW, Median: 0.614 kW), 4.19× evening peak surge (1.890 kW at 20:00 vs 0.451 kW at 04:00), winter heating seasonality.
4. **15 Data Analysis Questions:** Rigorously formulated and solved.
5. **Implementation & Justification:** Dedicated visual proofs for each question (Plotly/Matplotlib/Seaborn).
6. **Decision Making:** Demand-side energy management, evening peak-shaving, and weekend routine rescheduling.
7. **Hypothesis Testing:** Formal proof via Welch's Two-Sample t-test ($t = 106.33, p < 10^{-15}$, $\alpha=0.05$), conclusively rejecting $H_0$ and proving weekend consumption surge (+17.95%).
8. **Machine Learning Model Justification:** Continuous next-hour demand forecasting with **Random Forest Regressor** ($R^2 = 0.538$, $\text{RMSE} = 0.495\text{ kW}$) and Decision Tree vs. baseline.
9. **Power BI / Tableau Visualization:** Complete Star Schema data model, automated export pipeline (9 tables), DAX measures, and Power Query M live REST connector.
10. **Conclusion & Full-Stack Bonus:** Live Flask REST API backend (`app.py`) + Multi-Page Interactive Streamlit Web Application (`streamlit_app.py`).

> **Note on Academic Integrity:** This is strictly an empirical Foundations of Data Science (FODS) case study emphasizing data preprocessing, statistical inference, hypothesis testing, and continuous energy forecasting. It is independent of any course Machine Learning capstone.

---

## 🏗️ Full-Stack System Architecture

```text
  [ UCI Raw Dataset (2,075,259 records) ]
                     │
                     ▼
  [ Jupyter Notebook: Preprocessing, EDA, CLT, Welch Test, ML Forecasting ]
                     │
                     ▼
  [ powerbi_exports/ (9 Processed Analytical Tables) ]
         ┌───────────┴───────────────────────┐
         ▼                                   ▼
  [ Flask REST API: app.py ]      [ Power BI / Tableau Desktop ]
  - Live JSON endpoints           - Star Schema Fact/Dim Model
  - /api/daily, /api/hourly       - Custom DAX Measures
  - /api/regression-comparison    - Direct Power Query M Script
         │
         ▼
  [ Streamlit Cloud Web App: streamlit_app.py ]
  - Page 1: Executive KPI Dashboard & Diurnal Trends
  - Page 2: 15 Research Questions Interactive Explorer
  - Page 3: Hypothesis Testing & Welch Proof Hub
  - Page 4: ML Demand Forecasting Lab & What-If Estimator
  - Page 5: Power BI & Full-Stack Hub
```

---

## 📊 Summary of Hypothesis Testing Evidence (Rubric Item 7)

| Hypothesis | Mathematical Formulation | Test Statistic | p-value | Decision Rule | Empirical Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Null ($H_0$)** | $\mu_{\text{weekend}} = \mu_{\text{weekday}}$ | **Welch's $t = 106.33$** | **$p < 10^{-15}$** | Reject $H_0$ at $\alpha = 0.05$ | **Conclusively Rejected.** Weekend consumption is systematically higher by $+0.186\text{ kW}$ ($+17.95\%$). |
| **Alternative ($H_1$)** | $\mu_{\text{weekend}} > \mu_{\text{weekday}}$ | $df \approx 1,080,000$ | $95\%\text{ CI: } [0.183, 0.190]$ | Accept $H_1$ | Effect size Cohen's $d = 0.177$. Direct justification for weekend energy management. |

---

## 🤖 Machine Learning Model Benchmarks (Rubric Item 8)

Evaluated on the chronological holdout test horizon (later 20% of timeline):

| Model | MAE (kW) | RMSE (kW) | $R^2$ Variance Explained | Model Justification |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | **0.3445** | **0.4948** | **0.5384** | **Best Performing.** Captures nonlinear diurnal interactions and lag features. |
| **Decision Tree Regressor** | 0.3530 | 0.5084 | 0.5127 | Fast, interpretable threshold splits. |
| **Linear Regression** | 0.3880 | 0.5380 | 0.4542 | Baseline parametric reference model. |
| **Persistence Baseline** | 0.4420 | 0.6180 | 0.3210 | Naive $P_t = P_{t-1}$ benchmark. |

---

## 📈 Power BI Deployment & DAX Measures (Rubric Item 9)

See [**`POWERBI_DASHBOARD_GUIDE.md`**](POWERBI_DASHBOARD_GUIDE.md) for full instructions:
- **Average Active Power:** `Avg_Active_Power_kW = AVERAGE('hourly_summary'[Mean_Active_Power_kW])`
- **Peak Demand:** `Peak_Active_Power_kW = MAX('hourly_summary'[Max_Active_Power_kW])`
- **Weekend/Weekday Variance %:** `Weekend_Weekday_Variance_% = DIVIDE(WeekendAvg - WeekdayAvg, WeekdayAvg, 0) * 100`
- **Forecast Error MAE:** `Forecast_MAE_kW = AVERAGEX('regression_test_predictions', ABS(Actual - Predicted))`

---

## 🚀 Local Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/AdithyaPrasad-pro/Review2FODS.git
cd Review2FODS
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
python -m streamlit run streamlit_app.py
```
*(Opens automatically at `http://localhost:8501`)*

### 4. (Optional) Run the Backend REST API
```bash
python app.py
```
*(Runs at `http://127.0.0.1:5000`)*

---

## 🌐 Deploying to Streamlit Community Cloud

1. Push this repository to GitHub: `https://github.com/AdithyaPrasad-pro/Review2FODS`.
2. Go to [**share.streamlit.io**](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. Configure the deployment settings:
   - **Repository:** `AdithyaPrasad-pro/Review2FODS`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **"Deploy!"**.
