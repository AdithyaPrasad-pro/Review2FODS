"""Flask Backend API for the WattWise Household Electricity Consumption FODS Project.

Serves precomputed analysis and ML forecasting tables as JSON endpoints for Power BI,
Tableau, and Streamlit full-stack integration.

Run using:
    python app.py
Access at:
    http://127.0.0.1:5000/
"""

from pathlib import Path
from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "powerbi_exports"

EXPORTS = {
    "hourly": "hourly_summary.csv",
    "daily": "daily_summary.csv",
    "weekday": "weekday_summary.csv",
    "hour-of-day": "hour_of_day_summary.csv",
    "submeters": "submeter_summary.csv",
    "quality": "quality_and_test_summary.csv",
    "regression-comparison": "regression_model_comparison.csv",
    "regression-cross-validation": "regression_cross_validation.csv",
    "regression-predictions": "regression_test_predictions.csv",
}


def read_export(key):
    filename = EXPORTS.get(key)
    if filename is None:
        return None
    path = EXPORT_DIR / filename
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    # Replace NaN/NaT with None so JSON serialization succeeds
    frame = frame.astype(object).where(pd.notna(frame), None)
    return frame.to_dict(orient="records")


def export_response(key):
    data = read_export(key)
    if data is None:
        return jsonify({
            "error": f"Export file not found: {EXPORTS[key]}",
            "next_step": "Ensure powerbi_exports directory exists and contains CSV files."
        }), 404
    return jsonify(data)


@app.get("/")
def home():
    return jsonify({
        "project": "WattWise Household Electricity Consumption Analysis",
        "course": "Foundations of Data Science (FODS) Review 2",
        "status": "online",
        "available_endpoints": [f"/api/{name}" for name in EXPORTS],
        "full_stack_integration": "Power BI Desktop / Tableau / Streamlit Web App"
    })


@app.get("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "project": "WattWise Energy Intelligence Platform",
        "exports_available": len(list(EXPORT_DIR.glob("*.csv"))) if EXPORT_DIR.exists() else 0
    })


@app.get("/api/hourly")
def hourly():
    return export_response("hourly")


@app.get("/api/daily")
def daily():
    return export_response("daily")


@app.get("/api/weekday")
def weekday():
    return export_response("weekday")


@app.get("/api/hour-of-day")
def hour_of_day():
    return export_response("hour-of-day")


@app.get("/api/submeters")
def submeters():
    return export_response("submeters")


@app.get("/api/quality")
def quality():
    return export_response("quality")


@app.get("/api/regression-comparison")
def regression_comparison():
    return export_response("regression-comparison")


@app.get("/api/regression-predictions")
def regression_predictions():
    return export_response("regression-predictions")


@app.get("/api/regression-cross-validation")
def regression_cross_validation():
    return export_response("regression-cross-validation")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
