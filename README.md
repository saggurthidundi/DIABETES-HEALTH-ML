# Metabolic Intelligence Clinical Diagnostic & PDF Reporting Suite

A high-accuracy clinical machine learning platform comparing Logistic Regression, Random Forest, and Ridge Regression across clinical laboratory biomarkers and holistic lifestyle metrics (Diet, Physical Activity, Sleep, Stress).

## Features
- **High-Accuracy ML Benchmark**: Classification (Accuracy, Precision, Recall, F1-Score, Confusion Matrix) and Regression ($R^2$, RMSE, MAE).
- **Holistic Lifestyle Evaluation**: Ingestion of Dietary patterns, Physical Activity, Sleep hygiene, and Stress indicators.
- **Interactive Streamlit UI**: Dark medical glassmorphic design with Plotly diagnostic dials.
- **Vector PDF Generator**: Multi-table, clinical-grade medical dossier generation using `ReportLab`.

## Quick Start
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run ui/app.py