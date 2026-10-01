# Heart Disease Risk Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-prediction-using-catboostxgboost.streamlit.app/)

Machine learning web application for predicting heart disease risk using CatBoost and XGBoost on BRFSS 2015 data.

## Live Demo
**https://heart-disease-prediction-using-catboostxgboost.streamlit.app/**

## Features
- **Two input modes:**
  - Manual Entry: structured form with sliders and radio buttons
  - AI-Assisted: users describe symptoms in plain language; LLM extracts structured features
- **LLM extraction with multi-provider fallback** (Gemini -> Pollinations)
- **Mandatory user confirmation** before prediction
- Real-time risk prediction with calibrated CatBoost + XGBoost
- Dual SHAP explainability (CatBoost + XGBoost)
- Interactive visualizations (Risk Gauge, Model Comparison, Radar Chart)
- PDF report export with 8 sections
- Light/Dark/System theme toggle

## Dataset
BRFSS 2015 (Behavioral Risk Factor Surveillance System)
- 253,680 records
- 21 features
- 9.42% heart disease prevalence

## Model Performance (Test Set)
| Model | ROC-AUC | PR-AUC | Brier Score |
|-------|---------|--------|-------------|
| CatBoost | 0.8509 | 0.3717 | 0.0701 |
| XGBoost | 0.8494 | 0.3705 | 0.0703 |

## LLM Safety Design
- LLM only extracts features, never predicts disease
- Strict JSON schema with field validation
- Mandatory user confirmation before prediction
- Graceful fallback to Manual Entry if LLM fails
- Multi-provider (Gemini -> Pollinations) for reliability

## Disclaimer
Research prototype only -- NOT for clinical decision-making.

## Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```
