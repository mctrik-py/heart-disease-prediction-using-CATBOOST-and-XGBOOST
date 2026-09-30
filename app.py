
"""
app.py
Interactive Streamlit application for heart disease risk prediction.
Features: session state for predictions, independent PDF generation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from datetime import datetime

from pdf_report import generate_pdf_report

try:
    from llm_extractor import extract_features, DEFAULT_VALUES, FIELD_RULES
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'extraction_confirmed' not in st.session_state:
    st.session_state.extraction_confirmed = False
if 'confirmed_data' not in st.session_state:
    st.session_state.confirmed_data = None


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="heart",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@keyframes heartbeat {
    0%   { transform: scale(1); }
    14%  { transform: scale(1.15); }
    28%  { transform: scale(1); }
    42%  { transform: scale(1.15); }
    70%  { transform: scale(1); }
}
.beating-heart {
    display: inline-block;
    animation: heartbeat 1.3s infinite;
    font-size: 3rem;
    vertical-align: middle;
    margin-right: 10px;
}
.metric-card {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 10px;
}
div.stButton > button {
    transition: all 0.3s ease;
    border-radius: 8px;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)


def custom_alert(message, alert_type="warning"):
    colors = {
        "warning": {"bg": "#FFF3CD", "border": "#FFECB5", "text": "#664D03", "icon": "&#9888;&#65039;"},
        "success": {"bg": "#D1E7DD", "border": "#BADBCC", "text": "#0F5132", "icon": "&#9989;"},
        "error":   {"bg": "#F8D7DA", "border": "#F5C2C7", "text": "#842029", "icon": "&#10060;"},
        "info":    {"bg": "#CFE2FF", "border": "#B6D4FE", "text": "#084298", "icon": "&#8505;&#65039;"},
    }
    c = colors.get(alert_type, colors["warning"])
    st.markdown(f"""
    <div style="
        background-color: {c['bg']};
        border: 1px solid {c['border']};
        color: {c['text']};
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 1rem;
        font-weight: 500;
    ">
        {c['icon']} {message}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# THEME TOGGLE
# ============================================================
theme_col1, theme_col2 = st.columns([4, 1])
with theme_col2:
    selected_theme = st.selectbox(
        "Theme",
        options=["System Default", "Dark Mode", "Light Mode"],
        index=0,
        label_visibility="collapsed"
    )

if selected_theme == "Dark Mode":
    st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .metric-card { background-color: #1F2937; color: #FAFAFA; }
    </style>
    """, unsafe_allow_html=True)
elif selected_theme == "Light Mode":
    st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #0E1117; }
    .metric-card { background-color: #F3F4F6; color: #0E1117; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    cat_calibrated = joblib.load('models/catboost_calibrated.pkl')
    xgb_calibrated = joblib.load('models/xgboost_calibrated.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    cat_base = cat_calibrated.calibrated_classifiers_[0].estimator.estimator
    xgb_base = xgb_calibrated.calibrated_classifiers_[0].estimator.estimator
    return cat_calibrated, xgb_calibrated, cat_base, xgb_base, feature_names


@st.cache_resource
def load_shap_explainers():
    cat_calibrated = joblib.load('models/catboost_calibrated.pkl')
    xgb_calibrated = joblib.load('models/xgboost_calibrated.pkl')
    cat_base = cat_calibrated.calibrated_classifiers_[0].estimator.estimator
    xgb_base = xgb_calibrated.calibrated_classifiers_[0].estimator.estimator
    
    X_train = pd.read_csv('data/X_train_final.csv')
    np.random.seed(42)
    background_idx = np.random.choice(len(X_train), size=100, replace=False)
    X_background = X_train.iloc[background_idx]
    
    explainer_cat = shap.TreeExplainer(cat_base, data=X_background, feature_perturbation='interventional')
    explainer_xgb = shap.TreeExplainer(xgb_base, data=X_background, feature_perturbation='interventional')
    return explainer_cat, explainer_xgb


cat_calibrated, xgb_calibrated, cat_base, xgb_base, feature_names = load_models()
explainer_cat, explainer_xgb = load_shap_explainers()

# ============================================================
# SESSION STATE
# ============================================================
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None
if 'pdf_filename' not in st.session_state:
    st.session_state.pdf_filename = None


# ============================================================
# HEADING
# ============================================================
st.markdown(
    '<h1><span class="beating-heart">&#10084;&#65039;</span>Heart Disease Risk Prediction</h1>',
    unsafe_allow_html=True
)

st.markdown("""
This interactive tool uses **CatBoost** and **XGBoost** machine learning models 
to estimate heart disease risk based on demographic, lifestyle, and health factors.
""")

custom_alert("Research prototype only -- NOT for clinical decision-making.", "warning")
st.markdown("---")


# ============================================================
# PATIENT INPUT
# ============================================================
st.header("Step 1 -- Enter Patient Information")

input_mode = st.radio(
    "Choose input method:",
    ["Manual Entry", "Describe Symptoms (AI-Assisted)"],
    horizontal=True,
    key="input_mode_selector"
)

st.markdown("---")


if input_mode == "Describe Symptoms (AI-Assisted)":
    
    if not LLM_AVAILABLE:
        custom_alert(
            "AI extraction unavailable. Please use Manual Entry.",
            "warning"
        )
    else:
        st.markdown("### Describe your health condition in plain language")
        st.caption(
            "Write each fact on its own line, using simple sentences. "
            "Do not combine facts with commas.\n\n"
            "Example:\n"
            "'My age is 55.\n"
            "My sex is male.\n"
            "My BMI is 28.5.\n"
            "I smoke.\n"
            "I have high blood pressure.\n"
            "My general health is fair.'"
        )
        
        user_text = st.text_area(
            "Your description:",
            height=120,
            placeholder="Type your health condition here...",
            key="llm_user_input"
        )
        
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_b:
            extract_clicked = st.button(
                "Extract Information",
                use_container_width=True,
                type="primary",
                key="extract_btn"
            )
        
        if extract_clicked and user_text.strip():
            with st.spinner("Extracting information..."):
                try:
                    provider = st.secrets.get("LLM_PROVIDER", "auto")
                    for p in ["gemini", "groq"]:
                        key = st.secrets.get(p.upper() + "_API_KEY", None)
                        if key:
                            os.environ[p.upper() + "_API_KEY"] = key
                except Exception:
                    provider = os.environ.get("LLM_PROVIDER", "auto")
                
                result = extract_features(user_text, provider=provider)
                
                if not result["success"]:
                    custom_alert(
                        "Extraction failed: " + result["error"] + ". "
                        "Please use Manual Entry instead.",
                        "error"
                    )
                    st.session_state.extracted_data = None
                    st.session_state.extraction_confirmed = False
                else:
                    st.session_state.extracted_data = result
                    st.session_state.extraction_confirmed = False
                    
                    if result["warnings"]:
                        custom_alert(
                            "Some fields could not be extracted: "
                            + "; ".join(result["warnings"])
                            + ". Please review and correct below.",
                            "warning"
                        )
                    else:
                        custom_alert(
                            "Information extracted using " + result["provider_used"] + ". "
                            "Please review and confirm below.",
                            "success"
                        )
        
        if st.session_state.extracted_data is not None:
            data = st.session_state.extracted_data["data"]
            
            st.markdown("---")
            st.markdown("### Step 1b -- Review Extracted Information")
            st.caption(
                "**The AI only extracts information -- it does NOT predict disease.** "
                "Review and correct any errors before confirming."
            )
            
            with st.form("confirmation_form"):
                st.markdown("**Demographics**")
                col1, col2 = st.columns(2)
                with col1:
                    age = st.slider("Age Category", 1, 13,
                                    value=int(data.get("age_category") or 7),
                                    key="c_age")
                    sex = st.radio("Sex", [0, 1],
                                   index=int(data.get("sex") or 0),
                                   format_func=lambda x: "Female" if x == 0 else "Male",
                                   horizontal=True, key="c_sex")
                with col2:
                    education = st.slider("Education Level", 1, 6,
                                          value=int(data.get("education") or 4),
                                          key="c_edu")
                    income = st.slider("Income Level", 1, 8,
                                       value=int(data.get("income") or 5),
                                       key="c_inc")
                
                st.markdown("**Health Status**")
                col1, col2 = st.columns(2)
                with col1:
                    gen_health = st.slider("General Health", 1, 5,
                                           value=int(data.get("gen_health") or 2),
                                           key="c_gen")
                    bmi = st.number_input("BMI", 10.0, 60.0,
                                          value=float(data.get("bmi") or 25.0),
                                          step=0.5, key="c_bmi")
                with col2:
                    ment_health = st.slider("Mental Health Days", 0, 30,
                                            value=int(data.get("ment_health") or 0),
                                            key="c_ment")
                    phys_health = st.slider("Physical Health Days", 0, 30,
                                            value=int(data.get("phys_health") or 0),
                                            key="c_phys")
                
                st.markdown("**Medical History**")
                col1, col2 = st.columns(2)
                with col1:
                    high_bp = st.radio("High BP", [0, 1],
                                       index=int(data.get("high_bp") or 0),
                                       format_func=lambda x: "No" if x == 0 else "Yes",
                                       horizontal=True, key="c_bp")
                    high_chol = st.radio("High Cholesterol", [0, 1],
                                         index=int(data.get("high_chol") or 0),
                                         format_func=lambda x: "No" if x == 0 else "Yes",
                                         horizontal=True, key="c_chol")
                    chol_check = st.radio("Cholesterol Check", [0, 1],
                                          index=int(data.get("chol_check") or 1),
                                          format_func=lambda x: "No" if x == 0 else "Yes",
                                          horizontal=True, key="c_chk")
                with col2:
                    stroke = st.radio("Stroke", [0, 1],
                                      index=int(data.get("stroke") or 0),
                                      format_func=lambda x: "No" if x == 0 else "Yes",
                                      horizontal=True, key="c_str")
                    diabetes = st.radio("Diabetes", [0, 1],
                                        index=int(data.get("diabetes") or 0),
                                        format_func=lambda x: "No" if x == 0 else "Yes",
                                        horizontal=True, key="c_dia")
                    diff_walk = st.radio("Difficulty Walking", [0, 1],
                                         index=int(data.get("diff_walk") or 0),
                                         format_func=lambda x: "No" if x == 0 else "Yes",
                                         horizontal=True, key="c_walk")
                
                st.markdown("**Lifestyle & Access**")
                col1, col2 = st.columns(2)
                with col1:
                    smoker = st.radio("Smoker", [0, 1],
                                      index=int(data.get("smoker") or 0),
                                      format_func=lambda x: "No" if x == 0 else "Yes",
                                      horizontal=True, key="c_smk")
                    phys_activity = st.radio("Physical Activity", [0, 1],
                                             index=int(data.get("phys_activity") or 1),
                                             format_func=lambda x: "No" if x == 0 else "Yes",
                                             horizontal=True, key="c_act")
                    fruits = st.radio("Fruits", [0, 1],
                                      index=int(data.get("fruits") or 1),
                                      format_func=lambda x: "No" if x == 0 else "Yes",
                                      horizontal=True, key="c_frt")
                with col2:
                    veggies = st.radio("Vegetables", [0, 1],
                                       index=int(data.get("veggies") or 1),
                                       format_func=lambda x: "No" if x == 0 else "Yes",
                                       horizontal=True, key="c_veg")
                    heavy_alcohol = st.radio("Heavy Alcohol", [0, 1],
                                             index=int(data.get("heavy_alcohol") or 0),
                                             format_func=lambda x: "No" if x == 0 else "Yes",
                                             horizontal=True, key="c_alc")
                    any_healthcare = st.radio("Healthcare Coverage", [0, 1],
                                              index=int(data.get("any_healthcare") or 1),
                                              format_func=lambda x: "No" if x == 0 else "Yes",
                                              horizontal=True, key="c_hc")
                no_doc_cost = st.radio("No Doctor Due to Cost", [0, 1],
                                       index=int(data.get("no_doc_cost") or 0),
                                       format_func=lambda x: "No" if x == 0 else "Yes",
                                       horizontal=True, key="c_cost")
                
                st.markdown("---")
                st.markdown("**By clicking Confirm, you verify the information above is correct.**")
                confirmed = st.form_submit_button(
                    "Confirm and Proceed",
                    type="primary",
                    use_container_width=True
                )
                
                if confirmed:
                    st.session_state.extraction_confirmed = True
                    st.session_state.confirmed_data = {
                        'HighBP': high_bp, 'HighChol': high_chol, 'CholCheck': chol_check,
                        'BMI': bmi, 'Smoker': smoker, 'Stroke': stroke, 'Diabetes': diabetes,
                        'PhysActivity': phys_activity, 'Fruits': fruits, 'Veggies': veggies,
                        'HvyAlcoholConsump': heavy_alcohol, 'AnyHealthcare': any_healthcare,
                        'NoDocbcCost': no_doc_cost, 'GenHlth': gen_health, 'MentHlth': ment_health,
                        'PhysHlth': phys_health, 'DiffWalk': diff_walk, 'Sex': sex, 'Age': age,
                        'Education': education, 'Income': income
                    }
                    custom_alert(
                        "Information confirmed. Proceed to Step 2 to generate prediction.",
                        "success"
                    )


else:
    st.markdown("### Enter patient information manually")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Demographics", "Health Status", "Medical History", "Lifestyle & Access"]
    )
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age Category", 1, 13, 7)
            sex = st.radio("Sex", [0, 1],
                           format_func=lambda x: "Female" if x == 0 else "Male",
                           horizontal=True)
        with col2:
            education = st.slider("Education Level", 1, 6, 4)
            income = st.slider("Income Level", 1, 8, 5)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            gen_health = st.slider("General Health", 1, 5, 2)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.5)
        with col2:
            ment_health = st.slider("Mental Health Days", 0, 30, 0)
            phys_health = st.slider("Physical Health Days", 0, 30, 0)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            high_bp = st.radio("High BP", [0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes",
                               horizontal=True)
            high_chol = st.radio("High Cholesterol", [0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes",
                                 horizontal=True)
            chol_check = st.radio("Cholesterol Check", [0, 1],
                                  format_func=lambda x: "No" if x == 0 else "Yes",
                                  horizontal=True)
        with col2:
            stroke = st.radio("Stroke", [0, 1],
                              format_func=lambda x: "No" if x == 0 else "Yes",
                              horizontal=True)
            diabetes = st.radio("Diabetes", [0, 1],
                                format_func=lambda x: "No" if x == 0 else "Yes",
                                horizontal=True)
            diff_walk = st.radio("Difficulty Walking", [0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes",
                                 horizontal=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            smoker = st.radio("Smoker", [0, 1],
                              format_func=lambda x: "No" if x == 0 else "Yes",
                              horizontal=True)
            phys_activity = st.radio("Physical Activity", [0, 1],
                                     format_func=lambda x: "No" if x == 0 else "Yes",
                                     horizontal=True)
            fruits = st.radio("Fruits", [0, 1],
                              format_func=lambda x: "No" if x == 0 else "Yes",
                              horizontal=True)
        with col2:
            veggies = st.radio("Vegetables", [0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes",
                               horizontal=True)
            heavy_alcohol = st.radio("Heavy Alcohol", [0, 1],
                                     format_func=lambda x: "No" if x == 0 else "Yes",
                                     horizontal=True)
            any_healthcare = st.radio("Healthcare Coverage", [0, 1],
                                      format_func=lambda x: "No" if x == 0 else "Yes",
                                      horizontal=True)
        no_doc_cost = st.radio("No Doctor Due to Cost", [0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes",
                               horizontal=True)


def _plot_shap_bar(shap_values, patient_values, feature_names, base_value, prediction, model_name, ax):
    import numpy as np
    indices = np.argsort(np.abs(shap_values))[::-1][:12]
    sorted_shap = shap_values[indices]
    sorted_features = [feature_names[i] for i in indices]
    sorted_values = [patient_values[i] for i in indices]
    colors = ['#E74C3C' if v > 0 else '#2E86AB' for v in sorted_shap]
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_shap, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{f} = {v}" for f, v in zip(sorted_features, sorted_values)], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('SHAP value (impact on prediction)', fontsize=10)
    ax.set_title(f'{model_name} Feature Contributions\nBase value = {base_value:.3f} | Prediction = {prediction:.3f}', fontsize=11)
    for i, v in enumerate(sorted_shap):
        offset = 0.001 if v >= 0 else -0.001
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + offset, i, f'{v:+.4f}', va='center', ha=ha, fontsize=8)
    ax.grid(axis='x', alpha=0.3)


def build_patient_data():
    if (st.session_state.get('extraction_confirmed') and 
        st.session_state.get('confirmed_data') is not None):
        return st.session_state.confirmed_data
    else:
        return {
            'HighBP': high_bp, 'HighChol': high_chol, 'CholCheck': chol_check,
            'BMI': bmi, 'Smoker': smoker, 'Stroke': stroke, 'Diabetes': diabetes,
            'PhysActivity': phys_activity, 'Fruits': fruits, 'Veggies': veggies,
            'HvyAlcoholConsump': heavy_alcohol, 'AnyHealthcare': any_healthcare,
            'NoDocbcCost': no_doc_cost, 'GenHlth': gen_health, 'MentHlth': ment_health,
            'PhysHlth': phys_health, 'DiffWalk': diff_walk, 'Sex': sex, 'Age': age,
            'Education': education, 'Income': income
        }


st.markdown("---")
st.header("Step 2 -- Predict Risk")

if input_mode == "Describe Symptoms (AI-Assisted)" and not st.session_state.get("extraction_confirmed", False):
    custom_alert(
        "Please extract and confirm the patient information in Step 1b first.",
        "warning"
    )
    predict_clicked = False
else:
    center1, center2, center3 = st.columns([1, 2, 1])
    with center2:
        predict_clicked = st.button(
            "Predict Heart Disease Risk",
            use_container_width=True,
            type="primary"
        )


if predict_clicked:
    patient_dict = build_patient_data()
    patient_df = pd.DataFrame([patient_dict])[feature_names]
    
    cat_proba = cat_calibrated.predict_proba(patient_df)[0, 1]
    xgb_proba = xgb_calibrated.predict_proba(patient_df)[0, 1]
    avg_proba = (cat_proba + xgb_proba) / 2
    
    shap_values_cat = explainer_cat.shap_values(patient_df)[0]
    shap_values_xgb = explainer_xgb.shap_values(patient_df)[0]
    
    st.session_state.predictions = {
        'patient_dict': patient_dict,
        'patient_df': patient_df,
        'cat_proba': cat_proba,
        'xgb_proba': xgb_proba,
        'avg_proba': avg_proba,
        'shap_values_cat': shap_values_cat,
        'shap_values_xgb': shap_values_xgb,
        'base_value_cat': explainer_cat.expected_value,
        'base_value_xgb': explainer_xgb.expected_value,
    }
    st.session_state.pdf_bytes = None


# ============================================================
# RESULTS
# ============================================================
if st.session_state.predictions is not None:
    p = st.session_state.predictions
    cat_proba = p['cat_proba']
    xgb_proba = p['xgb_proba']
    avg_proba = p['avg_proba']
    
    st.markdown("---")
    st.header("Step 3 -- Results")
    
    # Risk category
    if avg_proba < 0.05:
        risk_category = "Very Low Risk"; risk_color = "#27AE60"
        custom_alert(f"<b>{risk_category}</b> -- {avg_proba:.1%}", "success")
    elif avg_proba < 0.10:
        risk_category = "Low Risk"; risk_color = "#2ECC71"
        custom_alert(f"<b>{risk_category}</b> -- {avg_proba:.1%}", "success")
    elif avg_proba < 0.20:
        risk_category = "Moderate Risk"; risk_color = "#F1C40F"
        custom_alert(f"<b>{risk_category}</b> -- {avg_proba:.1%}", "warning")
    elif avg_proba < 0.35:
        risk_category = "High Risk"; risk_color = "#E67E22"
        custom_alert(f"<b>{risk_category}</b> -- {avg_proba:.1%}", "warning")
    else:
        risk_category = "Very High Risk"; risk_color = "#E74C3C"
        custom_alert(f"<b>{risk_category}</b> -- {avg_proba:.1%}", "error")
    
    # ============================================================
    # PDF BUTTON — INDEPENDENT OF PREDICT
    # ============================================================
    pdf_col1, pdf_col2, pdf_col3 = st.columns([1, 2, 1])
    with pdf_col2:
        if st.button("Generate PDF Report", use_container_width=True, type="primary", key="pdf_generate_btn"):
            with st.spinner("Generating PDF report..."):
                os.makedirs('reports', exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                pdf_path = f'reports/heart_disease_report_{timestamp}.pdf'
                
                generate_pdf_report(
                    patient_data=p['patient_dict'],
                    catboost_proba=p['cat_proba'],
                    xgboost_proba=p['xgb_proba'],
                    catboost_shap_values=p['shap_values_cat'],
                    feature_names=feature_names,
                    catboost_base_value=p['base_value_cat'],
                    optimal_threshold=0.5,
                    output_path=pdf_path
                )
                
                with open(pdf_path, 'rb') as f:
                    st.session_state.pdf_bytes = f.read()
                st.session_state.pdf_filename = f'heart_disease_report_{timestamp}.pdf'
                
                custom_alert("PDF report generated successfully!", "success")
    
    # Download button (persistent via session state)
    if st.session_state.pdf_bytes is not None:
        st.download_button(
            label="Download PDF Report",
            data=st.session_state.pdf_bytes,
            file_name=st.session_state.pdf_filename,
            mime='application/pdf',
            use_container_width=True,
            key="pdf_download_btn"
        )
    
    # Metric cards
    st.markdown("### Risk Probabilities")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card" style="background-color: #2E86AB; color: white;"><h3>CatBoost</h3><h2>{cat_proba:.1%}</h2></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card" style="background-color: #A23B72; color: white;"><h3>XGBoost</h3><h2>{xgb_proba:.1%}</h2></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card" style="background-color: {risk_color}; color: white;"><h3>Average</h3><h2>{avg_proba:.1%}</h2></div>', unsafe_allow_html=True)
    
    # Risk Gauge
    st.markdown("### Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_proba * 100,
        title={'text': "Overall Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 5], 'color': '#D5F5E3'},
                {'range': [5, 10], 'color': '#ABEBC6'},
                {'range': [10, 20], 'color': '#F9E79F'},
                {'range': [20, 35], 'color': '#F5CBA7'},
                {'range': [35, 100], 'color': '#F5B7B1'}
            ]
        }
    ))
    fig_gauge.update_layout(height=350)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Model Comparison
    st.markdown("### Model Comparison")
    fig_compare = go.Figure(data=[
        go.Bar(x=['CatBoost', 'XGBoost', 'Average'],
               y=[cat_proba * 100, xgb_proba * 100, avg_proba * 100],
               marker_color=['#2E86AB', '#A23B72', risk_color],
               text=[f'{cat_proba:.1%}', f'{xgb_proba:.1%}', f'{avg_proba:.1%}'],
               textposition='outside')
    ])
    fig_compare.update_layout(height=400, yaxis_title="Risk (%)")
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # SHAP Waterfall — CatBoost
    st.markdown("### Feature Contributions -- CatBoost (SHAP)")
    fig_cat, ax_cat = plt.subplots(figsize=(10, 7))
    _plot_shap_bar(p['shap_values_cat'], p['patient_df'].iloc[0].values,
                   feature_names, p['base_value_cat'], p['cat_proba'],
                   "CatBoost", ax_cat)
    plt.tight_layout()
    st.pyplot(fig_cat)
    plt.close(fig_cat)
    
    # SHAP Waterfall — XGBoost
    st.markdown("### Feature Contributions -- XGBoost (SHAP)")
    fig_xgb, ax_xgb = plt.subplots(figsize=(10, 7))
    _plot_shap_bar(p['shap_values_xgb'], p['patient_df'].iloc[0].values,
                   feature_names, p['base_value_xgb'], p['xgb_proba'],
                   "XGBoost", ax_xgb)
    plt.tight_layout()
    st.pyplot(fig_xgb)
    plt.close(fig_xgb)
    
    # Side-by-Side SHAP
    st.markdown("### Top 10 Feature Contributions -- Side-by-Side")
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'CatBoost': p['shap_values_cat'],
        'XGBoost': p['shap_values_xgb']
    })
    shap_df['MaxAbs'] = shap_df[['CatBoost', 'XGBoost']].abs().max(axis=1)
    shap_df = shap_df.sort_values('MaxAbs', ascending=False).head(10)
    
    fig_compare_shap = go.Figure(data=[
        go.Bar(name='CatBoost', y=shap_df['Feature'], x=shap_df['CatBoost'],
               orientation='h', marker_color='#2E86AB'),
        go.Bar(name='XGBoost', y=shap_df['Feature'], x=shap_df['XGBoost'],
               orientation='h', marker_color='#A23B72')
    ])
    fig_compare_shap.update_layout(barmode='group', height=500,
                                     xaxis_title="SHAP Value",
                                     yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_compare_shap, use_container_width=True)
    
    # Radar Chart
    st.markdown("### Patient Profile Radar")
    top_features = shap_df.head(8)['Feature'].tolist()
    X_train_stats = pd.read_csv('data/X_train_final.csv')
    patient_values = [p['patient_dict'][f] for f in top_features]
    population_means = [X_train_stats[f].mean() for f in top_features]
    population_max = [X_train_stats[f].max() for f in top_features]
    patient_norm = [(v / m * 100) if m > 0 else 0 for v, m in zip(patient_values, population_max)]
    pop_norm = [(m / mx * 100) if mx > 0 else 0 for m, mx in zip(population_means, population_max)]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=patient_norm + [patient_norm[0]],
        theta=top_features + [top_features[0]],
        fill='toself', name='Patient', line_color='#2E86AB'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=pop_norm + [pop_norm[0]],
        theta=top_features + [top_features[0]],
        fill='toself', name='Population Average',
        line_color='#A23B72', opacity=0.5
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                              height=500, showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("---")
    st.caption(
        "WARNING: This tool is a research prototype trained on the BRFSS 2015 dataset. "
        "It has not been externally validated on independent clinical cohorts. "
        "Predictions should NOT be used for clinical decision-making."
    )
