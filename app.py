
"""
app.py
Interactive Streamlit application for heart disease risk prediction.
Features: beating heart heading, main-page input, dual visualizations, prominent PDF export, theme toggle.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

from pdf_report import generate_pdf_report

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
# CUSTOM CSS FOR INTERACTIVITY + THEME
# ============================================================
def apply_custom_css():
    st.markdown("""
    <style>
    /* Beating heart animation */
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
    
    /* Metric card styling */
    .metric-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    /* Big prominent PDF button */
    .pdf-button-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    
    /* Risk category badges */
    .risk-badge {
        padding: 15px 30px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 20px 0;
    }
    
    /* Interactive button hover */
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


apply_custom_css()

# ============================================================
# THEME TOGGLE
# ============================================================
theme_options = {
    "Dark Mode": "dark",
    "Light Mode": "light",
    "System Default": "auto"
}

theme_col1, theme_col2 = st.columns([4, 1])
with theme_col2:
    selected_theme = st.selectbox(
        "Theme",
        options=list(theme_options.keys()),
        index=2,
        label_visibility="collapsed"
    )

# Apply theme via HTML injection
if selected_theme == "Dark Mode":
    st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stMarkdown, .stText { color: #FAFAFA; }
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
# System Default -- Streamlit handles automatically


# ============================================================
# LOAD MODELS (CACHED)
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
    background_idx = np.random.choice(len(X_train), size=1000, replace=False)
    X_background = X_train.iloc[background_idx]
    
    explainer_cat = shap.TreeExplainer(cat_base, data=X_background, feature_perturbation='interventional')
    explainer_xgb = shap.TreeExplainer(xgb_base, data=X_background, feature_perturbation='interventional')
    return explainer_cat, explainer_xgb


cat_calibrated, xgb_calibrated, cat_base, xgb_base, feature_names = load_models()
explainer_cat, explainer_xgb = load_shap_explainers()


# ============================================================
# HEADING WITH BEATING HEART
# ============================================================
st.markdown(
    '<h1><span class="beating-heart">&#10084;&#65039;</span>Heart Disease Risk Prediction</h1>',
    unsafe_allow_html=True
)

st.markdown("""
This interactive tool uses **CatBoost** and **XGBoost** machine learning models 
to estimate heart disease risk based on demographic, lifestyle, and health factors.
""")

st.warning("Research prototype only -- NOT for clinical decision-making.")

st.markdown("---")

# ============================================================
# SECTION 1: PATIENT INPUT ON MAIN PAGE
# ============================================================
st.header("Step 1 -- Enter Patient Information")

# Use tabs for organized input
tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Health Status", "Medical History", "Lifestyle & Access"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age Category", 1, 13, 7, 
                        help="1=18-24, 2=25-29, ..., 13=80+")
        sex = st.radio("Sex", options=[0, 1], 
                       format_func=lambda x: "Female" if x == 0 else "Male",
                       horizontal=True)
    with col2:
        education = st.slider("Education Level", 1, 6, 4,
                              help="1=None, 2=Elementary, 3=Some HS, 4=HS Grad, 5=Some College, 6=College Grad")
        income = st.slider("Income Level", 1, 8, 5,
                           help="1=<$10k, ..., 8=>$75k")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        gen_health = st.slider("General Health", 1, 5, 2,
                               help="1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor")
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.5)
    with col2:
        ment_health = st.slider("Poor Mental Health Days (past 30)", 0, 30, 0)
        phys_health = st.slider("Poor Physical Health Days (past 30)", 0, 30, 0)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        high_bp = st.radio("High Blood Pressure", [0, 1], 
                           format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
        high_chol = st.radio("High Cholesterol", [0, 1], 
                             format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
        chol_check = st.radio("Cholesterol Check (past 5 yrs)", [0, 1], 
                              format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    with col2:
        stroke = st.radio("Ever Had Stroke", [0, 1], 
                          format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
        diabetes = st.radio("Diabetes", [0, 1], 
                            format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
        diff_walk = st.radio("Difficulty Walking", [0, 1], 
                             format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        smoker = st.radio("Smoker (100+ cigarettes)", [0, 1], 
                          format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
        phys_activity = st.radio("Physical Activity", [0, 1], 
                                 format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
        fruits = st.radio("Consume Fruits Daily", [0, 1], 
                          format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    with col2:
        veggies = st.radio("Consume Vegetables Daily", [0, 1], 
                           format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
        heavy_alcohol = st.radio("Heavy Alcohol Consumption", [0, 1], 
                                 format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
        any_healthcare = st.radio("Has Healthcare Coverage", [0, 1], 
                                  format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    no_doc_cost = st.radio("Could Not See Doctor Due to Cost", [0, 1], 
                            format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)

# ============================================================
# STEP 2 BUTTON
# ============================================================
st.markdown("---")
st.header("Step 2 -- Predict Risk")

center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
with center_col2:
    predict_clicked = st.button("Predict Heart Disease Risk", use_container_width=True, type="primary")


# ============================================================
# BUILD PATIENT DATA
# ============================================================
def build_patient_data():
    return {
        'HighBP': high_bp,
        'HighChol': high_chol,
        'CholCheck': chol_check,
        'BMI': bmi,
        'Smoker': smoker,
        'Stroke': stroke,
        'Diabetes': diabetes,
        'PhysActivity': phys_activity,
        'Fruits': fruits,
        'Veggies': veggies,
        'HvyAlcoholConsump': heavy_alcohol,
        'AnyHealthcare': any_healthcare,
        'NoDocbcCost': no_doc_cost,
        'GenHlth': gen_health,
        'MentHlth': ment_health,
        'PhysHlth': phys_health,
        'DiffWalk': diff_walk,
        'Sex': sex,
        'Age': age,
        'Education': education,
        'Income': income
    }


# ============================================================
# STEP 3 -- RESULTS
# ============================================================
if predict_clicked:
    patient_dict = build_patient_data()
    patient_df = pd.DataFrame([patient_dict])[feature_names]
    
    cat_proba = cat_calibrated.predict_proba(patient_df)[0, 1]
    xgb_proba = xgb_calibrated.predict_proba(patient_df)[0, 1]
    avg_proba = (cat_proba + xgb_proba) / 2
    
    shap_values_cat = explainer_cat.shap_values(patient_df)[0]
    shap_values_xgb = explainer_xgb.shap_values(patient_df)[0]
    base_value_cat = explainer_cat.expected_value
    base_value_xgb = explainer_xgb.expected_value
    
    st.markdown("---")
    st.header("Step 3 -- Results")
    
    # Risk category
    if avg_proba < 0.05:
        risk_category = "Very Low Risk"
        risk_color = "#27AE60"
        st.success(f"### {risk_category} -- {avg_proba:.1%}")
    elif avg_proba < 0.10:
        risk_category = "Low Risk"
        risk_color = "#2ECC71"
        st.success(f"### {risk_category} -- {avg_proba:.1%}")
    elif avg_proba < 0.20:
        risk_category = "Moderate Risk"
        risk_color = "#F1C40F"
        st.warning(f"### {risk_category} -- {avg_proba:.1%}")
    elif avg_proba < 0.35:
        risk_category = "High Risk"
        risk_color = "#E67E22"
        st.warning(f"### {risk_category} -- {avg_proba:.1%}")
    else:
        risk_category = "Very High Risk"
        risk_color = "#E74C3C"
        st.error(f"### {risk_category} -- {avg_proba:.1%}")
    
    # ============================================================
    # PDF BUTTON (PROMINENT, RIGHT AFTER RESULTS)
    # ============================================================
    pdf_col1, pdf_col2, pdf_col3 = st.columns([1, 2, 1])
    with pdf_col2:
        generate_pdf_clicked = st.button(
            "Generate PDF Report",
            use_container_width=True,
            type="primary",
            key="pdf_button_top"
        )
    
    # ============================================================
    # METRIC CARDS
    # ============================================================
    st.markdown("### Risk Probabilities")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="background-color: #2E86AB; color: white;">
            <h3>CatBoost</h3>
            <h2>{cat_proba:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="background-color: #A23B72; color: white;">
            <h3>XGBoost</h3>
            <h2>{xgb_proba:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="background-color: {risk_color}; color: white;">
            <h3>Average</h3>
            <h2>{avg_proba:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================
    # VISUALIZATION 1: RISK GAUGE (Plotly)
    # ============================================================
    st.markdown("### Risk Gauge")
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_proba * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Risk (%)", 'font': {'size': 20}},
        delta={'reference': 10, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 5], 'color': '#D5F5E3'},
                {'range': [5, 10], 'color': '#ABEBC6'},
                {'range': [10, 20], 'color': '#F9E79F'},
                {'range': [20, 35], 'color': '#F5CBA7'},
                {'range': [35, 100], 'color': '#F5B7B1'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        }
    ))
    
    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # ============================================================
    # VISUALIZATION 2: MODEL COMPARISON BAR CHART
    # ============================================================
    st.markdown("### Model Comparison")
    
    fig_compare = go.Figure(data=[
        go.Bar(
            x=['CatBoost', 'XGBoost', 'Average'],
            y=[cat_proba * 100, xgb_proba * 100, avg_proba * 100],
            marker_color=['#2E86AB', '#A23B72', risk_color],
            text=[f'{cat_proba:.1%}', f'{xgb_proba:.1%}', f'{avg_proba:.1%}'],
            textposition='outside'
        )
    ])
    
    fig_compare.update_layout(
        height=400,
        yaxis_title="Risk Probability (%)",
        yaxis_range=[0, max(100, max(cat_proba, xgb_proba, avg_proba) * 110)],
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # ============================================================
    # VISUALIZATION 3: SHAP WATERFALL -- CatBoost
    # ============================================================
    st.markdown("### Feature Contributions -- CatBoost (SHAP)")
    
    fig_cat, ax_cat = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_cat,
            base_values=base_value_cat,
            data=patient_df.iloc[0].values,
            feature_names=feature_names
        ),
        max_display=12,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig_cat)
    plt.close(fig_cat)
    
    # ============================================================
    # VISUALIZATION 4: SHAP WATERFALL -- XGBoost
    # ============================================================
    st.markdown("### Feature Contributions -- XGBoost (SHAP)")
    
    fig_xgb, ax_xgb = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_xgb,
            base_values=base_value_xgb,
            data=patient_df.iloc[0].values,
            feature_names=feature_names
        ),
        max_display=12,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig_xgb)
    plt.close(fig_xgb)
    
    # ============================================================
    # VISUALIZATION 5: TOP FEATURE COMPARISON (CatBoost vs XGBoost)
    # ============================================================
    st.markdown("### Top 10 Feature Contributions -- Side-by-Side Comparison")
    
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'CatBoost': shap_values_cat,
        'XGBoost': shap_values_xgb
    })
    shap_df['Abs_Cat'] = np.abs(shap_df['CatBoost'])
    shap_df['Abs_XGB'] = np.abs(shap_df['XGBoost'])
    shap_df['Max_Abs'] = shap_df[['Abs_Cat', 'Abs_XGB']].max(axis=1)
    shap_df = shap_df.sort_values('Max_Abs', ascending=False).head(10)
    
    fig_compare_shap = go.Figure(data=[
        go.Bar(
            name='CatBoost',
            y=shap_df['Feature'],
            x=shap_df['CatBoost'],
            orientation='h',
            marker_color='#2E86AB'
        ),
        go.Bar(
            name='XGBoost',
            y=shap_df['Feature'],
            x=shap_df['XGBoost'],
            orientation='h',
            marker_color='#A23B72'
        )
    ])
    
    fig_compare_shap.update_layout(
        barmode='group',
        height=500,
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis_title="Feature",
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_compare_shap, use_container_width=True)
    
    # ============================================================
    # VISUALIZATION 6: FEATURE RADAR (Interactive)
    # ============================================================
    st.markdown("### Patient Profile Radar")
    
    # Select top 8 features by absolute SHAP
    top_features = shap_df.head(8)['Feature'].tolist()
    
    # Get patient values and population means
    X_train_for_stats = pd.read_csv('data/X_train_final.csv')
    patient_values = [patient_dict[f] for f in top_features]
    population_means = [X_train_for_stats[f].mean() for f in top_features]
    population_max = [X_train_for_stats[f].max() for f in top_features]
    
    # Normalize to 0-100 scale
    patient_normalized = [(v / m * 100) if m > 0 else 0 
                          for v, m in zip(patient_values, population_max)]
    population_normalized = [(m / mx * 100) if mx > 0 else 0 
                              for m, mx in zip(population_means, population_max)]
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=patient_normalized + [patient_normalized[0]],
        theta=top_features + [top_features[0]],
        fill='toself',
        name='Patient',
        line_color='#2E86AB'
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=population_normalized + [population_normalized[0]],
        theta=top_features + [top_features[0]],
        fill='toself',
        name='Population Average',
        line_color='#A23B72',
        opacity=0.5
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=500,
        margin=dict(l=60, r=60, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # ============================================================
    # PDF GENERATION (HANDLE CLICK)
    # ============================================================
    if generate_pdf_clicked:
        with st.spinner("Generating PDF report..."):
            os.makedirs('reports', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_path = f'reports/heart_disease_report_{timestamp}.pdf'
            
            generate_pdf_report(
                patient_data=patient_dict,
                catboost_proba=cat_proba,
                xgboost_proba=xgb_proba,
                catboost_shap_values=shap_values_cat,
                feature_names=feature_names,
                catboost_base_value=base_value_cat,
                optimal_threshold=0.5,
                output_path=pdf_path
            )
            
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            st.success("PDF report generated successfully!")
            
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f'heart_disease_report_{timestamp}.pdf',
                mime='application/pdf',
                use_container_width=True,
                key="pdf_download"
            )
    
    # ============================================================
    # DISCLAIMER
    # ============================================================
    st.markdown("---")
    st.caption(
        "WARNING: This tool is a research prototype trained on the BRFSS 2015 dataset. "
        "It has not been externally validated on independent clinical cohorts. "
        "Predictions should NOT be used for clinical decision-making. "
        "Always consult qualified healthcare professionals."
    )
