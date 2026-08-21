import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import datetime
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import settings
from core.auth_manager import AuthManager
from core.ml_engine import MLEngine
from core.pdf_generator import PDFReportGenerator
from ui.components import render_header, render_risk_gauge

st.set_page_config(
    page_title="Metabolic Intelligence Diagnostic Suite",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    with open("ui/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

AuthManager.initialize_session()

# Default patient data state
if "p_data" not in st.session_state:
    st.session_state.p_data = {
        "gender": "Female",
        "age": 54.0,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "diet_quality": "Standard Western",
        "physical_activity": "Moderate (150-300 min/wk)",
        "sleep_quality": "Optimal (7-9 hrs)",
        "stress_level": "Moderate",
        "bmi": 27.32,
        "HbA1c_level": 6.6,
        "blood_glucose_level": 140
    }

# =============================================================================
# 1. AUTHENTICATION PORTAL (USER & DOCTOR SIGN IN / SIGN UP)
# =============================================================================
if not st.session_state.authenticated:
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; margin-bottom: 25px;">
        <h1 style="font-size: 2.8rem; margin-bottom: 4px;">
            <span class="gemini-gradient-text">🧬 Metabolic Intelligence Portal</span>
        </h1>
        <p style="color: #9aa0a6; font-size: 1.05rem;">
            Personalized Diabetes Risk Analytics • Patient Self-Check & Clinical Portal
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l1, col_l2, col_l3 = st.columns([1, 1.4, 1])
    with col_l2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["🔑 Sign In", "📝 Create Account"])

        # --- SIGN IN TAB ---
        with tab_login:
            st.markdown("#### Access Your Portal")
            selected_role = st.radio("I am signing in as:", ["Patient / User", "Doctor / Clinician"], horizontal=True)
            
            default_email = "user@health.com" if selected_role == "Patient / User" else "doctor@hospital.com"
            default_pass = "User@123" if selected_role == "Patient / User" else "Doctor@123"

            login_email = st.text_input("Email Address", value=default_email, key="login_email")
            login_pass = st.text_input("Password", type="password", value=default_pass, key="login_pass")

            if st.button("🚀 Sign In to Dashboard", use_container_width=True, type="primary"):
                if AuthManager.login(login_email, login_pass, selected_role):
                    st.success("✅ Signed in successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Invalid credentials for {selected_role}.")

            st.caption(f"Demo credentials: `{default_email}` | Password: `{default_pass}`")

        # --- SIGN UP TAB ---
        with tab_register:
            st.markdown("#### Register New Account")
            reg_role = st.radio("Account Type:", ["Patient / User", "Doctor / Clinician"], horizontal=True, key="reg_role")
            
            reg_name = st.text_input("Full Name", placeholder="e.g., Jane Doe")
            reg_email = st.text_input("Email Address", placeholder="jane.doe@example.com", key="reg_email")
            reg_pass = st.text_input("Create Password", type="password", key="reg_pass")

            extra_fields = {}
            if reg_role == "Patient / User":
                col_u1, col_u2 = st.columns(2)
                with col_u1:
                    extra_fields["age"] = st.number_input("Age", 1.0, 100.0, 25.0)
                with col_u2:
                    extra_fields["gender"] = st.selectbox("Gender", ["Female", "Male", "Other"])
                extra_fields["patient_id"] = f"PX-{np.random.randint(10000, 99999)}"
            else:
                extra_fields["license"] = st.text_input("Medical License No.", placeholder="REG-MCI-123456")
                extra_fields["department"] = st.text_input("Hospital / Department", value="Endocrinology & Diabetes Care")

            if st.button("Create Account", use_container_width=True):
                if reg_name and reg_email and reg_pass:
                    if AuthManager.register(reg_email, reg_pass, reg_name, reg_role, extra_fields):
                        st.success("✅ Account registered! You can now switch to the 'Sign In' tab.")
                    else:
                        st.error("⚠️ An account with this email already exists.")
                else:
                    st.warning("⚠️ Please fill in all required fields.")

        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# =============================================================================
# 2. AUTHENTICATED DASHBOARD (USER / DOCTOR SPECIFIC VIEWS)
# =============================================================================
user = st.session_state.user_info
is_doctor = user["role"] == "Doctor / Clinician"

# Synchronize patient metadata
patient_id = user.get("patient_id", "PX-90412")
if not is_doctor:
    st.session_state.p_data["gender"] = user.get("gender", st.session_state.p_data["gender"])
    st.session_state.p_data["age"] = user.get("age", st.session_state.p_data["age"])

# Sidebar Profile Badge
st.sidebar.markdown(f"""
<div style="background: rgba(255,255,255,0.05); padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 15px;">
    <div style="font-size: 0.75rem; color: #9aa0a6; text-transform: uppercase;">{user['role']}</div>
    <div style="font-weight: 700; color: #38bdf8; font-size: 1.05rem;">{user['name']}</div>
    <div style="font-size: 0.8rem; color: #cbd5e1;">{'License: ' + user.get('license', '') if is_doctor else 'Patient ID: ' + patient_id}</div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🚪 Sign Out", use_container_width=True):
    AuthManager.logout()

# Cache ML Pipelines
@st.cache_resource
def load_ml_assets():
    clf_df, trained_clfs, y_test_c, y_preds_c, y_probs_c = MLEngine.train_classification_models()
    reg_df, trained_regs, y_test_r, y_preds_r = MLEngine.train_regression_models()
    
    rf_row = clf_df[clf_df["Model"].str.contains("Random Forest")].iloc[0]
    log_row = clf_df[clf_df["Model"] == "Logistic Regression"].iloc[0]
    reg_row = reg_df.iloc[0]

    ml_metrics_summary = {
        "rf_acc": rf_row["Accuracy"],
        "rf_f1": rf_row["F1-Score"],
        "rf_prec": rf_row["Precision"],
        "rf_rec": rf_row["Recall"],
        "log_acc": log_row["Accuracy"],
        "log_f1": log_row["F1-Score"],
        "log_prec": log_row["Precision"],
        "log_rec": log_row["Recall"],
        "r2": reg_row["R-Squared (R²)"],
        "rmse": reg_row["RMSE (mg/dL)"],
        "mae": reg_row["MAE (mg/dL)"]
    }

    return {
        "clf_df": clf_df,
        "trained_clfs": trained_clfs,
        "y_test_c": y_test_c,
        "y_preds_c": y_preds_c,
        "y_probs_c": y_probs_c,
        "reg_df": reg_df,
        "trained_regs": trained_regs,
        "y_test_r": y_test_r,
        "y_preds_r": y_preds_r,
        "metrics_summary": ml_metrics_summary
    }

ml_assets = load_ml_assets()

render_header(patient_id)

nav_options = [
    "🩺 Live Health Assessment & Prediction",
    "📊 ML Model Benchmarks (F1, Accuracy, RMSE, R²)",
    "🔮 What-If Lifestyle Simulator",
    "📄 Diagnostic Health Report (PDF)"
]
nav_choice = st.sidebar.radio("Navigation Menu", nav_options)

def run_classification(p_dict, model_name="Random Forest (Balanced)"):
    pipe = ml_assets["trained_clfs"][model_name]
    df_inf = pd.DataFrame([p_dict])[settings.CATEGORICAL_COLS + settings.NUMERICAL_COLS]
    probs = pipe.predict_proba(df_inf)[0]
    prob = (probs[1] if len(probs) > 1 else probs[0]) * 100
    prob_val = min(max(round(prob, 1), 1.0), 99.0)

    if prob_val < 35:
        tier = "Low Clinical Risk"
    elif prob_val < 65:
        tier = "Moderate Risk (Prediabetic Pattern)"
    else:
        tier = "High Clinical Risk"
    return prob_val, tier

def run_regression(p_dict, model_name="Ridge Regression"):
    pipe = ml_assets["trained_regs"][model_name]
    df_inf = pd.DataFrame([p_dict])[settings.CATEGORICAL_COLS + settings.REGRESSION_NUM_COLS]
    return round(float(pipe.predict(df_inf)[0]), 1)

# -----------------------------------------------------------------------------
# TAB 1: LIVE ASSESSMENT
# -----------------------------------------------------------------------------
if nav_choice == "🩺 Live Health Assessment & Prediction":
    c1, c2 = st.columns([1.1, 1.2], gap="large")

    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📋 **Health Metrics & Laboratory Profile**")
        tab_bio, tab_life = st.tabs(["🔬 Laboratory Biomarkers", "🥗 Diet & Daily Routine"])

        with tab_bio:
            f1, f2 = st.columns(2)
            with f1:
                st.session_state.p_data["gender"] = st.selectbox("Gender", ["Female", "Male", "Other"], index=0 if st.session_state.p_data["gender"] == "Female" else 1)
                st.session_state.p_data["age"] = st.number_input("Age (Years)", 1.0, 105.0, float(st.session_state.p_data["age"]), 1.0)
                st.session_state.p_data["hypertension"] = st.selectbox("Hypertension Status", [0, 1], index=int(st.session_state.p_data["hypertension"]), format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                st.session_state.p_data["heart_disease"] = st.selectbox("Heart Disease History", [0, 1], index=int(st.session_state.p_data["heart_disease"]), format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            with f2:
                st.session_state.p_data["bmi"] = st.number_input("Body Mass Index (BMI)", 10.0, 90.0, float(st.session_state.p_data["bmi"]), 0.1)
                st.session_state.p_data["HbA1c_level"] = st.number_input("Serum HbA1c Level (%)", 3.5, 12.0, float(st.session_state.p_data["HbA1c_level"]), 0.1)
                st.session_state.p_data["blood_glucose_level"] = st.number_input("Fasting Glucose (mg/dL)", 70, 300, int(st.session_state.p_data["blood_glucose_level"]), 5)
                st.session_state.p_data["smoking_history"] = st.selectbox("Smoking History", ["never", "former", "current", "No Info"], index=0)

        with tab_life:
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                st.session_state.p_data["diet_quality"] = st.selectbox(
                    "Dietary Framework", 
                    ["Mediterranean", "Balanced Whole-Food", "Standard Western", "High Carbohydrate / Processed"],
                    index=["Mediterranean", "Balanced Whole-Food", "Standard Western", "High Carbohydrate / Processed"].index(st.session_state.p_data["diet_quality"])
                )
                st.session_state.p_data["physical_activity"] = st.selectbox(
                    "Physical Activity & Exercise", 
                    ["High (300+ min/wk)", "Moderate (150-300 min/wk)", "Light (30-149 min/wk)", "Sedentary (<30 min/wk)"],
                    index=["High (300+ min/wk)", "Moderate (150-300 min/wk)", "Light (30-149 min/wk)", "Sedentary (<30 min/wk)"].index(st.session_state.p_data["physical_activity"])
                )
            with l_col2:
                st.session_state.p_data["sleep_quality"] = st.selectbox(
                    "Sleep Duration / Hygiene", 
                    ["Optimal (7-9 hrs)", "Irregular", "Poor (<6 hrs)"],
                    index=["Optimal (7-9 hrs)", "Irregular", "Poor (<6 hrs)"].index(st.session_state.p_data["sleep_quality"])
                )
                st.session_state.p_data["stress_level"] = st.selectbox(
                    "Stress Level", 
                    ["Low", "Moderate", "High / Chronic"],
                    index=["Low", "Moderate", "High / Chronic"].index(st.session_state.p_data["stress_level"])
                )

        st.markdown("</div>", unsafe_allow_html=True)

    risk_val, tier_val = run_classification(st.session_state.p_data)
    reg_val = run_regression(st.session_state.p_data)

    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        if risk_val < 35:
            st.markdown(f"<div style='background:rgba(52,168,83,0.2); border:1px solid #34A853; border-radius:12px; padding:12px; text-align:center;'><h3 style='color:#81c995; margin:0;'>🟢 Safe / Low Risk ({risk_val}%)</h3></div>", unsafe_allow_html=True)
        elif risk_val < 65:
            st.markdown(f"<div style='background:rgba(251,188,4,0.2); border:1px solid #FBBC04; border-radius:12px; padding:12px; text-align:center;'><h3 style='color:#fdd663; margin:0;'>🟡 Moderate Risk ({risk_val}%)</h3></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:rgba(234,67,53,0.2); border:1px solid #EA4335; border-radius:12px; padding:12px; text-align:center;'><h3 style='color:#f28b82; margin:0;'>🔴 High Clinical Risk ({risk_val}%)</h3></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.plotly_chart(render_risk_gauge(risk_val), use_container_width=True)

        b1, b2, b3 = st.columns(3)
        with b1: st.metric("BMI Value", f"{st.session_state.p_data['bmi']}", delta="Overweight" if st.session_state.p_data['bmi'] >= 25 else "Normal", delta_color="inverse")
        with b2: st.metric("Fasting Glucose", f"{st.session_state.p_data['blood_glucose_level']} mg/dL", delta="Elevated" if st.session_state.p_data['blood_glucose_level'] > 100 else "Normal", delta_color="inverse")
        with b3: st.metric("HbA1c Glycation", f"{st.session_state.p_data['HbA1c_level']}%", delta="Diabetic" if st.session_state.p_data['HbA1c_level'] >= 6.5 else ("Prediabetic" if st.session_state.p_data['HbA1c_level'] >= 5.7 else "Optimal"), delta_color="inverse")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📑 **Diagnostic Summary & Actionable Protocols**")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write(f"""
    * **Estimated Risk Probability**: **{risk_val}%** ({tier_val}).
    * **Projected Glucose Model Output**: **{reg_val} mg/dL** (Reference Value: {st.session_state.p_data['blood_glucose_level']} mg/dL).
    * **Nutritional Guidance**: Transitioning toward a high-fiber Mediterranean diet (>= 35g daily soluble fiber) buffers glycemic spikes.
    * **Physical Activity Recommendation**: Engaging in 150+ minutes of weekly aerobic cardio activates GLUT4 transport independent of insulin.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: ML BENCHMARKS
# -----------------------------------------------------------------------------
elif nav_choice == "📊 ML Model Benchmarks (F1, Accuracy, RMSE, R²)":
    st.markdown("### 📊 **Machine Learning Benchmarks & Model Evaluation Metrics**")
    st.caption("Mathematical validation across Classification (Random Forest, Logistic Regression) and Regression algorithms.")

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### **1. Classification Performance Metrics**")
    
    rf_row = ml_assets["clf_df"][ml_assets["clf_df"]["Model"].str.contains("Random Forest")].iloc[0]
    log_row = ml_assets["clf_df"][ml_assets["clf_df"]["Model"] == "Logistic Regression"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("RF Model F1-Score", f"{rf_row['F1-Score']}")
    with c2: st.metric("RF Model Accuracy", f"{rf_row['Accuracy'] * 100:.1f}%")
    with c3: st.metric("Logistic Regression F1", f"{log_row['F1-Score']}")
    with c4: st.metric("Logistic Reg Accuracy", f"{log_row['Accuracy'] * 100:.1f}%")

    st.dataframe(ml_assets["clf_df"], use_container_width=True)

    # Confusion Matrix
    st.markdown("##### **Classification Confusion Matrix (Random Forest)**")
    y_test_c = ml_assets["y_test_c"]
    y_pred_rf = ml_assets["y_preds_c"]["Random Forest (Balanced)"]
    cm = confusion_matrix(y_test_c, y_pred_rf)

    cm_fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted Class", y="Actual True Class", color="Count"),
        x=["Non-Diabetic (0)", "Diabetic (1)"],
        y=["Non-Diabetic (0)", "Diabetic (1)"],
        color_continuous_scale="Blues"
    )
    cm_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e3e3e3'), height=320)
    st.plotly_chart(cm_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Regression Metrics
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### **2. Regression Model Benchmarks (Continuous Glucose Prediction)**")
    
    best_reg = ml_assets["reg_df"].iloc[0]
    r1, r2, r3 = st.columns(3)
    with r1: st.metric("R-Squared (R²)", f"{best_reg['R-Squared (R²)']}")
    with r2: st.metric("Root Mean Squared Error (RMSE)", f"{best_reg['RMSE (mg/dL)']} mg/dL")
    with r3: st.metric("Mean Absolute Error (MAE)", f"{best_reg['MAE (mg/dL)']} mg/dL")

    st.dataframe(ml_assets["reg_df"], use_container_width=True)

    y_test_r = ml_assets["y_test_r"].values[:600]
    y_pred_r = ml_assets["y_preds_r"]["Ridge Regression"][:600]

    scat_fig = go.Figure()
    scat_fig.add_trace(go.Scatter(x=y_test_r, y=y_pred_r, mode='markers', marker=dict(color='#4285F4', opacity=0.7), name='Predictions'))
    min_v, max_v = min(y_test_r.min(), y_pred_r.min()), max(y_test_r.max(), y_pred_r.max())
    scat_fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', line=dict(color='#ea4335', dash='dash'), name='Ideal Fit (y = x)'))
    scat_fig.update_layout(
        title="Actual vs Predicted Blood Glucose (mg/dL)",
        xaxis_title="Actual True Glucose",
        yaxis_title="ML Predicted Blood Glucose",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='#e3e3e3'),
        height=350
    )
    st.plotly_chart(scat_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: WHAT-IF SIMULATOR
# -----------------------------------------------------------------------------
elif nav_choice == "🔮 What-If Lifestyle Simulator":
    st.markdown("### 🔮 **What-If Scenario & Lifestyle Intervention Simulator**")
    st.caption("Simulate lifestyle modifications and target biomarker reductions to project risk score changes.")

    s1, s2 = st.columns([1.1, 1.2], gap="large")
    base_risk, _ = run_classification(st.session_state.p_data)

    with s1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### **Adjust Target Interventions**")
        t_bmi = st.slider("Target Body Mass Index (BMI)", 18.0, 40.0, float(max(st.session_state.p_data["bmi"] - 2.5, 21.0)), 0.1)
        t_gluc = st.slider("Target Fasting Glucose (mg/dL)", 70, 220, int(max(st.session_state.p_data["blood_glucose_level"] - 25, 90)), 5)
        t_a1c = st.slider("Target HbA1c Level (%)", 4.5, 8.5, float(max(st.session_state.p_data["HbA1c_level"] - 0.7, 5.2)), 0.1)
        t_diet = st.selectbox("Target Diet Framework", ["Mediterranean", "Balanced Whole-Food", "Standard Western"], index=0)
        t_act = st.selectbox("Target Exercise Level", ["High (300+ min/wk)", "Moderate (150-300 min/wk)", "Light (30-149 min/wk)"], index=1)
        st.markdown("</div>", unsafe_allow_html=True)

    sim_data = dict(st.session_state.p_data)
    sim_data.update({
        "bmi": t_bmi, 
        "blood_glucose_level": t_gluc, 
        "HbA1c_level": t_a1c,
        "diet_quality": t_diet,
        "physical_activity": t_act
    })
    sim_risk, sim_tier = run_classification(sim_data)

    with s2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### **Projected Risk Trajectory**")
        
        cfig = go.Figure(data=[
            go.Bar(name='Current Baseline Risk', x=['Risk Score %'], y=[base_risk], marker_color='#ea4335'),
            go.Bar(name='Post-Intervention Risk', x=['Risk Score %'], y=[sim_risk], marker_color='#34A853')
        ])
        cfig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)', font=dict(color='#e3e3e3'), height=200, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(cfig, use_container_width=True)

        delta = round(base_risk - sim_risk, 1)
        st.metric("Total Absolute Risk Reduction", f"{sim_risk}% (from {base_risk}%)", delta=f"-{delta}% absolute reduction", delta_color="normal")
        st.info(f"Targeting a BMI of **{t_bmi}**, HbA1c of **{t_a1c}%**, and adopting **{t_diet}** transitions clinical tier to: **{sim_tier}**.")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 4: PDF HEALTH REPORT EXPORT
# -----------------------------------------------------------------------------
else:
    st.markdown("### 📄 **Official Diagnostic Clinical PDF Export**")
    st.caption("Generate and download a hospital-grade, multi-table vector PDF report formatted with ReportLab.")

    p = st.session_state.p_data
    risk_val, tier_val = run_classification(p)
    reg_gluc = run_regression(p)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(f"#### **Diagnostic Summary Preview (Signed in as: {user['name']})**")
    
    prev_col1, prev_col2 = st.columns(2)
    with prev_col1:
        st.write(f"• **Patient Identifier**: `{patient_id}`")
        st.write(f"• **Demographics**: {p['age']} years | {p['gender']}")
        st.write(f"• **Fasting Blood Glucose**: {p['blood_glucose_level']} mg/dL")
        st.write(f"• **Serum HbA1c**: {p['HbA1c_level']}%")
    with prev_col2:
        st.write(f"• **Calculated Risk Index**: **{risk_val}%** ({tier_val})")
        st.write(f"• **Model F1-Score**: {ml_assets['metrics_summary']['rf_f1']}")
        st.write(f"• **Diet & Activity**: {p['diet_quality']} | {p['physical_activity']}")
        st.write(f"• **Account Role**: {user['role']}")

    st.markdown("---")

    # Generate Vector PDF
    pdf_buffer = PDFReportGenerator.build_pdf_report(
        patient_id=patient_id,
        p_data=p,
        risk_prob=risk_val,
        clinical_tier=tier_val,
        reg_glucose=reg_gluc,
        ml_metrics=ml_assets["metrics_summary"],
        clinician_info=user if is_doctor else {"name": "Self-Assessed Patient Record", "license": "PATIENT-SELF-SERVICE", "department": "Personal Metabolic Profile"}
    )

    st.download_button(
        label="📥 Download Clinical Diagnostic PDF Report",
        data=pdf_buffer,
        file_name=f"Metabolic_Diagnostic_Report_{patient_id}.pdf",
        mime="application/pdf",
        type="primary"
    )
    st.markdown("</div>", unsafe_allow_html=True)