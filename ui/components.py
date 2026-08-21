import streamlit as st
import plotly.graph_objects as go

def render_header(patient_id: str):
    st.markdown(f"""
    <div style="margin-top: 6px; margin-bottom: 20px;">
        <h1 style="font-size: 2.5rem; margin-bottom: 2px;">
            <span class="gemini-gradient-text">📋 Metabolic Intelligence Diagnostic & PDF Suite</span>
        </h1>
        <p style="color: #9aa0a6; font-size: 1.02rem; margin-top: 0;">
            Patient Identifier: <b style="color: #38bdf8;">{patient_id}</b> | Architecture: <b>Random Forest (F1: 0.90+), Logistic Regression & Ridge Regressor</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_risk_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "%", 'font': {'color': '#ffffff', 'size': 38}},
        title={'text': "Calculated Diabetes Risk Probability", 'font': {'color': '#9aa0a6', 'size': 14}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4285F4", 'thickness': 0.28},
            'steps': [
                {'range': [0, 35], 'color': 'rgba(52, 168, 83, 0.25)'},
                {'range': [35, 65], 'color': 'rgba(251, 188, 4, 0.25)'},
                {'range': [65, 100], 'color': 'rgba(234, 67, 53, 0.25)'}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=10, r=10, t=25, b=10)
    )
    return fig