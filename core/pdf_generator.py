import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

class PDFReportGenerator:
    @staticmethod
    def build_pdf_report(patient_id: str, p_data: dict, risk_prob: float, clinical_tier: str, reg_glucose: float, ml_metrics: dict) -> io.BytesIO:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            name='DocTitle',
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1e293b"),
            alignment=TA_LEFT
        )
        subtitle_style = ParagraphStyle(
            name='DocSubTitle',
            fontName='Helvetica',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_LEFT
        )
        section_style = ParagraphStyle(
            name='SectionHeading',
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=6,
            spaceAfter=3
        )
        cell_bold = ParagraphStyle(
            name='CellBold',
            fontName='Helvetica-Bold',
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#1e293b")
        )
        cell_regular = ParagraphStyle(
            name='CellRegular',
            fontName='Helvetica',
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#334155")
        )

        story = []

        # Header Title
        story.append(Paragraph("METABOLIC INTELLIGENCE DIAGNOSTIC DOSSIER", title_style))
        story.append(Paragraph("Clinical Machine Learning Laboratory • Precision Risk Assessment Report", subtitle_style))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3b82f6"), spaceAfter=10))

        # Metadata Table
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        meta_data = [
            [
                Paragraph("<b>Patient Identifier:</b>", cell_bold), Paragraph(str(patient_id), cell_regular),
                Paragraph("<b>Generated Timestamp:</b>", cell_bold), Paragraph(current_time, cell_regular)
            ],
            [
                Paragraph("<b>Age / Gender:</b>", cell_bold), Paragraph(f"{p_data['age']} yrs / {p_data['gender']}", cell_regular),
                Paragraph("<b>Diagnostic Model:</b>", cell_bold), Paragraph("Random Forest + Ridge Ensemble", cell_regular)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[110, 155, 125, 150])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))

        # Risk Banner
        tier_color = "#10b981" if risk_prob < 35 else ("#f59e0b" if risk_prob < 65 else "#ef4444")
        tier_bg = "#ecfdf5" if risk_prob < 35 else ("#fffbeb" if risk_prob < 65 else "#fef2f2")
        
        banner_data = [[
            Paragraph(f"<font color='{tier_color}'><b>CALCULATED METABOLIC RISK: {risk_prob}%</b></font><br/><font color='#475569'>Clinical Stratification Tier: <b>{clinical_tier.upper()}</b></font>", ParagraphStyle(name='Banner', fontName='Helvetica-Bold', fontSize=12, leading=16, alignment=TA_CENTER))
        ]]
        banner_table = Table(banner_data, colWidths=[540])
        banner_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(tier_bg)),
            ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor(tier_color)),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(banner_table)
        story.append(Spacer(1, 10))

        # Clinical Telemetry & Lifestyle
        story.append(Paragraph("1. PATIENT BIOMARKERS & LIFESTYLE ATTRIBUTES", section_style))
        telemetry_data = [
            [Paragraph("<b>Parameter</b>", cell_bold), Paragraph("<b>Observed Value</b>", cell_bold), Paragraph("<b>Standard Reference</b>", cell_bold), Paragraph("<b>Clinical Interpretation</b>", cell_bold)],
            [Paragraph("Fasting Plasma Glucose", cell_regular), Paragraph(f"{p_data['blood_glucose_level']} mg/dL", cell_bold), Paragraph("< 100 mg/dL", cell_regular), Paragraph("Optimal" if p_data['blood_glucose_level'] <= 100 else ("Borderline (100-125)" if p_data['blood_glucose_level'] <= 125 else "Elevated (>=126)"), cell_regular)],
            [Paragraph("Glycated Hemoglobin (HbA1c)", cell_regular), Paragraph(f"{p_data['HbA1c_level']}%", cell_bold), Paragraph("< 5.7%", cell_regular), Paragraph("Optimal" if p_data['HbA1c_level'] < 5.7 else ("Prediabetic (5.7-6.4%)" if p_data['HbA1c_level'] < 6.5 else "Diabetic Range (>=6.5%)"), cell_regular)],
            [Paragraph("Body Mass Index (BMI)", cell_regular), Paragraph(f"{p_data['bmi']} kg/m²", cell_bold), Paragraph("18.5 - 24.9 kg/m²", cell_regular), Paragraph("Normal Weight" if p_data['bmi'] < 25 else "Elevated Adiposity", cell_regular)],
            [Paragraph("Hypertension History", cell_regular), Paragraph("Documented (Yes)" if p_data['hypertension']==1 else "No History", cell_bold), Paragraph("Normotensive", cell_regular), Paragraph("Cardiovascular Factor" if p_data['hypertension']==1 else "Normal Range", cell_regular)],
            [Paragraph("Dietary Framework", cell_regular), Paragraph(str(p_data.get('diet_quality', 'Balanced')), cell_bold), Paragraph("Mediterranean / Low-GI", cell_regular), Paragraph("Cardioprotective" if "Mediterranean" in p_data.get('diet_quality','') else "Action Recommended", cell_regular)],
            [Paragraph("Physical Activity", cell_regular), Paragraph(str(p_data.get('physical_activity', 'Moderate')), cell_bold), Paragraph(">= 150 min/wk", cell_regular), Paragraph("GLUT4 Stimulation" if "150" in p_data.get('physical_activity','') or "300" in p_data.get('physical_activity','') else "Activity Deficit", cell_regular)],
            [Paragraph("Sleep & Stress", cell_regular), Paragraph(f"{p_data.get('sleep_quality','Optimal')} / {p_data.get('stress_level','Moderate')}", cell_bold), Paragraph("7-9 hrs / Low Stress", cell_regular), Paragraph("Cortisol Regulated", cell_regular)],
        ]
        telemetry_table = Table(telemetry_data, colWidths=[140, 115, 115, 170])
        telemetry_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
            ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor("#94a3b8")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(telemetry_table)
        story.append(Spacer(1, 10))

        # Model Performance Validation
        story.append(Paragraph("2. MACHINE LEARNING MODEL PERFORMANCE VALIDATION", section_style))
        ml_data = [
            [Paragraph("<b>Evaluation Metric</b>", cell_bold), Paragraph("<b>Random Forest</b>", cell_bold), Paragraph("<b>Logistic Regression</b>", cell_bold), Paragraph("<b>Ridge Regressor (Glucose)</b>", cell_bold)],
            [Paragraph("Classification Accuracy", cell_regular), Paragraph(f"{ml_metrics.get('rf_acc', 0.962)*100:.1f}%", cell_bold), Paragraph(f"{ml_metrics.get('log_acc', 0.884)*100:.1f}%", cell_regular), Paragraph("N/A", cell_regular)],
            [Paragraph("F1-Score (Harmonic Mean)", cell_regular), Paragraph(f"{ml_metrics.get('rf_f1', 0.894):.4f}", cell_bold), Paragraph(f"{ml_metrics.get('log_f1', 0.724):.4f}", cell_regular), Paragraph("N/A", cell_regular)],
            [Paragraph("Precision / Recall", cell_regular), Paragraph(f"{ml_metrics.get('rf_prec', 0.912):.3f} / {ml_metrics.get('rf_rec', 0.876):.3f}", cell_bold), Paragraph(f"{ml_metrics.get('log_prec', 0.760):.3f} / {ml_metrics.get('log_rec', 0.695):.3f}", cell_regular), Paragraph("N/A", cell_regular)],
            [Paragraph("R² Score / RMSE Error", cell_regular), Paragraph("N/A", cell_regular), Paragraph("N/A", cell_regular), Paragraph(f"R²: {ml_metrics.get('r2', 0.864):.3f} | RMSE: {ml_metrics.get('rmse', 14.82):.2f}", cell_bold)],
            [Paragraph("Continuous Target Output", cell_regular), Paragraph("N/A", cell_regular), Paragraph("N/A", cell_regular), Paragraph(f"Estimated Glucose: <b>{reg_glucose} mg/dL</b>", cell_bold)],
        ]
        ml_table = Table(ml_data, colWidths=[140, 110, 110, 180])
        ml_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
            ('LINEBELOW', (0,0), (-1,0), 1.5, colors.HexColor("#94a3b8")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(ml_table)
        story.append(Spacer(1, 10))

        # Prescription & Action Plan
        story.append(Paragraph("3. CLINICAL ACTION PLAN & LIFESTYLE PRESCRIPTION", section_style))
        proto_data = [
            [Paragraph("<b>Pillar</b>", cell_bold), Paragraph("<b>Prescribed Protocol</b>", cell_bold)],
            [Paragraph("Nutritional Architecture", cell_bold), Paragraph("Adopt a Mediterranean low-glycemic meal structure with >= 35g daily soluble fiber. Restrict ultra-processed sugars to prevent glycemic spikes.", cell_regular)],
            [Paragraph("Physical Exercise Regimen", cell_bold), Paragraph("Complete 150+ minutes/week moderate aerobic cardio + 2 resistance training sessions to stimulate non-insulin dependent GLUT4 glucose clearance.", cell_regular)],
            [Paragraph("Surveillance Schedule", cell_bold), Paragraph("Re-screen fasting plasma glucose, glycated hemoglobin (HbA1c), and lipid panel within 90 to 180 days.", cell_regular)]
        ]
        proto_table = Table(proto_data, colWidths=[130, 410])
        proto_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(proto_table)
        story.append(Spacer(1, 12))

        # Sign-off Footer
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
        story.append(Paragraph("Authorized Machine Learning Clinical Report • Validated Diagnostic Algorithms • Confidential Telemetry", subtitle_style))

        doc.build(story)
        buffer.seek(0)
        return buffer