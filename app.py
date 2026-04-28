import os
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from building_project_predictor import BuildingPredictor

st.set_page_config(
    page_title="SPbPU Building Project Prediction System",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(__file__)
EXCEL_PATH = os.path.join(BASE_DIR, "D_Building_2000_prediction_dataset.xlsx")
LOGO_PATH = os.path.join(BASE_DIR, "spbstu_logo.png")

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1280px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #eef3f8 0%, #f8fafc 100%);} 
.main-title {font-size: 2.45rem; font-weight: 900; letter-spacing: -0.045em; margin-bottom: .25rem; color:#0f172a;}
.sub-title {font-size: 1.05rem; color: #475569; margin-bottom: 1rem; max-width: 1050px; line-height:1.65;}
.section-title {font-size:1.32rem; font-weight:850; margin-top: 1.45rem; margin-bottom:.7rem; color:#0f172a;}
.hero-card {
    border: 1px solid #cbd5e1; border-radius: 26px; padding: 24px;
    background: radial-gradient(circle at top left, #e0f2fe 0%, #ffffff 38%, #f8fafc 100%);
    box-shadow: 0 18px 42px rgba(15, 23, 42, 0.075);
    margin-bottom: 16px;
}
.card {
    border: 1px solid #e2e8f0; border-radius: 22px; padding: 20px;
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.055);
    margin-bottom: 14px;
}
.team-card {
    border: 1px solid #dbe4ef; border-radius: 18px; padding: 16px;
    background: #ffffff; box-shadow: 0 8px 22px rgba(15,23,42,.05); margin-bottom: 12px;
}
.small-muted {color: #64748b; font-size: .9rem;}
.good {color: #027A48; font-weight: 850;}
.warn {color: #B54708; font-weight: 850;}
.bad {color: #B42318; font-weight: 850;}
.pill {display:inline-block; padding:7px 12px; border-radius:999px; background:#eef2ff; color:#1e293b; font-weight:800; font-size:.85rem; margin: 4px 6px 4px 0;}
.rec-box {border-left: 5px solid #2563eb; padding: 13px 16px; background:#f8fafc; border-radius: 13px; margin-bottom: 10px;}
.step-box {border-left: 5px solid #16a34a; padding: 12px 15px; background:#f7fee7; border-radius: 13px; margin-bottom: 9px;}
.stMetric {background: #ffffff; border: 1px solid #e2e8f0; padding: 14px; border-radius: 18px; box-shadow: 0 6px 18px rgba(15,23,42,.04);}
hr {margin-top: 1rem; margin-bottom: 1rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def pretty_name(col: str) -> str:
    return col.replace("_", " ").title()


def scenario_class(value: str) -> str:
    value = str(value).lower()
    if "optimistic" in value:
        return "good"
    if "pessimistic" in value:
        return "bad"
    return "warn"


def risk_class(value: str) -> str:
    value = str(value).lower()
    if value == "low":
        return "good"
    if value == "high":
        return "bad"
    return "warn"


def money(value: float) -> str:
    return f"${value:,.0f}"


def risk_interpretation(level: str) -> str:
    level = str(level).lower()
    if level == "low":
        return "The selected project profile shows relatively controlled cost and schedule exposure. Standard monitoring is still required because construction projects may change due to procurement, labor, site, or market conditions."
    if level == "high":
        return "The selected project profile shows strong exposure to cost growth and schedule delay. The project should be reviewed before execution, with stronger contingency planning, procurement control, and progress monitoring."
    return "The selected project profile shows moderate exposure. The project is manageable, but cost, time, labor, equipment, and procurement conditions should be controlled carefully from the early planning stage."


def generate_recommendations(out: dict, result: dict) -> list[str]:
    recs = []
    risk_level = str(out["risk_level"]).lower()
    cost_pct = float(out["cost_overrun_percentage"]) * 100
    time_pct = float(out["schedule_overrun_percentage"]) * 100

    if risk_level == "high":
        recs.append("Conduct a pre-execution risk review before approval, because the model classifies this project as high risk.")
        recs.append("Prepare a stronger contingency budget and define clear approval rules for design, procurement, and scope changes.")
    elif risk_level == "medium":
        recs.append("Apply weekly project control because the model indicates a moderate risk profile that can increase without early monitoring.")
        recs.append("Keep a practical contingency allowance for both cost and schedule uncertainty.")
    else:
        recs.append("Maintain standard project controls, because even low-risk projects may face unexpected site or supply conditions.")

    if cost_pct > 5:
        recs.append("Monitor material prices and supplier quotations during procurement because the predicted cost overrun is noticeable.")
    else:
        recs.append("Keep the cost baseline updated and compare planned versus predicted cost during each reporting period.")

    if time_pct > 5:
        recs.append("Strengthen schedule control through weekly planned-versus-actual comparison and early corrective actions.")
    else:
        recs.append("Maintain the current schedule logic, but continue tracking critical activities and productivity levels.")

    recs.extend([
        "Confirm labor and equipment availability before the main construction phase to reduce productivity risk.",
        "Use digital quantity, cost, and progress records to improve transparency and reduce manual reporting mistakes.",
        "Prepare a procurement plan for long-lead materials before site execution starts.",
        "Use the dashboard results as decision-support information together with engineering judgment and site investigation.",
        f"Give special attention to cost factors related to {str(result['cost_primary_cause']).lower()} and {str(result['cost_secondary_cause']).lower()}.",
        f"Give special attention to schedule factors related to {str(result['schedule_primary_cause']).lower()} and {str(result['schedule_secondary_cause']).lower()}.",
    ])
    return recs


def create_pdf_report(project_id: str, raw_df: pd.DataFrame, result: dict, recommendations: list[str]) -> bytes:
    out = result["predicted_outputs"]
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.4*cm, bottomMargin=1.4*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBlue", parent=styles["Title"], textColor=colors.HexColor("#0f172a"), fontSize=17, leading=21, spaceAfter=12))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#1d4ed8"), fontSize=12.5, leading=15, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))
    story = []
    story.append(Paragraph("Building Project Risk and Performance Prediction Report", styles["TitleBlue"]))
    story.append(Paragraph("Peter the Great St. Petersburg Polytechnic University (SPbPU)", styles["BodyText"]))
    story.append(Paragraph("Students: Hashemi Sayed Baset; Salehy Sayed Moh Meraj", styles["BodyText"]))
    story.append(Paragraph("Supervisor: Mikheev Pavel Yurievich", styles["BodyText"]))
    story.append(Paragraph(f"Project ID: {project_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Small"]))
    story.append(Spacer(1, 8))

    summary_data = [
        ["Predicted actual cost", money(out["actual_cost"])],
        ["Predicted duration", f"{out['actual_duration']:,.0f} days"],
        ["Cost overrun", f"{out['cost_overrun_percentage']*100:.2f}%"],
        ["Schedule overrun", f"{out['schedule_overrun_percentage']*100:.2f}%"],
        ["Risk level", str(out["risk_level"])],
        ["Risk score", f"{out['risk_score']:.1f}/100"],
        ["Cost scenario", str(out["cost_scenario"])],
        ["Time scenario", str(out["time_scenario"])],
    ]
    story.append(Paragraph("Executive Prediction Summary", styles["Section"]))
    table = Table(summary_data, colWidths=[7*cm, 8*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2ff")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(table)
    story.append(Paragraph("Risk Interpretation", styles["Section"]))
    story.append(Paragraph(risk_interpretation(out["risk_level"]), styles["BodyText"]))
    story.append(Paragraph("Model Explanation", styles["Section"]))
    story.append(Paragraph(str(result["result_reason_summary"]), styles["BodyText"]))
    story.append(Paragraph("Project Management Recommendations", styles["Section"]))
    for i, rec in enumerate(recommendations, start=1):
        story.append(Paragraph(f"{i}. {rec}", styles["Small"]))
    story.append(Paragraph("Input Project Profile", styles["Section"]))
    profile_rows = [[pretty_name(k), str(v)] for k, v in raw_df.iloc[0].to_dict().items()]
    profile_table = Table(profile_rows, colWidths=[7*cm, 8*cm])
    profile_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(profile_table)
    story.append(Paragraph("Academic Note", styles["Section"]))
    story.append(Paragraph("The report is produced by a decision-support system for early-stage construction project planning. The results should be interpreted together with professional engineering judgment, local market conditions, and site-specific investigation.", styles["Small"]))
    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


@st.cache_resource(show_spinner="Loading prediction model...")
def load_system():
    system = BuildingPredictor(EXCEL_PATH)
    system.load_data()
    system.train(system.data.copy())
    return system


system = load_system()

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    st.markdown("### Dashboard Overview")
    st.write("This application supports early-stage building project decision-making by estimating cost performance, schedule performance, scenario type, and risk level.")
    st.divider()
    st.markdown("### Developed by")
    st.markdown("**Hashemi Sayed Baset**")
    st.markdown("**Salehy Sayed Moh Meraj**")
    st.caption("Peter the Great St. Petersburg Polytechnic University (SPbPU)")
    st.markdown("**Supervisor:** Mikheev Pavel Yurievich")
    st.divider()
    st.caption("Model outputs")
    st.markdown("<span class='pill'>Actual Cost</span><span class='pill'>Duration</span><span class='pill'>Risk Level</span><span class='pill'>Scenarios</span>", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏗️ SPbPU Building Project Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A professional academic decision-support dashboard for estimating construction project cost, duration, cost/time scenarios, and risk level using machine-learning-based analysis.</div>',
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([1.3, 0.7])
with hero_left:
    st.markdown("""
    <div class="hero-card">
    <h3>Purpose of the System</h3>
    <p>This web application helps students, engineers, and project managers evaluate a building project before execution. Users enter technical, site, and resource information; the system then predicts cost performance, schedule performance, project risk level, and practical management recommendations.</p>
    <span class="pill">Academic Project</span><span class="pill">Construction Management</span><span class="pill">Machine Learning</span><span class="pill">Risk Intelligence</span>
    </div>
    """, unsafe_allow_html=True)
with hero_right:
    st.markdown("""
    <div class="team-card">
    <h4>Project Information</h4>
    <p><b>University:</b><br>Peter the Great St. Petersburg Polytechnic University (SPbPU)</p>
    <p><b>Students:</b><br>Hashemi Sayed Baset<br>Salehy Sayed Moh Meraj</p>
    <p><b>Supervisor:</b><br>Mikheev Pavel Yurievich</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title">How to Use This Application</div>', unsafe_allow_html=True)
h1, h2, h3 = st.columns(3)
with h1:
    st.markdown("<div class='step-box'><b>Step 1:</b><br>Enter the project identity, building characteristics, site conditions, and resource information.</div>", unsafe_allow_html=True)
with h2:
    st.markdown("<div class='step-box'><b>Step 2:</b><br>Click the prediction button to generate cost, duration, scenario, and risk outputs.</div>", unsafe_allow_html=True)
with h3:
    st.markdown("<div class='step-box'><b>Step 3:</b><br>Review the dashboard, management recommendations, and download the PDF report.</div>", unsafe_allow_html=True)

st.markdown('<div class="section-title">1. Enter Project Information</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    project_id = st.text_input("Project ID", value=f"NEW_{datetime.now().strftime('%Y%m%d_%H%M')}")

    tabs = st.tabs(["Main Data", "Technical / Site", "Resources / Environment"])
    row = {"project_id": project_id}

    input_cols = system.input_columns
    groups = [input_cols[0::3], input_cols[1::3], input_cols[2::3]]

    for tab, cols in zip(tabs, groups):
        with tab:
            col_left, col_right = st.columns(2)
            for i, col in enumerate(cols):
                target_col = col_left if i % 2 == 0 else col_right
                series = system.inputs_df[col]
                label = pretty_name(col)
                with target_col:
                    if col in system.numeric_columns:
                        values = pd.to_numeric(series, errors="coerce").dropna()
                        min_v = float(values.min())
                        max_v = float(values.max())
                        med_v = float(values.median())
                        is_int = bool((values.round() == values).all()) if len(values) else False
                        if is_int:
                            row[col] = st.number_input(label, min_value=int(min_v), max_value=int(max_v), value=int(round(med_v)), step=1)
                        else:
                            step = (max_v - min_v) / 100 if max_v > min_v else 1.0
                            row[col] = st.number_input(label, min_value=min_v, max_value=max_v, value=med_v, step=step)
                    else:
                        options = [str(x) for x in series.dropna().astype(str).unique().tolist()]
                        default = str(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else (options[0] if options else "")
                        default_index = options.index(default) if default in options else 0
                        row[col] = st.selectbox(label, options=options, index=default_index)

    submitted = st.form_submit_button("Predict Project Results", use_container_width=True)

if submitted:
    raw_df = pd.DataFrame([row])
    result = system.predict(raw_df)
    out = result["predicted_outputs"]

    st.markdown('<div class="section-title">2. Executive Prediction Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Actual Cost", money(out["actual_cost"]), f"{money(out['cost_overrun'])} overrun")
    c2.metric("Predicted Duration", f"{out['actual_duration']:,.0f} days", f"{out['schedule_deviation']:,.0f} days deviation")
    c3.metric("Cost Overrun", f"{out['cost_overrun_percentage']*100:.2f}%", out["cost_scenario"])
    c4.metric("Schedule Overrun", f"{out['schedule_overrun_percentage']*100:.2f}%", out["time_scenario"])

    st.markdown('<div class="section-title">3. Risk Interpretation Dashboard</div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, .95])
    with left:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.subheader("Overall Project Risk")
        st.markdown(f"Risk level: <span class='{risk_class(out['risk_level'])}'>{out['risk_level']}</span>", unsafe_allow_html=True)
        st.write(risk_interpretation(out["risk_level"]))
        st.progress(min(max(out["risk_score"] / 100, 0), 1), text=f"Overall risk score: {out['risk_score']:.1f}/100")
        st.progress(min(max(out["cost_risk_score"] / 100, 0), 1), text=f"Cost risk score: {out['cost_risk_score']:.1f}/100")
        st.progress(min(max(out["schedule_risk_score"] / 100, 0), 1), text=f"Schedule risk score: {out['schedule_risk_score']:.1f}/100")
        st.markdown(
            f"<span class='pill'>Cost Scenario: <span class='{scenario_class(out['cost_scenario'])}'>{out['cost_scenario']}</span></span>"
            f"<span class='pill'>Time Scenario: <span class='{scenario_class(out['time_scenario'])}'>{out['time_scenario']}</span></span>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Risk Probability Distribution")
        prob_df = pd.DataFrame({"Risk Level": list(result["risk_probabilities"].keys()), "Probability (%)": [v * 100 for v in result["risk_probabilities"].values()]})
        st.bar_chart(prob_df.set_index("Risk Level"))
        st.caption("The chart shows the model confidence distribution across Low, Medium, and High risk classes.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">4. Cost and Schedule Performance Charts</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Risk Score Comparison")
        score_df = pd.DataFrame({
            "Indicator": ["Overall Risk", "Cost Risk", "Schedule Risk"],
            "Score": [out["risk_score"], out["cost_risk_score"], out["schedule_risk_score"]],
        })
        st.bar_chart(score_df.set_index("Indicator"))
        st.markdown('</div>', unsafe_allow_html=True)
    with ch2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Overrun Comparison")
        overrun_df = pd.DataFrame({
            "Indicator": ["Cost Overrun", "Schedule Overrun"],
            "Percentage": [out["cost_overrun_percentage"] * 100, out["schedule_overrun_percentage"] * 100],
        })
        st.bar_chart(overrun_df.set_index("Indicator"))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">5. Prediction Explanation</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Cost Drivers")
        st.write(f"Primary cause: **{result['cost_primary_cause']}**")
        st.write(f"Secondary cause: **{result['cost_secondary_cause']}**")
        st.markdown('</div>', unsafe_allow_html=True)
    with e2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Schedule Drivers")
        st.write(f"Primary cause: **{result['schedule_primary_cause']}**")
        st.write(f"Secondary cause: **{result['schedule_secondary_cause']}**")
        st.markdown('</div>', unsafe_allow_html=True)
    st.info(result["result_reason_summary"])

    st.markdown('<div class="section-title">6. Project Management Recommendations</div>', unsafe_allow_html=True)
    recommendations = generate_recommendations(out, result)
    for i, rec in enumerate(recommendations, start=1):
        st.markdown(f"<div class='rec-box'><b>Recommendation {i}:</b> {rec}</div>", unsafe_allow_html=True)

    if result["input_driven_flags"]:
        st.markdown('<div class="section-title">7. Input-Based Warning Flags</div>', unsafe_allow_html=True)
        for item in result["input_driven_flags"]:
            st.warning(item)

    if result["input_warnings"]:
        st.markdown('<div class="section-title">8. Data Range Warnings</div>', unsafe_allow_html=True)
        for item in result["input_warnings"]:
            st.warning(item)

    st.markdown('<div class="section-title">9. Input Project Profile</div>', unsafe_allow_html=True)
    with st.expander("Show entered project data"):
        st.dataframe(raw_df.T.rename(columns={0: "Value"}), use_container_width=True)

    st.markdown('<div class="section-title">10. Methodology & Academic Note</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        This dashboard is designed as a decision-support tool for construction project management. The prediction model uses structured building project information to estimate actual cost, actual duration, cost overrun, schedule overrun, scenario classification, and risk level. The results support early planning, budgeting, and risk monitoring, but they should be used together with professional engineering judgment, site investigation, and expert review.
        </div>
        """,
        unsafe_allow_html=True,
    )

    pdf_bytes = create_pdf_report(project_id, raw_df, result, recommendations)
    st.download_button(
        "Download Professional Prediction Report (PDF)",
        data=pdf_bytes,
        file_name=f"building_prediction_report_{project_id}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
else:
    st.info("Fill the project information and click **Predict Project Results**. The system will generate a dashboard and a downloadable PDF report.")
