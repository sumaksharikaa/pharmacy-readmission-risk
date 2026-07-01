"""
app.py — Pharmacy Readmission Risk Predictor Dashboard
Run: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, sys, joblib
warnings = __import__("warnings"); warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

RISK_COLORS = {"Low":"#2ecc71","Moderate":"#f1c40f","High":"#e67e22","Very High":"#e74c3c"}
MODEL_COLORS= {"Logistic Regression":"#3498db","Random Forest":"#2ecc71","XGBoost":"#e74c3c"}

CCI_CONDITIONS = {
    "Myocardial Infarction":1,"Congestive Heart Failure":1,"Peripheral Vascular Disease":1,
    "Cerebrovascular Disease":1,"Dementia":1,"Chronic Pulmonary Disease":1,
    "Rheumatologic Disease":1,"Peptic Ulcer Disease":1,"Mild Liver Disease":1,
    "Diabetes w/o Complications":1,"Diabetes w/ Complications":2,"Hemiplegia/Paraplegia":2,
    "Renal Disease":2,"Malignancy":2,"Moderate/Severe Liver Disease":3,
    "Metastatic Solid Tumor":6,"AIDS/HIV":6,
}

@st.cache_data
def load_data():
    df      = pd.read_csv(f"{DATA_DIR}/risk_scores.csv")
    results = json.load(open(f"{MODELS_DIR}/results.json"))
    return df, results

@st.cache_resource
def load_model():
    return joblib.load(f"{MODELS_DIR}/xgboost.pkl")

df, results = load_data()
model       = load_model()

FEATURES = ["age","cci_score","n_conditions","length_of_stay","prior_admissions_6m",
            "medication_count","high_risk_med_count","discharge_med_count","med_reconciled",
            "lab_abnormal_count","sodium","creatinine","hemoglobin","wbc","bmi",
            "gender_M","insurance_Medicare","insurance_Medicaid","insurance_Self-Pay"]

# ── sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("🏥 Risk Filters")
sel_tier   = st.sidebar.multiselect("Risk Tier",
    ["Low","Moderate","High","Very High"], default=["Low","Moderate","High","Very High"])
sel_ins    = st.sidebar.multiselect("Insurance",
    df["insurance_type"].unique().tolist(), default=df["insurance_type"].unique().tolist())
cci_range  = st.sidebar.slider("CCI Score Range", 0, int(df["cci_score"].max()), (0, 15))
age_range  = st.sidebar.slider("Age Range", 18, 90, (18, 90))

mask = (df["risk_tier"].isin(sel_tier) &
        df["insurance_type"].isin(sel_ins) &
        df["cci_score"].between(*cci_range) &
        df["age"].between(*age_range))
fdf = df[mask].copy()

# ── header ────────────────────────────────────────────────────────────────────
st.title("🏥 Pharmacy Readmission Risk Predictor")
st.caption("Charlson Comorbidity Index · XGBoost · Logistic Regression · Random Forest · MLflow")
st.divider()

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
best = next(r for r in results if r["model"]=="Logistic Regression")
k1.metric("Patients Analyzed",   f"{len(fdf):,}")
k2.metric("Actual Readmit Rate", f"{fdf['readmit_30day'].mean()*100:.1f}%")
k3.metric("High/Very High Risk", f"{fdf['risk_tier'].isin(['High','Very High']).sum():,}")
k4.metric("Best ROC-AUC",        f"{best['metrics']['roc_auc']:.4f}")
k5.metric("Best F1 Score",       f"{best['metrics']['f1']:.4f}")
k6.metric("Avg CCI Score",       f"{fdf['cci_score'].mean():.1f}")
st.divider()

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊 Risk Overview","📈 Model Performance",
    "🔬 Feature Analysis","🧮 CCI Calculator","👤 Patient Explorer"
])

# ══ TAB 1 — RISK OVERVIEW ════════════════════════════════════════════════════
with tab1:
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Risk Tier Distribution")
        tier_cnt = fdf["risk_tier"].value_counts().reset_index()
        tier_cnt.columns = ["Tier","Count"]
        tier_order = ["Low","Moderate","High","Very High"]
        tier_cnt["Tier"] = pd.Categorical(tier_cnt["Tier"], categories=tier_order, ordered=True)
        tier_cnt = tier_cnt.sort_values("Tier")
        fig = px.bar(tier_cnt, x="Tier", y="Count", color="Tier",
                     color_discrete_map=RISK_COLORS,
                     text="Count",
                     labels={"Count":"Patients","Tier":"Risk Tier"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Risk Score Distribution")
        fig2 = px.histogram(fdf, x="risk_score", nbins=40,
                            color="risk_tier", color_discrete_map=RISK_COLORS,
                            labels={"risk_score":"Predicted Risk Score","count":"Patients"},
                            barmode="overlay", opacity=0.75)
        fig2.add_vline(x=0.50, line_dash="dash", line_color="#2c3e50",
                       annotation_text="Decision Threshold (0.50)")
        fig2.update_layout(legend_title="Risk Tier")
        st.plotly_chart(fig2, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Avg Risk Score by Insurance Type")
        ins_risk = (fdf.groupby("insurance_type")
                    .agg(avg_risk=("risk_score","mean"),
                         readmit_rate=("readmit_30day","mean"),
                         count=("patient_id","count"))
                    .reset_index().sort_values("avg_risk",ascending=False))
        fig3 = px.bar(ins_risk, x="insurance_type", y="avg_risk",
                      color="avg_risk", color_continuous_scale=["#2ecc71","#e74c3c"],
                      text=ins_risk["avg_risk"].round(3),
                      labels={"avg_risk":"Avg Risk Score","insurance_type":"Insurance"})
        fig3.update_layout(coloraxis_showscale=False, xaxis_title=None)
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        st.subheader("Risk Score vs CCI Score")
        fig4 = px.scatter(fdf.sample(min(800,len(fdf))),
                          x="cci_score", y="risk_score",
                          color="risk_tier", color_discrete_map=RISK_COLORS,
                          size="length_of_stay",
                          hover_data=["age","prior_admissions_6m","medication_count"],
                          labels={"cci_score":"CCI Score","risk_score":"Predicted Risk Score"})
        fig4.add_hline(y=0.50, line_dash="dash", line_color="#2c3e50")
        fig4.update_layout(legend_title="Risk Tier")
        st.plotly_chart(fig4, use_container_width=True)

# ══ TAB 2 — MODEL PERFORMANCE ════════════════════════════════════════════════
with tab2:
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("ROC Curves — All Models")
        fig = go.Figure()
        fig.add_shape(type="line", x0=0,y0=0,x1=1,y1=1,
                      line=dict(dash="dash",color="#95a5a6"))
        for r in results:
            fig.add_trace(go.Scatter(
                x=r["fpr"], y=r["tpr"], mode="lines",
                name=f"{r['model']} (AUC={r['metrics']['roc_auc']:.4f})",
                line=dict(color=MODEL_COLORS[r["model"]], width=2.5)
            ))
        fig.update_layout(xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          legend=dict(x=0.55,y=0.05),
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Model Comparison — Key Metrics")
        metrics_df = pd.DataFrame([{
            "Model":  r["model"],
            "ROC-AUC":r["metrics"]["roc_auc"],
            "F1":     r["metrics"]["f1"],
            "Precision":r["metrics"]["precision"],
            "Recall": r["metrics"]["recall"],
            "CV AUC": r["cv_mean"],
        } for r in results])

        fig2 = px.bar(metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                      x="Metric", y="Score", color="Model",
                      color_discrete_map=MODEL_COLORS, barmode="group",
                      labels={"Score":"Score","Metric":"Metric"})
        fig2.update_layout(legend_title="Model", yaxis_range=[0,1])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Confusion Matrix — XGBoost")
    xgb_result = next(r for r in results if r["model"]=="XGBoost")
    cm = np.array(xgb_result["cm"])
    fig3 = px.imshow(cm,
                     labels=dict(x="Predicted", y="Actual", color="Count"),
                     x=["No Readmit","Readmit"], y=["No Readmit","Readmit"],
                     color_continuous_scale="Blues", text_auto=True)
    fig3.update_layout(height=320)
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Metrics Summary Table")
    st.dataframe(metrics_df.style.highlight_max(subset=["ROC-AUC","F1","Precision","Recall","CV AUC"],
                                                 color="#d5f5e3"),
                 use_container_width=True, hide_index=True)

# ══ TAB 3 — FEATURE ANALYSIS ═════════════════════════════════════════════════
with tab3:
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("XGBoost Feature Importance")
        xgb_fi = next(r for r in results if r["model"]=="XGBoost")["feature_importance"]
        fi_df  = pd.DataFrame(list(xgb_fi.items()), columns=["Feature","Importance"])
        fi_df  = fi_df.sort_values("Importance",ascending=True).tail(12)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance",
                     color_continuous_scale=["#aed6f1","#1a3c5e"],
                     labels={"Importance":"Feature Importance","Feature":""})
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Logistic Regression Coefficients")
        lr_fi = next(r for r in results if r["model"]=="Logistic Regression")["feature_importance"]
        lr_df = pd.DataFrame(list(lr_fi.items()), columns=["Feature","Coefficient"])
        lr_df = lr_df.sort_values("Coefficient", ascending=True).tail(12)
        lr_df["Direction"] = lr_df["Coefficient"].apply(lambda x: "Increases Risk" if x>0 else "Decreases Risk")
        fig2 = px.bar(lr_df, x="Coefficient", y="Feature", orientation="h",
                      color="Direction",
                      color_discrete_map={"Increases Risk":"#e74c3c","Decreases Risk":"#2ecc71"},
                      labels={"Coefficient":"Coefficient","Feature":""})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Risk Score by CCI Risk Tier")
    cci_tiers = pd.cut(fdf["cci_score"], bins=[-1,0,2,4,100],
                       labels=["Low (0)","Moderate (1-2)","High (3-4)","Very High (5+)"])
    fig3 = px.box(fdf.assign(cci_tier=cci_tiers),
                  x="cci_tier", y="risk_score",
                  color="cci_tier",
                  color_discrete_sequence=["#2ecc71","#f1c40f","#e67e22","#e74c3c"],
                  labels={"risk_score":"Predicted Risk Score","cci_tier":"CCI Risk Tier"})
    fig3.add_hline(y=0.50, line_dash="dash", line_color="#2c3e50",
                   annotation_text="Decision Threshold")
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

# ══ TAB 4 — CCI CALCULATOR ═══════════════════════════════════════════════════
with tab4:
    st.subheader("🧮 Charlson Comorbidity Index Calculator + Risk Predictor")
    st.caption("Select patient conditions to compute CCI score and predict 30-day readmission risk")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**Patient Demographics**")
        inp_age  = st.slider("Age", 18, 90, 65)
        inp_los  = st.slider("Length of Stay (days)", 1, 30, 5)
        inp_prior= st.slider("Prior Admissions (6 months)", 0, 5, 1)
        inp_ins  = st.selectbox("Insurance Type",["Commercial","Medicare","Medicaid","Self-Pay"])

    with col2:
        st.markdown("**Pharmacy & Lab**")
        inp_meds     = st.slider("Total Medications", 1, 15, 6)
        inp_highrisk = st.slider("High-Risk Medications (Warfarin/Insulin/etc)", 0, 5, 1)
        inp_labs     = st.slider("Abnormal Lab Values", 0, 8, 2)
        inp_recon    = st.checkbox("Medication Reconciled at Discharge", value=True)

    st.markdown("**Comorbid Conditions (Charlson Index)**")
    cols = st.columns(3)
    selected_conditions = []
    for idx, (cond, weight) in enumerate(CCI_CONDITIONS.items()):
        col = cols[idx % 3]
        if col.checkbox(f"{cond} (weight: {weight})", key=cond):
            selected_conditions.append((cond, weight))

    cci = sum(w for _, w in selected_conditions)

    st.divider()
    c1,c2,c3 = st.columns(3)
    c1.metric("CCI Score", cci)
    cci_tier = "Low" if cci==0 else "Moderate" if cci<=2 else "High" if cci<=4 else "Very High"
    c2.metric("CCI Risk Tier", cci_tier)

    # Predict
    X_inp = pd.DataFrame([{
        "age":                inp_age,
        "cci_score":          cci,
        "n_conditions":       len(selected_conditions),
        "length_of_stay":     inp_los,
        "prior_admissions_6m":inp_prior,
        "medication_count":   inp_meds,
        "high_risk_med_count":inp_highrisk,
        "discharge_med_count":inp_meds,
        "med_reconciled":     int(inp_recon),
        "lab_abnormal_count": inp_labs,
        "sodium":138.0, "creatinine":1.0, "hemoglobin":13.0, "wbc":8.0, "bmi":27.0,
        "gender_M":1,
        "insurance_Medicare": int(inp_ins=="Medicare"),
        "insurance_Medicaid": int(inp_ins=="Medicaid"),
        "insurance_Self-Pay": int(inp_ins=="Self-Pay"),
    }])
    risk_score = model.predict_proba(X_inp)[0][1]
    risk_label = "Low" if risk_score<0.20 else "Moderate" if risk_score<0.40 else "High" if risk_score<0.60 else "Very High"
    c3.metric("Predicted Risk", f"{risk_score:.1%}", delta=risk_label)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score*100,
        title={"text":"30-Day Readmission Risk"},
        delta={"reference":30, "suffix":"%"},
        gauge={
            "axis":{"range":[0,100],"ticksuffix":"%"},
            "bar":{"color": RISK_COLORS.get(risk_label,"#95a5a6")},
            "steps":[{"range":[0,20],"color":"#d5f5e3"},
                     {"range":[20,40],"color":"#fef9e7"},
                     {"range":[40,60],"color":"#fdebd0"},
                     {"range":[60,100],"color":"#fadbd8"}],
            "threshold":{"line":{"color":"#2c3e50","width":3},"value":50}
        },
        number={"suffix":"%","font":{"size":32}},
    ))
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    if selected_conditions:
        st.markdown("**Conditions contributing to CCI:**")
        for cond, w in selected_conditions:
            st.markdown(f"- {cond} → +{w} point{'s' if w>1 else ''}")

# ══ TAB 5 — PATIENT EXPLORER ═════════════════════════════════════════════════
with tab5:
    st.subheader("👤 Patient-Level Risk Explorer")
    c1,c2 = st.columns(2)

    with c1:
        st.subheader("Readmit Rate by Risk Tier")
        tier_actual = (fdf.groupby("risk_tier")
                       .agg(readmit_rate=("readmit_30day","mean"),
                            avg_risk=("risk_score","mean"),
                            count=("patient_id","count"))
                       .reset_index())
        tier_actual["Risk Tier"] = pd.Categorical(tier_actual["risk_tier"],
            categories=["Low","Moderate","High","Very High"], ordered=True)
        tier_actual = tier_actual.sort_values("Risk Tier")
        fig = px.bar(tier_actual, x="risk_tier", y="readmit_rate",
                     color="risk_tier", color_discrete_map=RISK_COLORS,
                     text=tier_actual["readmit_rate"].apply(lambda x: f"{x:.1%}"),
                     labels={"readmit_rate":"Actual Readmit Rate","risk_tier":"Risk Tier"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Age Distribution by Risk Tier")
        fig2 = px.violin(fdf, x="risk_tier", y="age", color="risk_tier",
                         color_discrete_map=RISK_COLORS, box=True,
                         labels={"age":"Age","risk_tier":"Risk Tier"})
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("High-Risk Patient Table (Risk Score ≥ 0.60)")
    high_risk = fdf[fdf["risk_score"]>=0.60].sort_values("risk_score",ascending=False)
    display_cols = ["patient_id","age","insurance_type","cci_score","length_of_stay",
                    "prior_admissions_6m","medication_count","high_risk_med_count",
                    "risk_score","risk_tier","readmit_30day"]
    st.dataframe(
        high_risk[display_cols].head(50)
        .rename(columns={"patient_id":"Patient","age":"Age","insurance_type":"Insurance",
                          "cci_score":"CCI","length_of_stay":"LOS",
                          "prior_admissions_6m":"Prior Admits","medication_count":"Meds",
                          "high_risk_med_count":"High-Risk Meds","risk_score":"Risk Score",
                          "risk_tier":"Risk Tier","readmit_30day":"Actual Readmit"}),
        use_container_width=True, hide_index=True
    )

st.divider()
st.caption("Pharmacy Readmission Risk Predictor · XGBoost · scikit-learn · MLflow · Charlson Comorbidity Index · sumaksharika.com")
