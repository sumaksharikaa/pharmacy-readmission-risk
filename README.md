# 🏥 Pharmacy Readmission Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange?logo=scikit-learn)](https://scikit-learn.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-blue?logo=mlflow)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)

An end-to-end **30-Day Hospital Readmission Risk Prediction** pipeline incorporating the **Charlson Comorbidity Index (CCI)**, pharmacy features (high-risk medications, medication reconciliation), and clinical lab values. Three models trained, tracked with MLflow, and surfaced through an interactive Streamlit dashboard.

---

## 🗂️ Project Structure

```
pharmacy-readmission-risk/
├── data/
│   └── generate_data.py         # Patient cohort generator with CCI scoring
├── models/
│   ├── train_models.py          # ML pipeline: LR → RF → XGBoost + MLflow
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── results.json
├── sql/
│   └── schema.sql               # PostgreSQL schema + pharmacy alert view
├── app/
│   └── app.py                   # Streamlit dashboard (5 tabs)
├── requirements.txt
└── README.md
```

---

## 🏥 Charlson Comorbidity Index (CCI)

The CCI is a validated clinical scoring system used in pharmacy and health systems to predict 1-year mortality based on comorbid conditions. Each condition carries a weight (1, 2, 3, or 6):

| Weight | Conditions |
|---|---|
| **1** | MI, CHF, PVD, Dementia, COPD, Rheumatologic, PUD, Mild Liver Disease, Diabetes |
| **2** | Diabetes with complications, Hemiplegia, Renal Disease, Malignancy |
| **3** | Moderate/Severe Liver Disease |
| **6** | Metastatic Solid Tumor, AIDS/HIV |

This project uses ICD-10 code mapping to compute CCI scores as features for readmission risk prediction.

---

## 📊 Model Results

| Model | ROC-AUC | F1 Score | CV AUC |
|---|---|---|---|
| **Logistic Regression** | **0.8432** | **0.6705** | **0.8412** |
| Random Forest | 0.8251 | 0.6372 | 0.8174 |
| XGBoost | 0.8068 | 0.5956 | 0.7980 |

**Top Predictive Features:** CCI Score · Prior Admissions · Medication Count · High-Risk Meds · Lab Abnormalities · Length of Stay

---

## 📊 Dashboard Features

| Tab | What it shows |
|---|---|
| **📊 Risk Overview** | Risk tier distribution, score histogram, risk vs CCI scatter, insurance risk heatmap |
| **📈 Model Performance** | ROC curve overlay (3 models), metric comparison bar, confusion matrix, summary table |
| **🔬 Feature Analysis** | XGBoost importance, LR coefficients (direction), risk by CCI tier box plot |
| **🧮 CCI Calculator** | Interactive CCI scoring + real-time risk prediction gauge |
| **👤 Patient Explorer** | Actual readmit by tier, age violin, high-risk patient table |

---

## ⚙️ Setup & Run

```bash
git clone https://github.com/sumaksharikaa/pharmacy-readmission-risk.git
cd pharmacy-readmission-risk
pip install -r requirements.txt

# Generate cohort data
python data/generate_data.py

# Train models (logs to MLflow)
python models/train_models.py

# Launch dashboard
streamlit run app/app.py

# View MLflow experiment tracking
mlflow ui
```

---

## 🔑 Key Technical Concepts

| Concept | Implementation |
|---|---|
| **Charlson Comorbidity Index** | ICD-10 mapped, weighted CCI scoring per patient |
| **Class Imbalance** | `class_weight="balanced"` (LR/RF) + `scale_pos_weight` (XGBoost) |
| **MLflow Tracking** | Parameters, metrics, ROC curves, and model artifacts per run |
| **Cross-Validation** | Stratified 5-fold CV for all models |
| **Feature Engineering** | CCI score, high-risk med flag, med reconciliation, lab abnormal count |
| **PostgreSQL View** | `vw_high_risk_pharmacy_alerts` — pharmacy intervention prioritization |
| **Interactive CCI Calculator** | Real-time risk prediction from clinician-entered conditions |

---

## 🔗 Related Projects

- [Specialty Pharmacy Claims Analytics](https://github.com/sumaksharikaa/sp-claims-analytics)
- [Drug Utilization & Formulary Analytics](https://github.com/sumaksharikaa/drug-utilization-analytics)
- [Healthcare Data Quality & Governance Pipeline](https://github.com/sumaksharikaa/healthcare-dq-governance)

---

*Built by [Sumaksharika Nainavarapu](https://sumaksharika.com) · B.S. Pharmacy · M.S. Health Informatics & Analytics*
