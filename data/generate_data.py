"""
generate_data.py
Pharmacy Readmission Risk Predictor — Data Generator
Generates patient cohort with clinical features, CCI scores,
pharmacy data, lab abnormalities, and 30-day readmission labels.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

random.seed(42)
np.random.seed(42)

OUT = os.path.dirname(__file__)

# =============================================================================
# CHARLSON COMORBIDITY INDEX — ICD-10 mapping
# Each condition has a CCI weight (1, 2, 3, or 6)
# =============================================================================
CCI_CONDITIONS = {
    "Myocardial Infarction":       (1,  ["I21","I22","I25.2"]),
    "Congestive Heart Failure":    (1,  ["I50"]),
    "Peripheral Vascular Disease": (1,  ["I70","I71","I73"]),
    "Cerebrovascular Disease":     (1,  ["I60","I61","I62","I63","I64","I69"]),
    "Dementia":                    (1,  ["F00","F01","F02","F03","G30"]),
    "Chronic Pulmonary Disease":   (1,  ["J40","J41","J42","J43","J44","J45"]),
    "Rheumatologic Disease":       (1,  ["M05","M06","M32","M33","M34"]),
    "Peptic Ulcer Disease":        (1,  ["K25","K26","K27","K28"]),
    "Mild Liver Disease":          (1,  ["B18","K70","K71","K73","K74"]),
    "Diabetes w/o Complications":  (1,  ["E10","E11","E13"]),
    "Diabetes w/ Complications":   (2,  ["E10.2","E10.3","E11.2","E11.3","E13.2"]),
    "Hemiplegia/Paraplegia":       (2,  ["G04","G81","G82"]),
    "Renal Disease":               (2,  ["N03","N05","N18","N19","Z49","Z94.0"]),
    "Malignancy":                  (2,  ["C00","C01","C02","C10","C34","C50","C61","C90"]),
    "Moderate/Severe Liver Disease":(3, ["K72","K76.6","I85"]),
    "Metastatic Solid Tumor":      (6,  ["C77","C78","C79","C80"]),
    "AIDS/HIV":                    (6,  ["B20","B21","B22","B24","Z21"]),
}

MEDICATIONS = ["Metformin","Lisinopril","Atorvastatin","Amlodipine","Furosemide",
               "Warfarin","Insulin","Metoprolol","Carvedilol","Spironolactone",
               "Omeprazole","Gabapentin","Sertraline","Hydrocodone","Prednisone"]

STATES = ["NC","TX","CA","FL","NY","GA","OH","PA","IL","AZ"]


def assign_cci(n_conditions):
    """Randomly assign CCI conditions to a patient."""
    conditions = random.sample(list(CCI_CONDITIONS.keys()),
                               min(n_conditions, len(CCI_CONDITIONS)))
    score = sum(CCI_CONDITIONS[c][0] for c in conditions)
    icd10 = [random.choice(CCI_CONDITIONS[c][1]) for c in conditions]
    return conditions, score, icd10


def readmission_prob(age, cci_score, los, prior_admissions,
                     med_count, lab_abnormal_count, insurance):
    """Logistic function — higher CCI/age/prior admissions → higher probability."""
    logit = (-7.0
             + 0.04  * age
             + 0.35  * cci_score
             + 0.08  * los
             + 0.50  * prior_admissions
             + 0.12  * med_count
             + 0.20  * lab_abnormal_count
             + (0.40 if insurance == "Medicaid" else 0)
             + (0.25 if insurance == "Self-Pay"  else 0)
             + np.random.normal(0, 0.3))
    return 1 / (1 + np.exp(-logit))


# =============================================================================
# GENERATE PATIENT COHORT (2500 index admissions)
# =============================================================================
n = 2500
records = []

for i in range(1, n + 1):
    age          = random.randint(18, 90)
    gender       = random.choice(["M","F"])
    state        = random.choice(STATES)
    insurance    = random.choice(["Commercial","Medicare","Medicaid","Self-Pay"],
                                  ) if False else np.random.choice(
                       ["Commercial","Medicare","Medicaid","Self-Pay"],
                       p=[0.40, 0.35, 0.18, 0.07])

    # CCI
    n_cond       = np.random.choice([0,1,2,3,4,5], p=[0.15,0.25,0.25,0.20,0.10,0.05])
    conditions, cci_score, icd10_codes = assign_cci(n_cond)

    # Clinical features
    los              = max(1, int(np.random.exponential(4.5)))
    prior_admissions = np.random.choice([0,1,2,3,4], p=[0.45,0.28,0.15,0.08,0.04])
    med_count        = random.randint(1, 12)
    lab_abnormal     = random.randint(0, min(med_count, 6))
    bmi              = round(np.random.normal(28.5, 6.0), 1)
    bmi              = max(15, min(55, bmi))
    sodium           = round(np.random.normal(138, 4), 1)
    creatinine       = round(abs(np.random.normal(1.1, 0.8)), 2)
    hemoglobin       = round(np.random.normal(12.5, 2.5), 1)
    wbc              = round(abs(np.random.normal(8.5, 3.0)), 1)

    # Pharmacy features
    meds_list        = random.sample(MEDICATIONS, min(med_count, len(MEDICATIONS)))
    high_risk_meds   = sum(1 for m in meds_list if m in ["Warfarin","Insulin","Furosemide","Prednisone"])
    discharge_meds   = random.randint(max(1, med_count - 2), med_count + 3)
    med_reconciled   = random.random() > 0.25  # 75% reconciled at discharge

    # Admission date
    adm_date = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 1094))

    # Readmission label
    prob     = readmission_prob(age, cci_score, los, prior_admissions,
                                med_count, lab_abnormal, insurance)
    readmit  = int(np.random.binomial(1, min(prob, 0.95)))

    records.append({
        "patient_id":          f"PT{i:06d}",
        "age":                 age,
        "gender":              gender,
        "state":               state,
        "insurance_type":      insurance,
        "admission_date":      adm_date.strftime("%Y-%m-%d"),
        "length_of_stay":      los,
        "prior_admissions_6m": prior_admissions,
        "cci_score":           cci_score,
        "n_conditions":        n_cond,
        "conditions":          "|".join(conditions),
        "icd10_codes":         "|".join(icd10_codes),
        "medication_count":    med_count,
        "high_risk_med_count": high_risk_meds,
        "medications":         "|".join(meds_list),
        "discharge_med_count": discharge_meds,
        "med_reconciled":      int(med_reconciled),
        "lab_abnormal_count":  lab_abnormal,
        "sodium":              sodium,
        "creatinine":          creatinine,
        "hemoglobin":          hemoglobin,
        "wbc":                 wbc,
        "bmi":                 bmi,
        "readmit_30day":       readmit,
        "readmit_prob_actual": round(prob, 4),
    })

df = pd.DataFrame(records)
df.to_csv(f"{OUT}/patient_cohort.csv", index=False)

readmit_rate = df["readmit_30day"].mean() * 100
print(f"✅ Generated patient_cohort.csv → {len(df):,} rows")
print(f"   30-day readmission rate: {readmit_rate:.1f}%")
print(f"   Avg CCI score:           {df['cci_score'].mean():.2f}")
print(f"   Avg LOS:                 {df['length_of_stay'].mean():.1f} days")
print(f"   Avg age:                 {df['age'].mean():.1f}")
