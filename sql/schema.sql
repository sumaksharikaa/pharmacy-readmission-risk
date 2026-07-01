-- =============================================================================
-- schema.sql
-- Pharmacy Readmission Risk Predictor — PostgreSQL Schema
-- =============================================================================

DROP TABLE IF EXISTS model_predictions CASCADE;
DROP TABLE IF EXISTS model_runs         CASCADE;
DROP TABLE IF EXISTS patient_cohort     CASCADE;

CREATE TABLE patient_cohort (
    patient_id              VARCHAR(10)     PRIMARY KEY,
    age                     SMALLINT        NOT NULL,
    gender                  CHAR(1),
    state                   CHAR(2),
    insurance_type          VARCHAR(20),
    admission_date          DATE,
    length_of_stay          SMALLINT,
    prior_admissions_6m     SMALLINT        DEFAULT 0,
    cci_score               SMALLINT        NOT NULL DEFAULT 0,
    n_conditions            SMALLINT        DEFAULT 0,
    conditions              TEXT,
    icd10_codes             TEXT,
    medication_count        SMALLINT,
    high_risk_med_count     SMALLINT        DEFAULT 0,
    discharge_med_count     SMALLINT,
    med_reconciled          BOOLEAN         DEFAULT TRUE,
    lab_abnormal_count      SMALLINT        DEFAULT 0,
    sodium                  NUMERIC(5,1),
    creatinine              NUMERIC(5,2),
    hemoglobin              NUMERIC(5,1),
    wbc                     NUMERIC(5,1),
    bmi                     NUMERIC(5,1),
    readmit_30day           SMALLINT        CHECK (readmit_30day IN (0,1)),
    created_at              TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    INDEX_CHECK             SMALLINT        GENERATED ALWAYS AS (
                                CASE WHEN cci_score >= 5 THEN 1 ELSE 0 END
                            ) STORED
);

CREATE INDEX idx_cohort_cci      ON patient_cohort(cci_score);
CREATE INDEX idx_cohort_ins      ON patient_cohort(insurance_type);
CREATE INDEX idx_cohort_readmit  ON patient_cohort(readmit_30day);

CREATE TABLE model_runs (
    run_id          VARCHAR(50)     PRIMARY KEY,
    model_name      VARCHAR(50)     NOT NULL,
    run_time        TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    roc_auc         NUMERIC(6,4),
    accuracy        NUMERIC(6,4),
    f1_score        NUMERIC(6,4),
    precision_score NUMERIC(6,4),
    recall_score    NUMERIC(6,4),
    cv_auc_mean     NUMERIC(6,4),
    cv_auc_std      NUMERIC(6,4),
    train_size      INT,
    test_size       INT,
    n_features      SMALLINT,
    hyperparams     JSONB,
    notes           TEXT
);

CREATE TABLE model_predictions (
    prediction_id   SERIAL          PRIMARY KEY,
    patient_id      VARCHAR(10)     NOT NULL REFERENCES patient_cohort(patient_id),
    run_id          VARCHAR(50)     NOT NULL REFERENCES model_runs(run_id),
    risk_score      NUMERIC(6,4)    NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    risk_tier       VARCHAR(12)     CHECK (risk_tier IN ('Low','Moderate','High','Very High')),
    predicted_readmit SMALLINT      CHECK (predicted_readmit IN (0,1)),
    actual_readmit  SMALLINT        CHECK (actual_readmit IN (0,1)),
    is_correct      BOOLEAN         GENERATED ALWAYS AS (predicted_readmit = actual_readmit) STORED,
    scored_at       TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pred_patient   ON model_predictions(patient_id);
CREATE INDEX idx_pred_risk_tier ON model_predictions(risk_tier);
CREATE INDEX idx_pred_run       ON model_predictions(run_id);

-- View: High-risk patients requiring pharmacy intervention
CREATE OR REPLACE VIEW vw_high_risk_pharmacy_alerts AS
SELECT
    pc.patient_id,
    pc.age,
    pc.insurance_type,
    pc.cci_score,
    pc.medication_count,
    pc.high_risk_med_count,
    pc.med_reconciled,
    mp.risk_score,
    mp.risk_tier,
    mp.actual_readmit,
    CASE
        WHEN pc.high_risk_med_count >= 2 AND mp.risk_score >= 0.60
            THEN 'Pharmacist Review Required'
        WHEN pc.med_reconciled = FALSE AND mp.risk_score >= 0.40
            THEN 'Medication Reconciliation Needed'
        WHEN mp.risk_score >= 0.60
            THEN 'High Risk — Discharge Planning'
        ELSE 'Monitor'
    END AS pharmacy_action
FROM patient_cohort pc
JOIN model_predictions mp ON pc.patient_id = mp.patient_id
WHERE mp.risk_tier IN ('High','Very High')
ORDER BY mp.risk_score DESC;
