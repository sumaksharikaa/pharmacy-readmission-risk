"""
train_models.py
Pharmacy Readmission Risk Predictor — ML Training Pipeline
Models: Logistic Regression → Random Forest → XGBoost
Tracking: MLflow (local)
"""

import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (roc_auc_score, accuracy_score, f1_score,
                                       precision_score, recall_score,
                                       confusion_matrix, roc_curve,
                                       average_precision_score)
from sklearn.pipeline          import Pipeline
from sklearn.calibration       import CalibratedClassifierCV
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    "age", "cci_score", "n_conditions", "length_of_stay",
    "prior_admissions_6m", "medication_count", "high_risk_med_count",
    "discharge_med_count", "med_reconciled", "lab_abnormal_count",
    "sodium", "creatinine", "hemoglobin", "wbc", "bmi",
    "gender_M", "insurance_Medicare", "insurance_Medicaid", "insurance_Self-Pay",
]
TARGET = "readmit_30day"

# ── load & engineer features ──────────────────────────────────────────────────
def load_features():
    df = pd.read_csv(f"{DATA_DIR}/patient_cohort.csv")

    # One-hot encode categoricals
    df["gender_M"]               = (df["gender"] == "M").astype(int)
    df["insurance_Medicare"]     = (df["insurance_type"] == "Medicare").astype(int)
    df["insurance_Medicaid"]     = (df["insurance_type"] == "Medicaid").astype(int)
    df["insurance_Self-Pay"]     = (df["insurance_type"] == "Self-Pay").astype(int)

    # CCI risk tier
    df["cci_risk_tier"] = pd.cut(df["cci_score"],
                                  bins=[-1,0,2,4,100],
                                  labels=["Low","Moderate","High","Very High"])

    X = df[FEATURES].fillna(0)
    y = df[TARGET]
    return df, X, y


def evaluate(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    metrics = {
        "roc_auc":          round(roc_auc_score(y_test, y_proba), 4),
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "f1":               round(f1_score(y_test, y_pred), 4),
        "precision":        round(precision_score(y_test, y_pred), 4),
        "recall":           round(recall_score(y_test, y_pred), 4),
        "avg_precision":    round(average_precision_score(y_test, y_proba), 4),
    }
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm          = confusion_matrix(y_test, y_pred)
    print(f"\n  {name}")
    print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    return metrics, fpr.tolist(), tpr.tolist(), cm.tolist(), y_proba.tolist()


def train():
    print("="*55)
    print("PHARMACY READMISSION RISK — MODEL TRAINING")
    print("="*55)

    df, X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    print(f"\nDataset: {len(df):,} patients | "
          f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Readmission rate: {y.mean()*100:.1f}% positive class")

    mlflow.set_experiment("pharmacy_readmission_risk")
    all_results = []

    # ── 1. Logistic Regression ────────────────────────────────────────────────
    with mlflow.start_run(run_name="logistic_regression"):
        params = {"C":0.5, "max_iter":500, "random_state":42}
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(**params, class_weight="balanced"))
        ])
        pipe.fit(X_train, y_train)
        metrics, fpr, tpr, cm, proba = evaluate("Logistic Regression", pipe, X_test, y_test)
        cv = cross_val_score(pipe, X, y, cv=StratifiedKFold(5), scoring="roc_auc")

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_roc_auc_mean", round(cv.mean(), 4))
        mlflow.log_metric("cv_roc_auc_std",  round(cv.std(), 4))
        mlflow.sklearn.log_model(pipe, "logistic_regression")
        joblib.dump(pipe, f"{MODELS_DIR}/logistic_regression.pkl")

        # Feature importance (coefficients)
        coefs = pipe.named_steps["model"].coef_[0]
        feat_imp = sorted(zip(FEATURES, coefs), key=lambda x: abs(x[1]), reverse=True)

        all_results.append({"model":"Logistic Regression","metrics":metrics,
                              "fpr":fpr,"tpr":tpr,"cm":cm,"proba":proba,
                              "feature_importance":{f:float(v) for f,v in feat_imp},
                              "cv_mean":round(cv.mean(),4),"cv_std":round(cv.std(),4)})

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    with mlflow.start_run(run_name="random_forest"):
        params = {"n_estimators":200, "max_depth":8, "min_samples_leaf":10,
                  "random_state":42, "class_weight":"balanced", "n_jobs":-1}
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        metrics, fpr, tpr, cm, proba = evaluate("Random Forest", rf, X_test, y_test)
        cv = cross_val_score(rf, X, y, cv=StratifiedKFold(5), scoring="roc_auc")

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_roc_auc_mean", round(cv.mean(), 4))
        mlflow.sklearn.log_model(rf, "random_forest")
        joblib.dump(rf, f"{MODELS_DIR}/random_forest.pkl")

        feat_imp = sorted(zip(FEATURES, rf.feature_importances_),
                          key=lambda x: x[1], reverse=True)
        all_results.append({"model":"Random Forest","metrics":metrics,
                              "fpr":fpr,"tpr":tpr,"cm":cm,"proba":proba,
                              "feature_importance":{f:float(v) for f,v in feat_imp},
                              "cv_mean":round(cv.mean(),4),"cv_std":round(cv.std(),4)})

    # ── 3. XGBoost ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="xgboost"):
        scale_pos = (y_train==0).sum() / (y_train==1).sum()
        params = {"n_estimators":300, "max_depth":5, "learning_rate":0.05,
                  "subsample":0.8, "colsample_bytree":0.8,
                  "scale_pos_weight":scale_pos, "random_state":42,
                  "eval_metric":"auc", "use_label_encoder":False}
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)], verbose=False)
        metrics, fpr, tpr, cm, proba = evaluate("XGBoost", xgb_model, X_test, y_test)
        cv = cross_val_score(xgb_model, X, y, cv=StratifiedKFold(5), scoring="roc_auc")

        mlflow.log_params({k:v for k,v in params.items() if k != "use_label_encoder"})
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_roc_auc_mean", round(cv.mean(), 4))
        mlflow.xgboost.log_model(xgb_model, "xgboost")
        joblib.dump(xgb_model, f"{MODELS_DIR}/xgboost.pkl")

        feat_imp = sorted(zip(FEATURES, xgb_model.feature_importances_),
                          key=lambda x: x[1], reverse=True)
        all_results.append({"model":"XGBoost","metrics":metrics,
                              "fpr":fpr,"tpr":tpr,"cm":cm,"proba":proba,
                              "feature_importance":{f:float(v) for f,v in feat_imp},
                              "cv_mean":round(cv.mean(),4),"cv_std":round(cv.std(),4)})

    # ── Save results for dashboard ────────────────────────────────────────────
    with open(f"{MODELS_DIR}/results.json", "w") as f:
        json.dump(all_results, f)

    # Save patient-level risk scores (XGBoost — best model)
    df_out = df.copy()
    xgb_final = joblib.load(f"{MODELS_DIR}/xgboost.pkl")
    df_out["risk_score"]    = xgb_final.predict_proba(X.fillna(0))[:,1].round(4)
    df_out["risk_tier"]     = pd.cut(df_out["risk_score"],
                                      bins=[0,.20,.40,.60,1.0],
                                      labels=["Low","Moderate","High","Very High"])
    df_out["predicted_readmit"] = (df_out["risk_score"] >= 0.50).astype(int)
    df_out.to_csv(f"{DATA_DIR}/risk_scores.csv", index=False)

    # Summary
    print(f"\n{'='*55}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"{'Model':<22} {'ROC-AUC':>8} {'F1':>8} {'CV AUC':>8}")
    print("-"*50)
    for r in all_results:
        print(f"  {r['model']:<20} {r['metrics']['roc_auc']:>8.4f} "
              f"{r['metrics']['f1']:>8.4f} {r['cv_mean']:>8.4f}")
    print(f"{'='*55}")
    print(f"\nRisk scores saved → data/risk_scores.csv")
    print(f"Models saved       → models/")
    print(f"MLflow results     → mlruns/\n")


if __name__ == "__main__":
    train()
