from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import io

app = FastAPI(title="Credit Risk API")

# CORS must be first, before anything else
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Explicit OPTIONS handler as safety net
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str, request: Request):
    return JSONResponse(
        content="OK",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )

xgb_model = joblib.load("xgb_model.pkl")
iso_model = joblib.load("iso_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

ANOMALY_FEATURES = [
    'revolving_utilization', 'age', 'monthly_income', 'debt_ratio',
    'open_credit_lines', 'real_estate_loans', 'dependents',
    'total_delinquencies', 'dti_ratio', 'high_utilization'
]

COLUMN_MAP = {
    'SeriousDlqin2yrs': 'default_val',
    'RevolvingUtilizationOfUnsecuredLines': 'revolving_utilization',
    'NumberOfTime30-59DaysPastDueNotWorse': 'past_due_30_59',
    'DebtRatio': 'debt_ratio',
    'MonthlyIncome': 'monthly_income',
    'NumberOfOpenCreditLinesAndLoans': 'open_credit_lines',
    'NumberOfTimes90DaysLate': 'times_90_days_late',
    'NumberRealEstateLoansOrLines': 'real_estate_loans',
    'NumberOfTime60-89DaysPastDueNotWorse': 'past_due_60_89',
    'NumberOfDependents': 'dependents',
}

def engineer_features(df):
    df = df.copy()
    df.rename(columns=COLUMN_MAP, inplace=True)
    if 'monthly_income' in df.columns:
        df['monthly_income'].fillna(df['monthly_income'].median(), inplace=True)
    if 'dependents' in df.columns:
        df['dependents'].fillna(0, inplace=True)
    df['monthly_debt'] = df.get('debt_ratio', 0) * df.get('monthly_income', 0)
    df['dti_ratio'] = np.where(df.get('monthly_income', 1) > 0,
                               df['monthly_debt'] / df.get('monthly_income', 1), 0)
    df['total_delinquencies'] = (
        df.get('past_due_30_59', 0) +
        df.get('past_due_60_89', 0) +
        df.get('times_90_days_late', 0)
    )
    df['has_delinquency'] = (df['total_delinquencies'] > 0).astype(int)
    df['high_utilization'] = (df.get('revolving_utilization', 0) > 0.8).astype(int)
    df['young_borrower'] = (df.get('age', 30) < 25).astype(int)
    return df

def get_risk_tier(prob):
    if prob < 0.05: return "Low Risk"
    elif prob < 0.15: return "Medium Risk"
    return "High Risk"

def get_decision(risk_tier, fraud_flag):
    if fraud_flag: return "Review"
    if risk_tier == "High Risk": return "Reject"
    if risk_tier == "Medium Risk": return "Approve with Conditions"
    return "Approve"

def get_fraud_signals(row):
    signals = []
    if row.get('monthly_income', 0) > 15000 and row.get('revolving_utilization', 0) > 0.7:
        signals.append("High income + high utilization")
    if row.get('age', 30) < 22 and row.get('open_credit_lines', 0) > 5:
        signals.append("Identity theft signal")
    if row.get('real_estate_loans', 0) > 3 and row.get('total_delinquencies', 0) > 2:
        signals.append("Loan stacking")
    if row.get('monthly_income', 1) % 1000 == 0 and row.get('monthly_income', 0) > 0:
        signals.append("Round income flag")
    return signals

def process_dataframe(df):
    df = engineer_features(df)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].fillna(0)
    X_anomaly = df[ANOMALY_FEATURES].fillna(0)
    probs = xgb_model.predict_proba(X)[:, 1]
    fraud_flags = (iso_model.predict(X_anomaly) == -1).astype(int)
    rule_flags = []
    for _, row in df.iterrows():
        signals = get_fraud_signals(row)
        rule_flags.append(1 if len(signals) >= 2 else 0)
    combined_fraud = np.array([max(f, r) for f, r in zip(fraud_flags, rule_flags)])
    risk_tiers = [get_risk_tier(p) for p in probs]
    decisions = [get_decision(t, f) for t, f in zip(risk_tiers, combined_fraud)]
    return probs, combined_fraud, risk_tiers, decisions, df

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze/csv")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only CSV files supported")
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents), index_col=0)
    if len(df) > 5000:
        df = df.head(5000)
    probs, fraud_flags, risk_tiers, decisions, processed = process_dataframe(df)
    tier_counts = {}
    for t in risk_tiers:
        tier_counts[t] = tier_counts.get(t, 0) + 1
    decision_counts = {}
    for d in decisions:
        decision_counts[d] = decision_counts.get(d, 0) + 1
    buckets = [0] * 10
    for p in probs:
        idx = min(int(p * 10), 9)
        buckets[idx] += 1
    rows = []
    for i in range(min(200, len(df))):
        rows.append({
            "id": i + 1,
            "default_probability": round(float(probs[i]) * 100, 1),
            "risk_tier": risk_tiers[i],
            "fraud_flag": int(fraud_flags[i]),
            "decision": decisions[i],
            "age": int(processed['age'].iloc[i]) if 'age' in processed.columns else None,
            "monthly_income": round(float(processed['monthly_income'].iloc[i]), 0) if 'monthly_income' in processed.columns else None,
            "revolving_utilization": round(float(processed['revolving_utilization'].iloc[i]) * 100, 1) if 'revolving_utilization' in processed.columns else None,
        })
    return {
        "total_borrowers": len(df),
        "tier_counts": tier_counts,
        "decision_counts": decision_counts,
        "fraud_flagged": int(fraud_flags.sum()),
        "avg_default_prob": round(float(probs.mean()) * 100, 1),
        "prob_distribution": buckets,
        "borrowers": rows
    }

class SingleBorrower(BaseModel):
    age: int
    monthly_income: float
    revolving_utilization: float
    debt_ratio: float
    open_credit_lines: int
    real_estate_loans: int
    dependents: int
    past_due_30_59: int
    past_due_60_89: int
    times_90_days_late: int

@app.post("/analyze/single")
def analyze_single(borrower: SingleBorrower):
    df = pd.DataFrame([borrower.dict()])
    probs, fraud_flags, risk_tiers, decisions, processed = process_dataframe(df)
    signals = get_fraud_signals(processed.iloc[0].to_dict())
    return {
        "default_probability": round(float(probs[0]) * 100, 1),
        "risk_tier": risk_tiers[0],
        "fraud_flag": int(fraud_flags[0]),
        "fraud_signals": signals,
        "decision": decisions[0],
    }
