"""SageMaker inference entry point.

Generic sklearn Pipeline loader. Works for any pipeline saved as model_credit.joblib,
regardless of which classifier is inside.

Four functions form the SageMaker contract:
    model_fn   - load model from disk (called once per container)
    input_fn   - parse request body (called per request)
    predict_fn - run inference (called per request)
    output_fn  - serialize response (called per request)
"""


import json
import os

import joblib
import numpy as np
import pandas as pd


JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"

FEATURE_NAMES = [
    "Month",
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Num_Loan_Types",
    "Credit_Mix",
    "Occupation",
    "Payment_of_Min_Amount",
    "Payment_Behaviour"
]

NUMERIC_COLS = {
    "Month", "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries","Outstanding_Debt", "Credit_Utilization_Ratio",
    "Credit_History_Age","Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance", "Num_Loan_Types"
}


def model_fn(model_dir: str):
    """Load the pickled sklearn Pipeline and the label encoder."""
    model = joblib.load(os.path.join(model_dir, "model_credit.joblib"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    return {"model": model, "label_encoder": label_encoder}


def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    """Parse incoming request body into a DataFrame.

    Accepts JSON: {"instances": [[month, age, annual_income, ...]]}
    Or CSV: one row per instance, values in FEATURE_NAMES order.
    """
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        return pd.DataFrame(instances, columns=FEATURE_NAMES)

    if request_content_type == CSV_CONTENT_TYPE:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        rows = []
        for line in request_body.strip().splitlines():
            if not line.strip():
                continue
            raw = line.split(",")
            typed_row = [
                float(v.strip()) if name in NUMERIC_COLS else v.strip()
                for name, v in zip(FEATURE_NAMES, raw)
            ]
            rows.append(typed_row)
        return pd.DataFrame(rows, columns=FEATURE_NAMES)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: pd.DataFrame, artifacts) -> dict:
    """Run inference. Returns probabilities, predicted class IDs, and labels."""
    model = artifacts["model"]
    label_encoder = artifacts["label_encoder"]
    probs = model.predict_proba(input_data)
    class_ids = np.argmax(probs, axis=1)
    labels = label_encoder.inverse_transform(class_ids)
    return {
        "probabilities": probs.tolist(),
        "predictions": class_ids.tolist(),
        "labels": labels.tolist(),
    }


def output_fn(prediction: dict, accept_content_type: str):
    """Serialize the prediction dict for the response body."""
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")
