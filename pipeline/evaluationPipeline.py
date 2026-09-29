import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR #/ "data" / "processed"


class EvaluationPipeline:
    def evaluate_classifier(self, x_test, y_test, run_id, model_name):
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}_model")

        preds = model.predict(x_test)
        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro")
        rec  = recall_score(y_test, preds, average="macro")
        f1   = f1_score(y_test, preds, average="macro")

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("accuracy",  acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall",    rec)
            mlflow.log_metric("f1_score",  f1)

        print(f"{model_name} Evaluation Completed | Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}")
        return acc, prec, rec, f1
