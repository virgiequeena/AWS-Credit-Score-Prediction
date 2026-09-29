from data_ingestion import ingest_data
from trainPipeline import PreprocessingPipeline, TrainingPipeline
from evaluationPipeline import EvaluationPipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd

F1_THRESHOLD = 0.70   #classification threshold

def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()

    df = pd.read_csv("ingested/credit_data.csv", sep=",")

    print("Step 2: Preprocessing")
    preprocessor = PreprocessingPipeline()
    df = preprocessor.clean_data(df)

    x = df.drop(['Credit_Score'], axis=1)
    y = df["Credit_Score"]   #classification target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    joblib.dump(le, "artifacts/label_encoder.pkl")

    num_cols = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    preprocess = preprocessor.build_preprocessor(num_cols)

    print("Step 3: Training Classifiers")
    trainer = TrainingPipeline(preprocess)
    run_ids = trainer.train(x_train, y_train_enc)

    print("Step 4: Evaluation")
    evaluator = EvaluationPipeline()
    results = {}
    for name, run_id in run_ids.items():
        acc, prec, rec, f1 = evaluator.evaluate_classifier(x_test, y_test_enc, run_id, name)
        results[name] = f1

    best_model = max(results, key=results.get)

    if results[best_model] >= F1_THRESHOLD:
        print(f"Best model: {best_model} (F1={results[best_model]:.3f}) approved for deployment")
    else:
        print("Models rejected")

if __name__ == "__main__":
    run_pipeline()
