import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

import os
os.makedirs("artifacts", exist_ok=True)


class PreprocessingPipeline:
    def __init__(self):
        self.preprocess = None

    def clean_data(self, df):
        df = df.copy()

        str_numeric = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
                       'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance']
        for col in str_numeric:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('_', '', regex=False), errors='coerce')

        bounds = {
            'Age': (14, 100),
            'Num_Bank_Accounts': (0, 20),
            'Num_Credit_Card': (0, 20),
            'Interest_Rate': (1, 50),
            'Num_of_Loan': (0, 15),
            'Num_of_Delayed_Payment': (0, 50),
            'Num_Credit_Inquiries': (0, 50)
        }
        for col, (lo, hi) in bounds.items():
            df[col] = df[col].where(df[col].between(lo, hi), np.nan)

        df['Occupation'] = df['Occupation'].replace('_______', np.nan)
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', np.nan)
        df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)

        def parse_history(value):
            if pd.isna(value):
                return np.nan
            parts = str(value).split()
            years = int(parts[0])
            months = int(parts[3])
            return years * 12 + months
        df['Credit_History_Age'] = df['Credit_History_Age'].apply(parse_history)

        def count_loan_types(value):
            if pd.isna(value) or str(value).strip() == '':
                return 0
            loans = str(value).replace(' and ', ',').split(',')
            return len([loan for loan in loans if loan.strip()])
        df['Num_Loan_Types'] = df['Type_of_Loan'].apply(count_loan_types)

        month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        df['Month'] = df['Month'].map(month_map)

        df = df.drop(columns=['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan'])

        return df

    def build_preprocessor(self, num_cols):
        #For imputing missing value & encoding numerical features and categorical
        numeric_preprocess = Pipeline([('num_imputer', SimpleImputer(strategy='median')),
                                       ('scaler', StandardScaler())])

        categorical_preprocess = Pipeline([
            ('cat_imputer', SimpleImputer(strategy='most_frequent')),
            ('cat_encoder', OrdinalEncoder(categories=[
                ['Bad', 'Standard', 'Good'],    # Credit_Mix
            ], handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        ohe_preprocess = Pipeline([
            ('cat_imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocess = ColumnTransformer(transformers=[
            ('numPreprocess', numeric_preprocess, num_cols),
            ('catPreprocess', categorical_preprocess, ['Credit_Mix']),
            ('ohePreprocess', ohe_preprocess, ['Occupation', 'Payment_of_Min_Amount', 'Payment_Behaviour'])
        ], remainder='drop')

        return self.preprocess


class TrainingPipeline:
    def __init__(self, preprocess):
        self.__preprocess = preprocess
        self.__is_trained = False
        self.__run_ids = {}

    @property
    def run_ids(self):
        if not self.__is_trained:
            raise ValueError("You must train the models before accessing run IDs!")
        return self.__run_ids

    def train(self, x_train, y_train):
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                                   class_weight='balanced', random_state=42, n_jobs=-1)
        }

        #MLflow Tracking
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Credit Score Classification")

        for name, model in models.items():
            credit_pred = Pipeline([
                ('preprocessing', self.__preprocess),
                ('classifier', model)
            ])

            with mlflow.start_run(run_name=name) as run:
                #log parameters
                mlflow.log_param("model", name)
                for param, value in model.get_params().items():
                    mlflow.log_param(param, value)

                #train
                credit_pred.fit(x_train, y_train)

                #save and log model
                joblib.dump(credit_pred, f"artifacts/{name}_model.pkl")
                mlflow.sklearn.log_model(credit_pred, artifact_path=f"{name}_model", serialization_format="cloudpickle")

                self.__run_ids[name] = run.info.run_id

        self.__is_trained = True
        return self.__run_ids
