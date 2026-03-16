"""
Automated Predictive Pipeline for Tabular Data
End-to-end implementation including preprocessing, feature engineering, and training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
from typing import Dict, Any, Union

class PredictivePipeline:
    def __init__(self, target_column: str = "target"):
        self.target_column = target_column
        self.model = None
        self.pipeline = None

    def build_pipeline(self, numerical_cols: list, categorical_cols: list) -> Pipeline:
        """
        Build a Scikit-Learn preprocessing pipeline.
        """
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
        ])
        
        return clf

    def train(self, data: pd.DataFrame, numerical_cols: list, categorical_cols: list):
        """
        Train the pipeline on a given dataset.
        """
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.pipeline = self.build_pipeline(numerical_cols, categorical_cols)
        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Report:\n{report}")

    def save_model(self, path: str = "models/model_v1.pkl"):
        """
        Persist the model for production use.
        """
        joblib.dump(self.pipeline, path)

    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Predict on new data.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not trained or loaded.")
        return self.pipeline.predict(input_data)

if __name__ == "__main__":
    # Example usage with mock data
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    })
    
    pipeline = PredictivePipeline()
    pipeline.train(df, numerical_cols=['feature1'], categorical_cols=['feature2'])
    print("Predictive Pipeline Initialized and Trained on Mock Data.")
