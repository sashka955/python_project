from typing import List, Dict, Any
import os
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import pandas as pd


class HousePricePredictor:
    def __init__(self, train_data_path: str, test_data_path: str):
        """
        Initialize the predictor class with paths to the training and testing datasets.
        """
        self.train_data = pl.read_csv(train_data_path)
        self.test_data = pl.read_csv(test_data_path)
        self.models = {}
        self.output_folder = "src/real_estate_toolkit/ml_models/outputs"
        os.makedirs(self.output_folder, exist_ok=True)

    def clean_data(self):
        """
        Perform comprehensive data cleaning on the training and testing datasets.
        """
        def handle_missing_values(df: pl.DataFrame):
            return df.fill_null(strategy="mean")

        # Clean training and test data
        self.train_data = handle_missing_values(self.train_data)
        self.test_data = handle_missing_values(self.test_data)

    def prepare_features(self, target_column: str = 'SalePrice', selected_predictors: List[str] = None):
        """
        Prepare the dataset for machine learning by separating features and the target variable.
        """
        # Convert Polars DataFrame to Pandas for scikit-learn compatibility
        train_df = self.train_data.to_pandas()
        test_df = self.test_data.to_pandas()

        # Separate features and target
        y = train_df[target_column]
        X = train_df.drop(columns=[target_column])

        if selected_predictors:
            X = X[selected_predictors]

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Define preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return preprocessor, X_train, X_test, y_train, y_test

    def train_baseline_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate baseline machine learning models for house price prediction.
        """
        # Prepare data
        preprocessor, X_train, X_test, y_train, y_test = self.prepare_features()

        results = {}

        # Define models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Train the model
            pipeline.fit(X_train, y_train)

            # Evaluate on training and test sets
            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)

            metrics = {
                "MSE_train": mean_squared_error(y_train, y_train_pred),
                "MSE_test": mean_squared_error(y_test, y_test_pred),
                "R2_train": r2_score(y_train, y_train_pred),
                "R2_test": r2_score(y_test, y_test_pred),
                "MAE_train": mean_absolute_error(y_train, y_train_pred),
                "MAE_test": mean_absolute_error(y_test, y_test_pred),
                "MAPE_train": mean_absolute_percentage_error(y_train, y_train_pred),
                "MAPE_test": mean_absolute_percentage_error(y_test, y_test_pred),
            }

            results[model_name] = {"metrics": metrics, "model": pipeline}
            self.models[model_name] = pipeline

        return results

    def forecast_sales_price(self, model_type: str = 'Linear Regression') -> None:
        """
        Use the trained model to forecast house prices on the test dataset.
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} has not been trained.")

        # Use the selected model to make predictions
        model = self.models[model_type]
        test_df = self.test_data.to_pandas()
        X_test = test_df.drop(columns=["Id"])  # Assuming "Id" is not a feature
        predictions = model.predict(X_test)

        # Create submission file
        submission = pd.DataFrame({
            "Id": test_df["Id"],
            "SalePrice": predictions
        })
        submission_path = os.path.join(self.output_folder, "submission.csv")
        submission.to_csv(submission_path, index=False)

        print(f"Predictions saved to {submission_path}")
