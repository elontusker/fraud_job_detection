import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import re


def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", str(text))
    text = text.lower()
    return text


def preprocess_data(df):
    # Handle missing values
    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
    for col in text_cols:
        df[col] = df[col].fillna("")
        df[col] = df[col].apply(clean_text)

    # Convert binary columns to numeric (if they exist)
    binary_cols = ["telecommuting", "has_company_logo", "has_questions"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Convert target to binary if it exists (for training data)
    if "fraudulent" in df.columns:
        df["fraudulent"] = df["fraudulent"].astype(int)

    return df


def train_model():
    # Load data
    df = pd.read_csv("data/training_data.csv")
    df = preprocess_data(df)

    # Split data
    X = df.drop(["fraudulent", "job_id"], axis=1)  # Exclude ID column
    y = df["fraudulent"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate class weights
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Define preprocessing
    text_transformer = TfidfVectorizer(max_features=1000, stop_words="english")
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("title", text_transformer, "title"),
            ("profile", text_transformer, "company_profile"),
            ("desc", text_transformer, "description"),
            ("req", text_transformer, "requirements"),
            ("benefits", text_transformer, "benefits"),
            (
                "binary",
                "passthrough",
                ["telecommuting", "has_company_logo", "has_questions"],
            ),
            (
                "cat",
                categorical_transformer,
                [
                    "employment_type",
                    "required_experience",
                    "required_education",
                    "industry",
                    "function",
                ],
            ),
        ]
    )

    # Create pipeline
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight=class_weights,
                    random_state=42,
                    n_estimators=200,
                    max_depth=10,
                ),
            ),
        ]
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    # Save model
    joblib.dump(model, "model.joblib")

    return model


def predict_new_data(model, new_data):
    # Keep job_id for reference
    job_ids = new_data["job_id"].copy()

    # Preprocess new data - handle missing fraudulent column
    new_data = preprocess_data(new_data)

    # Drop columns that might not exist in test data
    columns_to_drop = ["fraudulent", "job_id"]
    columns_to_drop = [col for col in columns_to_drop if col in new_data.columns]
    X_new = new_data.drop(columns_to_drop, axis=1)

    # Predict
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]

    # Create results dataframe
    results = new_data.copy()
    results["job_id"] = job_ids
    results["fraud_prediction"] = predictions
    results["fraud_probability"] = probabilities

    # Sort by most suspicious first
    results = results.sort_values("fraud_probability", ascending=False)

    return results
