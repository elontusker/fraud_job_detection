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
