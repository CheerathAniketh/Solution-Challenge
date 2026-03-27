from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def prepare_features(df, target_col, sensitive_col):
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(
            df_encoded[col].astype(str))
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    return X, y


def train_and_evaluate(df, target_col, sensitive_col):
    X, y = prepare_features(df, target_col, sensitive_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_test