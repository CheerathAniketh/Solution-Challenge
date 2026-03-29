from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np

def prepare_features(df, target_col, sensitive_col):
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        if col != sensitive_col:  # ← DON'T encode sensitive col here
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    # encode target separately
    if df_encoded[target_col].dtype == object:
        df_encoded[target_col] = le.fit_transform(df_encoded[target_col].astype(str))
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    return X, y
def safe_float(x):
    """Convert to float, replacing nan/inf with None."""
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return round(v, 3)
    except:
        return None

def get_roc_data(y_true, y_prob, sensitive_values, sensitive_test):
    """Compute ROC curves per group."""
    groups = sorted(sensitive_test.unique())
    roc_data = {}
    for group in groups:
        mask = sensitive_test == group
        if mask.sum() < 10:
            continue
        if y_true[mask].sum() == 0:  # no positive samples in this group
            continue
        fpr, tpr, _ = roc_curve(y_true[mask], y_prob[mask])
        roc_auc = round(auc(fpr, tpr), 3)
        # downsample to 20 points for frontend
        idx = np.linspace(0, len(fpr) - 1, min(20, len(fpr)), dtype=int)
        roc_data[str(group)] = {
            "fpr": [safe_float(x) for x in fpr[idx]],
            "tpr": [safe_float(x) for x in tpr[idx]],
            "auc": roc_auc
        }
    return roc_data


def get_calibration_data(y_true, y_prob, sensitive_values, sensitive_test):
    """Compute calibration curves per group."""
    groups = sorted(sensitive_test.unique())
    calib_data = {}
    for group in groups:
        mask = sensitive_test == group
        if mask.sum() < 10:
            continue
        fraction_of_positives, mean_predicted = calibration_curve(
            y_true[mask], y_prob[mask], n_bins=10, strategy='uniform'
        )
        calib_data[str(group)] = {
            "mean_predicted": [safe_float(x) for x in mean_predicted],
            "fraction_positive": [safe_float(x) for x in fraction_of_positives],
        }
    return calib_data


def train_and_evaluate(df, target_col, sensitive_col):
    # Keep original sensitive values BEFORE encoding (for curve labels)
    sensitive_original = df[sensitive_col].astype(str) if sensitive_col in df.columns else None

    X, y = prepare_features(df, target_col, sensitive_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Drop sensitive col from training features to avoid leakage
    X_train_model = X_train.drop(columns=[sensitive_col], errors='ignore')
    X_test_model = X_test.drop(columns=[sensitive_col], errors='ignore')

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_model, y_train)

    # Compute real ROC + calibration curves using original string group names
    y_prob = model.predict_proba(X_test_model)[:, 1]
    curves = {}
    if sensitive_original is not None:
        sensitive_test = sensitive_original.iloc[y_test.index].reset_index(drop=True)
        y_test_reset = y_test.reset_index(drop=True)
        curves["roc"] = get_roc_data(y_test_reset, y_prob, sensitive_test.unique(), sensitive_test)
        curves["calibration"] = get_calibration_data(y_test_reset, y_prob, sensitive_test.unique(), sensitive_test)
    y_pred = model.predict(X_test_model)
    sensitive_test_series = sensitive_original.iloc[y_test.index].reset_index(drop=True) if sensitive_original is not None else None
    return model, X_train_model, X_test_model, y_test, curves, y_pred, y_prob, sensitive_test_series

