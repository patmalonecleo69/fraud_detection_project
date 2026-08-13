import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FEATURE_COLS_EXCLUDE = {"transaction_id", "customer_id", "Time", "Class", "hour_of_day"}


def time_based_split(df: pd.DataFrame, test_frac: float = 0.25):
    df_sorted = df.sort_values("Time")
    cutoff_idx = int(len(df_sorted) * (1 - test_frac))
    cutoff_time = df_sorted.iloc[cutoff_idx]["Time"]
    train = df_sorted[df_sorted["Time"] < cutoff_time].copy()
    test = df_sorted[df_sorted["Time"] >= cutoff_time].copy()
    return train, test


def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]


def prepare_xy(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].values
    y = df["Class"].values
    return X, y


def train_all_models(train_df: pd.DataFrame, feature_cols, random_state=42):
    X_train, y_train = prepare_xy(train_df, feature_cols)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    smote = SMOTE(random_state=random_state, sampling_strategy=0.15)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE: {len(y_res)} samples, "
          f"{y_res.sum()} fraud ({100*y_res.mean():.2f}%)")

    models = {}

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    lr.fit(X_res, y_res)
    models["Logistic Regression"] = lr

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=random_state
    )
    rf.fit(X_res, y_res)
    models["Random Forest"] = rf

    if HAS_XGB:
        scale_pos_weight = (y_res == 0).sum() / max((y_res == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", random_state=random_state, n_jobs=-1
        )
        xgb.fit(X_res, y_res)
        models["XGBoost"] = xgb

    return models, scaler


def evaluate_models(models: dict, scaler: StandardScaler, test_df: pd.DataFrame, feature_cols):
    X_test, y_test = prepare_xy(test_df, feature_cols)
    X_test_scaled = scaler.transform(X_test)

    results = []
    predictions = {}
    for name, model in models.items():
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        results.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, y_proba),
            "avg_precision": average_precision_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        })
        predictions[name] = (y_test, y_proba, y_pred)

    return pd.DataFrame(results).sort_values("avg_precision", ascending=False), predictions


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/fraud_detection/data/features.csv")
    train_df, test_df = time_based_split(df)
    print(f"Train: {len(train_df)} ({train_df['Class'].sum()} fraud) | "
          f"Test: {len(test_df)} ({test_df['Class'].sum()} fraud)")

    feature_cols = get_feature_columns(df)
    models, scaler = train_all_models(train_df, feature_cols)
    results_df, predictions = evaluate_models(models, scaler, test_df, feature_cols)
    print("\n=== Model Comparison (sorted by Average Precision) ===")
    print(results_df.to_string(index=False))

    results_df.to_csv("/home/claude/fraud_detection/outputs/model_comparison.csv", index=False)
    joblib.dump({"models": models, "scaler": scaler, "feature_cols": feature_cols},
                "/home/claude/fraud_detection/outputs/trained_models.joblib")
    joblib.dump(predictions, "/home/claude/fraud_detection/outputs/predictions.joblib")
    print("\nSaved model comparison, trained models, and predictions to outputs/")
