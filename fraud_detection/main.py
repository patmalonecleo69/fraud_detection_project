import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import joblib

from generate_data import generate_transactions
from feature_engineering import build_feature_matrix
from modeling import (
    time_based_split, get_feature_columns, train_all_models, evaluate_models
)
import visualize as viz

DATA_DIR = "data"
OUT_DIR = "outputs"
FIG_DIR = "figures"


def main():
    for d in (DATA_DIR, OUT_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("STAGE 1: Generating synthetic transaction data")
    print("=" * 60)
    raw_df = generate_transactions()
    raw_df.to_csv(f"{DATA_DIR}/transactions.csv", index=False)
    print(f"-> {len(raw_df):,} transactions, {raw_df['Class'].sum()} fraud "
          f"({100 * raw_df['Class'].mean():.3f}%)\n")

    print("=" * 60)
    print("STAGE 2: Feature engineering")
    print("=" * 60)
    feat_df = build_feature_matrix(raw_df)
    feat_df.to_csv(f"{DATA_DIR}/features.csv", index=False)
    print(f"-> Feature matrix: {feat_df.shape[0]:,} rows x {feat_df.shape[1]} columns\n")

    print("=" * 60)
    print("STAGE 3: Model training & evaluation")
    print("=" * 60)
    train_df, test_df = time_based_split(feat_df)
    print(f"-> Train: {len(train_df):,} ({train_df['Class'].sum()} fraud) | "
          f"Test: {len(test_df):,} ({test_df['Class'].sum()} fraud)")

    feature_cols = get_feature_columns(feat_df)
    models, scaler = train_all_models(train_df, feature_cols)
    results_df, predictions = evaluate_models(models, scaler, test_df, feature_cols)

    print("\nModel comparison (sorted by Average Precision):")
    print(results_df.to_string(index=False))

    results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
    joblib.dump({"models": models, "scaler": scaler, "feature_cols": feature_cols},
                f"{OUT_DIR}/trained_models.joblib")
    joblib.dump(predictions, f"{OUT_DIR}/predictions.joblib")
    print(f"\n-> Saved model comparison + artifacts to {OUT_DIR}/\n")

    print("=" * 60)
    print("STAGE 4: Generating figures")
    print("=" * 60)
    viz.fig01_class_imbalance(feat_df)
    viz.fig02_amount_distribution(feat_df)
    viz.fig03_hourly_fraud_rate(feat_df)
    viz.fig04_correlation_heatmap(feat_df)
    viz.fig05_roc_curves(predictions)
    viz.fig06_pr_curves(predictions, feat_df)
    viz.fig07_confusion_matrices(predictions)
    viz.fig08_feature_importance(models, feature_cols)
    viz.fig09_model_comparison_bars(results_df)
    viz.fig10_threshold_tradeoff(predictions, results_df.iloc[0]["model"])
    print(f"-> Saved 10 figures to {FIG_DIR}/\n")

    best = results_df.iloc[0]
    print("=" * 60)
    print(f"DONE. Best model: {best['model']} "
          f"(Average Precision={best['avg_precision']:.3f}, "
          f"ROC-AUC={best['roc_auc']:.3f}, Recall={best['recall']:.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
