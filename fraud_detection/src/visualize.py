import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

sns.set_theme(style="whitegrid", palette="deep")
FIG_DIR = "figures"
DATA_DIR = "data"
OUT_DIR = "outputs"

COLORS = {"Logistic Regression": "#4C72B0", "Random Forest": "#55A868", "XGBoost": "#C44E52"}


def fig01_class_imbalance(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["Class"].value_counts().sort_index()
    bars = ax.bar(["Legitimate", "Fraud"], counts.values, color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_ylabel("Number of transactions (log scale)")
    ax.set_title(f"Class Imbalance: {100*df['Class'].mean():.3f}% Fraud Rate")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_class_imbalance.png", dpi=150)
    plt.close(fig)


def fig02_amount_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for cls, label, color in [(0, "Legitimate", "#4C72B0"), (1, "Fraud", "#C44E52")]:
        sns.kdeplot(df.loc[df.Class == cls, "Amount"].clip(upper=500), ax=axes[0],
                    label=label, color=color, fill=True, alpha=0.3)
    axes[0].set_title("Transaction Amount Distribution (clipped at $500)")
    axes[0].set_xlabel("Amount ($)")
    axes[0].legend()

    sns.boxplot(data=df, x="Class", y="Amount", ax=axes[1])
    axes[1].set_yscale("log")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Legitimate", "Fraud"])
    axes[1].set_title("Amount by Class (log scale)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_amount_distribution.png", dpi=150)
    plt.close(fig)


def fig03_hourly_fraud_rate(df):
    hourly = df.groupby("hour_of_day")["Class"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(hourly.index, hourly["mean"] * 100, color="#C44E52")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Hour of Day")
    ax.axvspan(-0.5, 5.5, alpha=0.15, color="gray", label="Night window (0-5am)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_hourly_fraud_rate.png", dpi=150)
    plt.close(fig)


def fig04_correlation_heatmap(df):
    feature_cols = [c for c in df.columns if c not in
                     {"transaction_id", "customer_id", "Time", "merchant_category"}]
    corr = df[feature_cols].corr(numeric_only=True)["Class"].drop("Class").sort_values()
    fig, ax = plt.subplots(figsize=(7, 9))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.set_title("Feature Correlation with Fraud Label")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def fig05_roc_curves(predictions):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_test, y_proba, _) in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        from sklearn.metrics import auc
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", color=COLORS.get(name))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/05_roc_curves.png", dpi=150)
    plt.close(fig)


def fig06_pr_curves(predictions, df):
    baseline = df["Class"].mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_test, y_proba, _) in predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=COLORS.get(name))
    ax.axhline(baseline, color="k", linestyle="--", alpha=0.4,
               label=f"Random baseline (AP={baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison\n(more informative than ROC under class imbalance)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/06_pr_curves.png", dpi=150)
    plt.close(fig)


def fig07_confusion_matrices(predictions):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (name, (y_test, _, y_pred)) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices (threshold = 0.5)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/07_confusion_matrices.png", dpi=150)
    plt.close(fig)


def fig08_feature_importance(models, feature_cols):
    tree_models = {k: v for k, v in models.items() if hasattr(v, "feature_importances_")}
    fig, axes = plt.subplots(1, len(tree_models), figsize=(7 * len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, tree_models.items()):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        top = importances.sort_values(ascending=True).tail(15)
        ax.barh(top.index, top.values, color=COLORS.get(name, "#4C72B0"))
        ax.set_title(f"Top 15 Feature Importances — {name}")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/08_feature_importance.png", dpi=150)
    plt.close(fig)


def fig09_model_comparison_bars(results_df):
    metrics = ["roc_auc", "avg_precision", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax.bar(x + i * width, [row[m] for m in metrics], width,
               label=row["model"], color=COLORS.get(row["model"]))
    ax.set_xticks(x + width)
    ax.set_xticklabels(["ROC-AUC", "Avg Precision", "Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/09_model_comparison_bars.png", dpi=150)
    plt.close(fig)


def fig10_threshold_tradeoff(predictions, best_model_name):
    y_test, y_proba, _ = predictions[best_model_name]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, precision[:-1], label="Precision", color="#4C72B0")
    ax.plot(thresholds, recall[:-1], label="Recall", color="#C44E52")
    ax.plot(thresholds, f1_scores[:-1], label="F1", color="#55A868", linestyle="--")
    best_t = thresholds[np.argmax(f1_scores[:-1])]
    ax.axvline(best_t, color="gray", linestyle=":", label=f"Best F1 threshold={best_t:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision/Recall Trade-off vs Threshold — {best_model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/10_threshold_tradeoff.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(f"{DATA_DIR}/features.csv")
    bundle = joblib.load(f"{OUT_DIR}/trained_models.joblib")
    models, feature_cols = bundle["models"], bundle["feature_cols"]
    predictions = joblib.load(f"{OUT_DIR}/predictions.joblib")
    results_df = pd.read_csv(f"{OUT_DIR}/model_comparison.csv")

    fig01_class_imbalance(df)
    fig02_amount_distribution(df)
    fig03_hourly_fraud_rate(df)
    fig04_correlation_heatmap(df)
    fig05_roc_curves(predictions)
    fig06_pr_curves(predictions, df)
    fig07_confusion_matrices(predictions)
    fig08_feature_importance(models, feature_cols)
    fig09_model_comparison_bars(results_df)
    fig10_threshold_tradeoff(predictions, results_df.iloc[0]["model"])

    print("All figures saved to", FIG_DIR)
