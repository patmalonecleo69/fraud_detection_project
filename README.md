# Credit Card Fraud Detection — ML Pipeline

A complete, reproducible machine learning pipeline for detecting fraudulent
credit card transactions in a severely imbalanced dataset (~0.2% fraud rate).

This project component (source code + data + visualizations) is designed to
support a technical report and slide deck. Everything below is written so
you can lift definitions, rationale, and numbers directly into those
documents.

## 1. Project Structure

```
fraud_detection/
├── main.py                    # Runs the entire pipeline end-to-end
├── src/
│   ├── generate_data.py       # Synthetic transaction data generator
│   ├── feature_engineering.py # Feature engineering pipeline
│   ├── modeling.py            # Model training, SMOTE, evaluation
│   └── visualize.py           # All report/slide figures
├── data/
│   ├── transactions.csv       # Raw generated transactions
│   └── features.csv           # Feature-engineered matrix
├── outputs/
│   ├── model_comparison.csv   # Metrics table for all models
│   ├── trained_models.joblib  # Pickled trained models + scaler
│   └── predictions.joblib     # Test-set predictions for each model
└── figures/                   # 10 PNG figures, ready for report/slides
```

Run everything with:
```bash
pip install -r requirements.txt
python3 main.py
```

## 2. Problem Statement

Credit card fraud costs issuers and consumers billions of dollars annually.
The core technical challenge is **extreme class imbalance**: fraudulent
transactions typically make up well under 1% of all transactions, so a
naive classifier that predicts "legitimate" for everything achieves
>99% accuracy while catching zero fraud. The goal of this project is to
build and compare classifiers that maximize fraud detection (recall)
while keeping false positives (blocked legitimate transactions) low
enough to be operationally usable — evaluated with metrics appropriate
for imbalanced classification (ROC-AUC, Average Precision, F1) rather
than raw accuracy.

## 3. Data

**Why synthetic data:** Real fraud datasets (e.g. the ULB/Kaggle European
credit card dataset) are protected by strict privacy/PCI-DSS constraints
and are typically distributed pre-anonymized via PCA, which strips away
the raw fields needed to demonstrate feature engineering. This project
generates a **synthetic dataset that mirrors the statistical structure of
real-world fraud data** — matching its class imbalance (~0.2%), amount
distribution shape, temporal fraud concentration, and latent/PCA-style
signal structure — while retaining interpretable raw fields (customer ID,
timestamp, merchant category) so the full feature-engineering process can
be demonstrated end-to-end. The generation logic and every distributional
assumption is documented in `src/generate_data.py`.

**Dataset summary:**
- 120,000 transactions across 3,000 simulated customers over a 30-day window
- 264 fraudulent transactions (0.22% fraud rate)
- Fields: `transaction_id`, `customer_id`, `Time`, `Amount`, `hour_of_day`,
  `merchant_category`, 10 anonymized/PCA-style features (`V1`-`V10`), `Class` (target)

**Injected fraud signal (intentionally imperfect/overlapping, as in real data):**
- Fraud concentrates in the 12am-5am window (55% of fraud transactions)
- Fraud skews toward `electronics`, `online_retail`, and `travel` merchants
- Fraud amounts are bimodal: small "card-testing" charges (~$0.5-5) or unusually large purchases
- Three of the ten anonymized features carry a strong mean/variance shift for fraud; two carry a weak shift; the remainder are pure noise

## 4. Feature Engineering

Implemented in `src/feature_engineering.py`. Five feature groups, 36 total columns:

| Group | Features | Rationale |
|---|---|---|
| Temporal | `hour_sin`, `hour_cos`, `is_night`, `is_weekend` | Cyclical encoding of hour avoids treating 23:00 and 00:00 as distant; night flag captures elevated fraud concentration |
| Amount | `amount_log`, `amount_is_round` | Log-transform tames heavy right skew; round amounts are a common card-testing signal |
| Customer velocity | `cust_txn_count_1h`, `cust_txn_count_24h` | Rolling, **causal** (past-only) counts of a customer's recent transactions — fraud often clusters in bursts |
| Customer behavior | `cust_amount_zscore`, `cust_mean_amount_hist`, `cust_prior_txn_count` | How far this transaction deviates from the customer's own historical spending pattern |
| Merchant | `merchant_freq` + one-hot merchant category | Frequency encoding avoids dimensionality blow-up; captures category-level fraud base rates |

**Leakage control:** all customer-behavior features are computed using only
transactions strictly *before* the current one in time, and the train/test
split is time-based (not random), so no future information leaks into
training — a common pitfall in fraud-detection pipelines that inflates
reported performance.

## 5. Modeling

Implemented in `src/modeling.py`.

- **Train/test split:** time-based (75%/25%), not random — realistic for
  a deployed fraud system that only ever sees the future.
- **Class imbalance handling:** SMOTE (Synthetic Minority Over-sampling)
  applied to the *training set only*, oversampling fraud to 15% of the
  training data. Test data is left at its natural ~0.2% rate.
- **Models compared:**
  1. **Logistic Regression** (`class_weight="balanced"`) — interpretable linear baseline
  2. **Random Forest** (300 trees, `class_weight="balanced_subsample"`) — non-linear, robust
  3. **XGBoost** (400 trees, learning rate 0.05) — gradient boosting, typically strongest on tabular fraud data
- **Evaluation metrics:** ROC-AUC, Average Precision (area under PR curve —
  the most informative metric under severe imbalance), Precision, Recall, F1,
  all computed on the untouched, naturally-imbalanced test set.

## 6. Results

| Model | ROC-AUC | Avg. Precision | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **XGBoost** | **0.992** | **0.854** | 0.839 | 0.813 | **0.825** |
| Random Forest | 0.987 | 0.607 | 0.418 | 0.719 | 0.529 |
| Logistic Regression | 0.982 | 0.636 | 0.099 | 0.875 | 0.178 |

**Key finding:** all three models achieve high ROC-AUC (>0.98), but
ROC-AUC is a misleading headline metric here — it's dominated by the
huge number of easy true negatives. **Average Precision** exposes the
real gap: XGBoost catches 81% of fraud while keeping precision at 84%
(few false alarms), whereas Logistic Regression, despite the highest
raw recall (87.5%), does so at the cost of enormous false-positive
volume (precision of only 10% — it would flag 9 legitimate transactions
for every real fraud). Random Forest sits between the two. This
precision/recall trade-off, and *why* accuracy/ROC-AUC alone would have
hidden it, is the central analytical point to make in the report.

Feature importance analysis (Figure 8) shows the engineered
`cust_amount_zscore` and `cust_txn_count_1h`/`cust_txn_count_24h` velocity
features rank among the top predictors for both tree models — direct
evidence that the engineered behavioral features add value beyond the raw
anonymized signal.

## 7. Figures (in `figures/`, ready to drop into report or slides)

| File | Content |
|---|---|
| `01_class_imbalance.png` | Log-scale bar chart of class distribution |
| `02_amount_distribution.png` | Amount distributions, legit vs. fraud |
| `03_hourly_fraud_rate.png` | Fraud rate by hour of day (shows night concentration) |
| `04_correlation_heatmap.png` | Per-feature correlation with fraud label |
| `05_roc_curves.png` | ROC curves, all 3 models |
| `06_pr_curves.png` | Precision-Recall curves, all 3 models (the key evaluation figure) |
| `07_confusion_matrices.png` | Confusion matrix per model at 0.5 threshold |
| `08_feature_importance.png` | Top 15 feature importances, RF & XGBoost |
| `09_model_comparison_bars.png` | Grouped bar chart, all metrics, all models |
| `10_threshold_tradeoff.png` | Precision/Recall/F1 vs. decision threshold for best model |

## 8. Limitations & Future Work

- **Synthetic data:** results demonstrate methodology correctly but
  absolute numbers won't match a production system trained on real data;
  the report should frame this as a deliberate, documented substitution
  for a data source that's practically unobtainable at student scale
  (see Section 3).
- **Static model:** a production system would need drift monitoring, since
  fraud patterns evolve as fraudsters adapt.
- **Threshold selection:** the default 0.5 decision threshold is rarely
  optimal for imbalanced problems; `10_threshold_tradeoff.png` shows how
  precision/recall/F1 shift across thresholds, and a deployed system
  should pick a threshold based on the business cost ratio of a missed
  fraud vs. a false alarm.
- **Possible extensions:** cost-sensitive learning with explicit
  false-negative/false-positive cost weighting, SHAP-based explainability
  for individual predictions, an ensemble/stacking model, and graph-based
  features capturing merchant/customer network structure.

## 9. Requirements

```
numpy
pandas
scikit-learn
imbalanced-learn
xgboost
matplotlib
seaborn
joblib
```
