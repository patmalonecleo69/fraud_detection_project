import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 3000
N_TRANSACTIONS = 120_000
FRAUD_RATE = 0.0022  # ~0.22%, in line with real-world fraud prevalence

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "gas_station",
    "online_retail", "restaurant", "entertainment", "utilities"
]
# Fraud is not uniform across merchant categories in real life either
MERCHANT_FRAUD_WEIGHT = {
    "grocery": 0.4, "electronics": 2.5, "travel": 2.0, "gas_station": 0.6,
    "online_retail": 2.8, "restaurant": 0.5, "entertainment": 1.2, "utilities": 0.2
}


def generate_transactions(
    n_transactions: int = N_TRANSACTIONS,
    n_customers: int = N_CUSTOMERS,
    fraud_rate: float = FRAUD_RATE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(n_transactions * fraud_rate))
    n_legit = n_transactions - n_fraud
    labels = np.array([0] * n_legit + [1] * n_fraud)
    rng.shuffle(labels)

    n = len(labels)
    customer_id = rng.integers(0, n_customers, size=n)

    # Time: seconds elapsed over a 30-day observation window
    total_seconds = 30 * 24 * 3600
    time_seconds = rng.uniform(0, total_seconds, size=n)

    # Fraud is more likely at night (local-time hour derived from time_seconds)
    hour_of_day = (time_seconds // 3600 % 24).astype(int)
    # Shift a fraction of fraud transactions into the 0-5am window
    night_shift_mask = (labels == 1) & (rng.random(n) < 0.55)
    time_seconds[night_shift_mask] = (
        (time_seconds[night_shift_mask] // 86400) * 86400
        + rng.uniform(0, 5 * 3600, size=night_shift_mask.sum())
    )

    # Merchant category: fraud skews toward electronics/online_retail/travel
    merchant_category = np.empty(n, dtype=object)
    base_weights = np.array([1.0] * len(MERCHANT_CATEGORIES))
    fraud_weights = np.array([MERCHANT_FRAUD_WEIGHT[c] for c in MERCHANT_CATEGORIES])
    for cls, weights in [(0, base_weights), (1, fraud_weights)]:
        mask = labels == cls
        p = weights / weights.sum()
        merchant_category[mask] = rng.choice(MERCHANT_CATEGORIES, size=mask.sum(), p=p)

    # Amount: legit amounts lognormal (small everyday purchases dominate);
    # fraud amounts are bimodal -- many small "card testing" charges plus
    # some unusually large ones.
    amount = np.empty(n)
    legit_mask = labels == 0
    fraud_mask = labels == 1
    amount[legit_mask] = rng.lognormal(mean=3.0, sigma=1.1, size=legit_mask.sum())
    small_test = rng.random(fraud_mask.sum()) < 0.5
    fraud_amounts = np.where(
        small_test,
        rng.uniform(0.5, 5.0, size=fraud_mask.sum()),
        rng.lognormal(mean=5.3, sigma=0.9, size=fraud_mask.sum()),
    )
    amount[fraud_mask] = fraud_amounts
    amount = np.round(np.clip(amount, 0.5, 25000), 2)

    # V1..V10: anonymized PCA-like components. Legit ~ N(0,1) independent.
    # Fraud has a mean/covariance shift on a subset of components to create
    # a learnable but imperfect signal (mimics real PCA components V4, V11,
    # V14, V17 etc. being the most predictive in the real dataset).
    n_v = 10
    V = rng.normal(0, 1, size=(n, n_v))
    fraud_shift = np.zeros(n_v)
    fraud_shift[[1, 3, 6]] = [2.2, -2.6, 1.8]   # strong signal components
    fraud_shift[[0, 8]] = [0.6, -0.5]            # weak signal components
    V[fraud_mask] = V[fraud_mask] * 1.4 + fraud_shift  # also higher variance
    # add correlated noise between a couple of components (realistic PCA leakage)
    V[:, 2] = V[:, 2] + 0.3 * V[:, 1]

    df = pd.DataFrame(V, columns=[f"V{i+1}" for i in range(n_v)])
    df.insert(0, "transaction_id", np.arange(1, n + 1))
    df.insert(1, "customer_id", customer_id)
    df["Time"] = time_seconds
    df["hour_of_day"] = hour_of_day
    df["merchant_category"] = merchant_category
    df["Amount"] = amount
    df["Class"] = labels

    df = df.sort_values("Time").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_transactions()
    df.to_csv("/home/claude/fraud_detection/data/transactions.csv", index=False)
    print(f"Generated {len(df):,} transactions, {df['Class'].sum()} fraudulent "
          f"({100*df['Class'].mean():.3f}% fraud rate)")
    print(df.head())
