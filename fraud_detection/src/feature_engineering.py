import numpy as np
import pandas as pd
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_index"] = (df["Time"] // 86400).astype(int)
    df["is_weekend"] = (df["day_index"] % 7 >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_log"] = np.log1p(df["Amount"])
    df["amount_is_round"] = (df["Amount"] % 1 == 0).astype(int)
    return df


def add_customer_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal rolling features computed per-customer in chronological order."""
    df = df.sort_values(["customer_id", "Time"]).copy()

    counts_1h, counts_24h = [], []
    zscores, run_mean, run_count_total = [], [], []

    for _, group in df.groupby("customer_id", sort=False):
        times = group["Time"].values
        amounts = group["Amount"].values
        n = len(group)
        c1h = np.zeros(n)
        c24h = np.zeros(n)
        z = np.zeros(n)
        rmean = np.zeros(n)
        rcount = np.zeros(n)

        hist_amounts = []
        for i in range(n):
            t = times[i]
            # counts strictly before current transaction, within window
            window_1h = (times[:i] > t - 3600) & (times[:i] <= t)
            window_24h = (times[:i] > t - 86400) & (times[:i] <= t)
            c1h[i] = window_1h.sum()
            c24h[i] = window_24h.sum()

            if len(hist_amounts) >= 2:
                m = np.mean(hist_amounts)
                s = np.std(hist_amounts) + 1e-6
                z[i] = (amounts[i] - m) / s
                rmean[i] = m
            else:
                z[i] = 0.0
                rmean[i] = amounts[i]
            rcount[i] = len(hist_amounts)
            hist_amounts.append(amounts[i])

        counts_1h.append(c1h)
        counts_24h.append(c24h)
        zscores.append(z)
        run_mean.append(rmean)
        run_count_total.append(rcount)

    df["cust_txn_count_1h"] = np.concatenate(counts_1h)
    df["cust_txn_count_24h"] = np.concatenate(counts_24h)
    df["cust_amount_zscore"] = np.concatenate(zscores)
    df["cust_mean_amount_hist"] = np.concatenate(run_mean)
    df["cust_prior_txn_count"] = np.concatenate(run_count_total)

    return df.sort_values("Time").reset_index(drop=True)


def add_merchant_frequency_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    freq = df["merchant_category"].value_counts(normalize=True)
    df["merchant_freq"] = df["merchant_category"].map(freq)
    df = pd.get_dummies(df, columns=["merchant_category"], prefix="merchant", drop_first=True)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = add_temporal_features(df)
    df = add_amount_features(df)
    df = add_customer_velocity_features(df)
    df = add_merchant_frequency_encoding(df)
    return df


if __name__ == "__main__":
    raw = pd.read_csv("/home/claude/fraud_detection/data/transactions.csv")
    feats = build_feature_matrix(raw)
    feats.to_csv("/home/claude/fraud_detection/data/features.csv", index=False)
    print(f"Feature matrix shape: {feats.shape}")
    print(feats.columns.tolist())
