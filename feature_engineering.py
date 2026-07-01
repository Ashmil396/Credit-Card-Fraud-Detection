"""
feature_engineering.py
----------------------
Implements the two feature-engineering frameworks compared in the paper:

  HOBA  (Section 3.3) -- homogeneity-oriented behaviour analysis.
        For each *characteristic* (a homogeneous subset of transactions) we run
        four behaviour analyses (recency / frequency / monetary / location) by
        combining:
            aggregation characteristic  x  aggregation period
            x transaction behaviour measure  x  aggregation statistic
        Behaviour measures include the two the paper adds beyond prior work:
        geographic distance between sequential transactions, and open-to-buy
        utilisation (relative monetary). Plus a few rule-based binary flags.

  RFM   (benchmark, Van Vlasselaer 2015) -- recency / frequency / monetary
        aggregated over several time windows on ALL transactions, ignoring
        per-characteristic heterogeneity, location and open-to-buy.

The full paper builds 1410 HOBA variables (13 characteristics x 9 periods x
4 measures x 4 statistics, subset) and 87 RFM variables. We build a faithful,
representative subset of each (documented by the printed counts) so the whole
pipeline runs in the course-demo environment while preserving the framework.

KEY IMPLEMENTATION IDEA
-----------------------
"Last k transactions with characteristic c, take statistic s of measure m" is
computed by masking measure m to NaN on rows where c==0, then applying a
groupby(account).rolling(window) statistic that *skips* NaN. The rolling window
counts rows (the period), while the statistic is taken only over matching rows
-- exactly the HOBA semantics, fully vectorised.
"""
import numpy as np
import pandas as pd
from data_gen import haversine, CITIES, HIGHRISK_MCC

# ---------------------------------------------------------------- helpers ----
COUNT_WINDOWS = {"c1": 1, "c5": 5, "c15": 15}        # past 1 / 5 / 15 transactions
TIME_WINDOWS = {"d1": "1D", "d5": "5D"}              # past 1 day / 5 days
STATS = ["mean", "max", "sum", "std"]


def _prep(df):
    df = df.sort_values(["account_number", "tx_datetime"]).reset_index(drop=True)
    dt = df["tx_datetime"]
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["abs_amount"] = df["transaction_amount"].abs()
    df["otb_util"] = (df["abs_amount"] / df["open_to_buy"].clip(lower=1)).clip(0, 5)
    # ---- per-account sequential measures: time gap & geo distance ----
    g = df.groupby("account_number", sort=False)
    df["prev_t"] = g["tx_datetime"].shift(1)
    df["time_interval"] = (df["tx_datetime"] - df["prev_t"]).dt.total_seconds()
    df["prev_lat"] = g["lat"].shift(1)
    df["prev_lon"] = g["lon"].shift(1)
    df["geo_distance"] = haversine(df["lat"], df["lon"], df["prev_lat"], df["prev_lon"])
    df["time_interval"] = df["time_interval"].fillna(0)
    df["geo_distance"] = df["geo_distance"].fillna(0)
    return df


def _characteristics(df):
    """Binary indicator columns: homogeneous transaction characteristics."""
    c = pd.DataFrame(index=df.index)
    c["all"] = 1
    c["purchase"] = (df["transaction_type"] == "purchase").astype(int)
    c["cash"] = (df["transaction_type"] == "cash_withdrawal").astype(int)
    c["online"] = (df["transaction_type"] == "online_purchase").astype(int)
    c["foreign"] = (df["transaction_country"] != "CN").astype(int)
    c["highrisk_mcc"] = df["merchant_category_code"].isin(HIGHRISK_MCC).astype(int)
    c["abnormal_time"] = df["hour"].between(1, 5).astype(int)
    c["magstripe"] = (df["entry_mode"] == "magstripe").astype(int)
    return c


def build_hoba(df):
    """Return (feature_matrix, meta) for the HOBA framework."""
    df = _prep(df.copy())
    chars = _characteristics(df)
    measures = ["abs_amount", "time_interval", "geo_distance", "otb_util"]
    acct = df["account_number"].values
    feats = {}

    # time-indexed frame for time-window rolling
    tdf = df.set_index("tx_datetime")

    for cname in chars.columns:
        mask = chars[cname].values.astype(bool)
        for m in measures:
            masked = df[m].where(mask)                       # NaN where char != 1
            # ---- count-based periods (past k transactions) ----
            s = masked.groupby(df["account_number"]).shift(0)  # align
            grp = masked.groupby(acct)
            for wn, k in COUNT_WINDOWS.items():
                roll = grp.rolling(k, min_periods=1)
                for st in STATS:
                    col = f"H_{cname}_{m}_{wn}_{st}"
                    vals = getattr(roll, st)().reset_index(level=0, drop=True)
                    feats[col] = vals.values
            # ---- time-based periods (past d days) ----
            mt = pd.Series(masked.values, index=tdf.index)
            grpt = mt.groupby(tdf["account_number"].values)
            for wn, off in TIME_WINDOWS.items():
                for st in ["mean", "sum"]:                   # keep time-window set lean
                    col = f"H_{cname}_{m}_{wn}_{st}"
                    vals = getattr(grpt.rolling(off), st)()
                    feats[col] = vals.values

    X = pd.DataFrame(feats, index=df.index)

    # ---- rule-based binary variables (Section 3.3) -------------------------
    X["R_abnormal_time"] = chars["abnormal_time"]
    X["R_foreign"] = chars["foreign"]
    X["R_magstripe"] = chars["magstripe"]
    X["R_highamt_abntime"] = ((df["abs_amount"] > 500) & chars["abnormal_time"].astype(bool)).astype(int)
    X["R_foreign_highrisk"] = (chars["foreign"].astype(bool) & chars["highrisk_mcc"].astype(bool)).astype(int)
    X["R_otb_gt80"] = (df["otb_util"] > 0.8).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    meta = df[["account_number", "tx_datetime", "is_fraud"]].copy()
    return X, meta


def build_rfm(df):
    """Recency / Frequency / Monetary benchmark over several windows (no
    per-characteristic split, no location, no open-to-buy)."""
    df = _prep(df.copy())
    acct = df["account_number"].values
    feats = {}
    amt = df["abs_amount"]

    # Recency: seconds since previous transaction (and count-window recency proxy)
    feats["RFM_recency_sec"] = df["time_interval"].values

    # Count-based frequency & monetary
    for wn, k in {"c1": 1, "c5": 5, "c10": 10, "c15": 15}.items():
        roll = amt.groupby(acct).rolling(k, min_periods=1)
        feats[f"RFM_mon_sum_{wn}"] = roll.sum().reset_index(level=0, drop=True).values
        feats[f"RFM_mon_mean_{wn}"] = roll.mean().reset_index(level=0, drop=True).values
        feats[f"RFM_mon_max_{wn}"] = roll.max().reset_index(level=0, drop=True).values
        feats[f"RFM_mon_std_{wn}"] = roll.std().reset_index(level=0, drop=True).fillna(0).values

    # Time-based frequency & monetary
    tdf = df.set_index("tx_datetime")
    amt_t = pd.Series(amt.values, index=tdf.index)
    one = pd.Series(1.0, index=tdf.index)
    for wn, off in {"d1": "1D", "d3": "3D", "d7": "7D", "d30": "30D"}.items():
        feats[f"RFM_freq_{wn}"] = one.groupby(tdf["account_number"].values).rolling(off).sum().values
        feats[f"RFM_mon_sum_{wn}"] = amt_t.groupby(tdf["account_number"].values).rolling(off).sum().values
        feats[f"RFM_mon_mean_{wn}"] = amt_t.groupby(tdf["account_number"].values).rolling(off).mean().values

    X = pd.DataFrame(feats, index=df.index)

    # categorical variables the RFM literature also adds (one-hot)
    cat = pd.get_dummies(df[["transaction_type", "entry_mode"]].astype(str), prefix=["tt", "em"]).astype(int)
    hour_oh = pd.get_dummies(pd.cut(df["hour"], bins=[-1, 5, 11, 17, 23],
                             labels=["night", "morn", "noon", "eve"]), prefix="hr").astype(int)
    X = pd.concat([X, cat.reset_index(drop=True), hour_oh.reset_index(drop=True)], axis=1)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    meta = df[["account_number", "tx_datetime", "is_fraud"]].copy()
    return X, meta


if __name__ == "__main__":
    df = pd.read_parquet("/home/claude/hoba/raw_transactions.parquet")
    Xh, meta = build_hoba(df)
    Xr, _ = build_rfm(df)
    print(f"HOBA feature matrix: {Xh.shape[0]:,} x {Xh.shape[1]} features")
    print(f"RFM  feature matrix: {Xr.shape[0]:,} x {Xr.shape[1]} features")
    Xh.to_parquet("/home/claude/hoba/X_hoba.parquet")
    Xr.to_parquet("/home/claude/hoba/X_rfm.parquet")
    meta.to_parquet("/home/claude/hoba/meta.parquet")
    print("saved.")
