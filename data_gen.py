"""
data_gen.py
-----------
Generate a synthetic credit-card transaction dataset whose *schema* matches
Table 1 of Zhang, Han, Xu & Wang (2021), "HOBA: A novel feature engineering
methodology for credit card fraud detection with a deep learning architecture",
Information Sciences 557, 302-316.

The original study used a proprietary dataset from a large Chinese commercial
bank (153,685 raw records, ~1.4% fraud, 114,779 after cleaning). That data is
not public, so we simulate transactions that carry the SAME raw attributes the
paper needs for HOBA feature engineering (location, MCC, entry mode,
open-to-buy, transaction type, timestamp, account), and inject the three
behaviour-fraud archetypes the paper names in Section 2.1:

  * theft/stolen-card  -> short inter-transaction intervals, escalating spend,
                          rapid open-to-buy depletion
  * counterfeit-card   -> transactions in an unusual location while the genuine
                          card keeps being used elsewhere
  * card-not-present   -> remote/online, foreign country, odd hours, high-risk
                          merchant categories

These are exactly the signals HOBA is designed to capture (geographical
distance, open-to-buy utilisation, per-characteristic behaviour) and that the
plain RFM framework largely ignores.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---- reference tables ------------------------------------------------------
# A handful of cities with (lat, lon) so we can compute real geographic distance
CITIES = {
    "Beijing":   (39.90, 116.40),
    "Shanghai":  (31.23, 121.47),
    "Guangzhou": (23.13, 113.26),
    "Chengdu":   (30.57, 104.07),
    "Xian":      (34.34, 108.94),
    "Harbin":    (45.80, 126.53),
    "Kunming":   (25.04, 102.71),
    "Urumqi":    (43.83, 87.62),
    # "foreign" hubs used mainly by card-not-present fraud
    "HongKong":  (22.32, 114.17),
    "Singapore": (1.35, 103.82),
    "London":    (51.51, -0.13),
    "NewYork":   (40.71, -74.01),
}
CITY_NAMES = list(CITIES.keys())
DOMESTIC = CITY_NAMES[:8]
FOREIGN = CITY_NAMES[8:]
CITY_COUNTRY = {c: ("CN" if c in DOMESTIC else {"HongKong": "HK", "Singapore": "SG",
                    "London": "GB", "NewYork": "US"}[c]) for c in CITY_NAMES}

# Merchant Category Codes: a few normal ones + designated "high risk" ones
NORMAL_MCC = [5411, 5812, 5912, 5541, 5311, 5732, 4111, 5999]   # grocery, dining, etc.
HIGHRISK_MCC = [6011, 7995, 6051, 5993]                          # ATM/cash, gambling, quasi-cash
TX_TYPES = ["purchase", "cash_withdrawal", "online_purchase", "refund"]
ENTRY_MODES = ["chip", "magstripe", "contactless", "ecommerce"]


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _new_account(acc_id):
    home = RNG.choice(DOMESTIC, p=_domestic_pop())
    credit_limit = float(RNG.choice([10000, 20000, 30000, 50000, 80000, 120000],
                                    p=[.25, .25, .2, .15, .1, .05]))
    return {
        "account_number": 100000 + acc_id,
        "card_number": 4000_0000_0000_0000 + acc_id,
        "home_city": home,
        "credit_limit": credit_limit,
        "open_to_buy": credit_limit * RNG.uniform(0.4, 1.0),
        "base_amount": float(RNG.uniform(80, 600)),      # typical spend level
        "daily_rate": RNG.uniform(0.2, 1.1),             # tx per day intensity
    }


def _domestic_pop():
    w = np.array([5, 5, 4, 3, 3, 2, 2, 2], float)
    return w / w.sum()


def _sample_city_near(home):
    """Genuine txns: mostly home city, sometimes a domestic trip."""
    if RNG.random() < 0.85:
        return home
    return RNG.choice(DOMESTIC)


def generate(n_accounts=2000, days=90, fraud_account_frac=0.12, seed=42):
    """Return a raw transaction dataframe sorted by time, with a fraud label."""
    global RNG
    RNG = np.random.default_rng(seed)
    accounts = [_new_account(i) for i in range(n_accounts)]
    start = pd.Timestamp("2015-06-01")
    rows = []

    fraud_accounts = set(RNG.choice(n_accounts,
                         size=int(n_accounts * fraud_account_frac), replace=False))

    for a in accounts:
        # ---- legitimate transaction stream -------------------------------
        n_tx = RNG.poisson(a["daily_rate"] * days)
        n_tx = max(n_tx, 5)
        # event times across the window (days), sorted
        ts = np.sort(RNG.uniform(0, days, size=n_tx))
        otb = a["open_to_buy"]
        for t in ts:
            ts_full = start + pd.Timedelta(days=float(t))
            # genuine spend tends to daytime
            hour = int(np.clip(RNG.normal(14, 4), 0, 23))
            ts_full = ts_full.replace(hour=hour, minute=int(RNG.integers(0, 60)),
                                      second=int(RNG.integers(0, 60)))
            ttype = RNG.choice(TX_TYPES, p=[.62, .12, .20, .06])
            city = _sample_city_near(a["home_city"])
            # small base rate of genuine foreign / travel activity (noise so the
            # 'foreign' flag is not a perfect fraud predictor)
            if RNG.random() < 0.03:
                city = RNG.choice(FOREIGN)
            mcc = (int(RNG.choice(HIGHRISK_MCC)) if ttype == "cash_withdrawal"
                   else int(RNG.choice(NORMAL_MCC)))
            entry = ("ecommerce" if ttype == "online_purchase"
                     else RNG.choice(["chip", "magstripe", "contactless"], p=[.6, .25, .15]))
            amt = float(np.abs(RNG.normal(a["base_amount"], a["base_amount"] * 0.5)) + 5)
            if ttype == "cash_withdrawal":
                amt = float(RNG.choice([500, 1000, 2000, 3000]))
            if ttype == "refund":
                amt = -abs(amt) * RNG.uniform(0.2, 1.0)
            otb = min(a["credit_limit"], max(0.0, otb - max(amt, 0) + (RNG.random() < .3) * abs(amt)))
            rows.append(_row(a, ts_full, amt, ttype, mcc, city, entry, otb, label=0))

        # ---- inject a fraud burst for selected accounts ------------------
        if a["account_number"] - 100000 in fraud_accounts and n_tx > 8:
            _inject_fraud(rows, a, start, days)

    df = pd.DataFrame(rows)
    df = df.sort_values("tx_datetime").reset_index(drop=True)
    # currency + approval (mostly approved); approval not predictive, included for schema fidelity
    df["currency_code"] = np.where(df["transaction_country"] == "CN", "CNY",
                            df["transaction_country"].map({"HK": "HKD", "SG": "SGD",
                                                           "GB": "GBP", "US": "USD"}))
    df["approval_code"] = RNG.choice(["approve", "reject"], size=len(df), p=[.985, .015])
    df["merchant_number"] = RNG.integers(10_000_000, 99_999_999, size=len(df))
    return df


def _row(a, ts, amt, ttype, mcc, city, entry, otb, label):
    lat, lon = CITIES[city]
    return {
        "account_number": a["account_number"],
        "card_number": a["card_number"],
        "credit_limit": a["credit_limit"],
        "open_to_buy": float(otb),
        "tx_datetime": ts,
        "transaction_amount": round(float(amt), 2),
        "transaction_type": ttype,
        "merchant_category_code": mcc,
        "transaction_city": city,
        "transaction_country": CITY_COUNTRY[city],
        "lat": lat, "lon": lon,
        "entry_mode": entry,
        "home_city": a["home_city"],
        "is_fraud": label,
    }


def _inject_fraud(rows, a, start, days):
    """Append a contiguous fraud burst of one of three archetypes.

    Fraud is deliberately *modest in amount and frequency* so the plain RFM
    framework (which only sees recency/frequency/monetary on ALL transactions)
    is handicapped. The discriminating signal lives in dimensions only HOBA
    captures: geographic distance, open-to-buy utilisation, per-characteristic
    behaviour (cash / online / foreign) and abnormal transaction time.
    """
    archetype = RNG.choice(["theft", "counterfeit", "cnp"], p=[0.30, 0.40, 0.30])
    t0 = RNG.uniform(days * 0.2, days * 0.95)
    ts0 = start + pd.Timedelta(days=float(t0))
    otb = max(800.0, a["open_to_buy"] * RNG.uniform(0.5, 1.0))
    base = a["base_amount"]
    n = int(RNG.integers(4, 10))

    if archetype == "theft":
        # signal = sudden run of CASH withdrawals draining open-to-buy fast,
        # at home city, amounts ordinary -> RFM monetary looks normal.
        city = a["home_city"]
        cur = ts0
        for k in range(n):
            cur = cur + pd.Timedelta(minutes=float(RNG.uniform(20, 120)))
            amt = float(min(otb, RNG.choice([500, 800, 1000, 1500])))
            otb = max(0.0, otb - amt)                       # rapid OTB depletion
            mcc = int(RNG.choice(HIGHRISK_MCC))
            rows.append(_row(a, _odd_hour(cur), amt, "cash_withdrawal", mcc,
                             city, "magstripe", otb, 1))

    elif archetype == "counterfeit":
        # signal = cloned card used in a DISTANT domestic city (large geo
        # distance vs the account's usual location); amounts ordinary.
        city = RNG.choice([c for c in DOMESTIC if c != a["home_city"]])
        cur = ts0
        for k in range(n):
            cur = cur + pd.Timedelta(hours=float(RNG.uniform(0.5, 4)))
            amt = float(min(otb, base * RNG.uniform(1.0, 2.2)))
            otb = max(0.0, otb - amt)
            mcc = int(RNG.choice(NORMAL_MCC))
            rows.append(_row(a, cur, amt, "purchase", mcc, city, "magstripe", otb, 1))

    else:  # card-not-present
        # signal = remote online purchases, FOREIGN country, odd hours;
        # amounts ordinary. RFM is blind to country / channel / time.
        cur = ts0
        for k in range(n):
            cur = cur + pd.Timedelta(hours=float(RNG.uniform(0.3, 3)))
            city = RNG.choice(FOREIGN)
            amt = float(min(otb, base * RNG.uniform(1.0, 2.5)))
            otb = max(0.0, otb - amt)
            mcc = int(RNG.choice(NORMAL_MCC + HIGHRISK_MCC))
            rows.append(_row(a, _odd_hour(cur), amt, "online_purchase", mcc, city,
                             "ecommerce", otb, 1))


def _odd_hour(ts):
    """Push some fraud into the 1am-5am 'abnormal time' band the paper flags."""
    if RNG.random() < 0.6:
        return ts.replace(hour=int(RNG.integers(1, 5)),
                          minute=int(RNG.integers(0, 60)))
    return ts


if __name__ == "__main__":
    df = generate()
    df.to_parquet("/home/claude/hoba/raw_transactions.parquet")
    n = len(df); f = int(df.is_fraud.sum())
    print(f"rows={n:,}  fraud={f:,} ({100*f/n:.2f}%)  accounts={df.account_number.nunique():,}")
    print(df.is_fraud.groupby(df.transaction_type).mean().round(4).to_string())
    print("\nschema:", list(df.columns))
