"""
run_experiments.py
------------------
Reproduces the empirical design of Section 4:
  * 6 classifiers  x  2 feature sets (RFM benchmark vs proposed HOBA)
  * Table 3  : F1 / precision / recall / accuracy / AUC at the 0.5 cutoff
  * Table 5a : same metrics constrained to <= 1% false-positive rate
  * Table 5b : same metrics constrained to <= 3% false-positive rate
  * Fig. 6   : ROC curves

Deep models: DBN (RBM pre-train + fine-tune), CNN, RNN  (PyTorch).
Traditional: BPNN (1 hidden layer MLP), SVM (RBF), RF   (scikit-learn).
Class imbalance handled by class weights / minority oversampling / BCE pos_weight,
matching the paper's remark that "either a good sampling procedure or an
adjustment to the cost function is needed".
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, accuracy_score, roc_curve)
import torch
import deep_models as dm

RNG = np.random.default_rng(0)
T_SEQ = 5          # recent transactions per CNN/RNN sample
SEQ_DIM = 120      # feature dim fed to sequence models (top-variance subset)


# ----------------------------------------------------------- data utils -----
def chrono_split(meta, frac=0.523):
    order = meta["tx_datetime"].argsort().values
    cut = int(len(order) * frac)
    tr_idx = np.sort(order[:cut]); te_idx = np.sort(order[cut:])
    return tr_idx, te_idx


def oversample(Xtr, ytr, ratio=0.3):
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    target_pos = int(len(neg) * ratio)
    if len(pos) == 0 or target_pos <= len(pos):
        return Xtr, ytr
    extra = RNG.choice(pos, size=target_pos - len(pos), replace=True)
    idx = np.concatenate([np.arange(len(ytr)), extra])
    RNG.shuffle(idx)
    return Xtr[idx], ytr[idx]


def build_sequences(Xstd, meta, top_idx):
    """For each row, stack the last T_SEQ transactions (same account) -> (N,T,F)."""
    Xs = Xstd[:, top_idx]
    order = np.lexsort((meta["tx_datetime"].values, meta["account_number"].values))
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    Xo = Xs[order]
    acc = meta["account_number"].values[order]
    N, F = Xo.shape
    seq = np.zeros((N, T_SEQ, F), dtype=np.float32)
    for t in range(T_SEQ):
        shifted = np.zeros_like(Xo)
        if t == 0:
            shifted = Xo
        else:
            shifted[t:] = Xo[:-t]
            same = acc[t:] == acc[:-t]
            shifted[t:][~same] = 0
        seq[:, T_SEQ - 1 - t, :] = shifted
    return seq[inv]            # back to original row order


# --------------------------------------------------------------- metrics ----
def metrics_at_threshold(y, p, thr):
    yhat = (p >= thr).astype(int)
    return dict(
        F1=f1_score(y, yhat, zero_division=0),
        Precision=precision_score(y, yhat, zero_division=0),
        Recall=recall_score(y, yhat, zero_division=0),
        Accuracy=accuracy_score(y, yhat),
    )


def threshold_for_fpr(y, p, fpr_tol):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(fpr <= fpr_tol)[0]
    return thr[ok[-1]] if len(ok) else 1.0


# ----------------------------------------------------------- classifiers ----
def fit_predict(name, Xtr, ytr, Xte, scaler_kind="standard"):
    if scaler_kind == "minmax":
        sc = MinMaxScaler()
    else:
        sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    if name == "BPNN":
        Xo, yo = oversample(Xtr_s, ytr)
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=80,
                            early_stopping=True, random_state=0)
        clf.fit(Xo, yo); return clf.predict_proba(Xte_s)[:, 1]

    if name == "SVM":
        # undersample negatives for tractable RBF SVM, keep all positives.
        # use decision_function -> sigmoid as score (avoids slow Platt CV).
        pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
        neg = RNG.choice(neg, size=min(len(neg), 12 * len(pos)), replace=False)
        sel = np.concatenate([pos, neg])
        clf = SVC(C=2.0, gamma="scale", class_weight="balanced", random_state=0)
        clf.fit(Xtr_s[sel], ytr[sel])
        d = clf.decision_function(Xte_s)
        return 1.0 / (1.0 + np.exp(-d))

    if name == "RF":
        clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                     class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=0)
        clf.fit(Xtr_s, ytr); return clf.predict_proba(Xte_s)[:, 1]

    if name == "DBN":
        mm = MinMaxScaler(); Xtr_m = mm.fit_transform(Xtr); Xte_m = mm.transform(Xte)
        hidden = (128, 64)
        net = dm.DBN(Xtr_m.shape[1], hidden).pretrain(Xtr_m.astype(np.float32), hidden)
        Xo, yo = oversample(Xtr_m, ytr)
        dm.train_torch(net, Xo, yo, epochs=14, pos_weight=None)
        return dm.predict_proba_torch(net, Xte_m)

    raise ValueError(name)


def fit_predict_seq(name, seq_tr, ytr, seq_te):
    T, F = seq_tr.shape[1], seq_tr.shape[2]
    # oversample minority sequences for balanced 0.5-cutoff behaviour
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    target_pos = int(len(neg) * 0.3)
    if len(pos) and target_pos > len(pos):
        extra = RNG.choice(pos, size=target_pos - len(pos), replace=True)
        idx = np.concatenate([np.arange(len(ytr)), extra]); RNG.shuffle(idx)
    else:
        idx = np.arange(len(ytr))
    seq_tr_o, ytr_o = seq_tr[idx], ytr[idx]
    if name == "CNN":
        net = dm.CNNNet(T, F)
        dm.train_torch(net, seq_tr_o[:, None, :, :], ytr_o, epochs=12)
        return dm.predict_proba_torch(net, seq_te[:, None, :, :])
    if name == "RNN":
        net = dm.RNNNet(F)
        dm.train_torch(net, seq_tr_o, ytr_o, epochs=12)
        return dm.predict_proba_torch(net, seq_te)
    raise ValueError(name)


# ------------------------------------------------------------------ main ----
def main():
    import sys
    only_fs = sys.argv[1] if len(sys.argv) > 1 else None
    meta = pd.read_parquet("/home/claude/hoba/meta.parquet")
    y = meta["is_fraud"].values.astype(int)
    tr_idx, te_idx = chrono_split(meta)
    ytr, yte = y[tr_idx], y[te_idx]
    print(f"train={len(tr_idx):,} (fraud {ytr.sum()})  "
          f"test={len(te_idx):,} (fraud {yte.sum()})")

    feature_sets = {"RFM": "X_rfm.parquet", "HOBA": "X_hoba.parquet"}
    if only_fs:
        feature_sets = {only_fs: feature_sets[only_fs]}
    flat_models = ["BPNN", "SVM", "RF", "DBN"]
    seq_models = ["CNN", "RNN"]

    table3, roc_store, fpr_tables = {}, {}, {"1%": {}, "3%": {}}

    for fs, path in feature_sets.items():
        print(f"\n==== feature set: {fs} ====", flush=True)
        X = pd.read_parquet(f"/home/claude/hoba/{path}").values.astype(np.float32)
        Xtr, Xte = X[tr_idx], X[te_idx]

        Fdim = X.shape[1]
        if Fdim > SEQ_DIM:
            sc = StandardScaler().fit(Xtr)
            scores, _ = f_classif(sc.transform(Xtr), ytr)
            scores = np.nan_to_num(scores)
            top_idx = np.argsort(scores)[-SEQ_DIM:]
        else:
            top_idx = np.arange(Fdim)
        Xall_std = StandardScaler().fit(Xtr).transform(X)
        seq = build_sequences(Xall_std, meta, top_idx)
        seq_tr, seq_te = seq[tr_idx], seq[te_idx]

        for name in flat_models:
            p = fit_predict(name, Xtr, ytr, Xte, scaler_kind="standard")
            _store(name, fs, yte, p, table3, roc_store, fpr_tables)
        for name in seq_models:
            p = fit_predict_seq(name, seq_tr, ytr, seq_te)
            _store(name, fs, yte, p, table3, roc_store, fpr_tables)

    tag = only_fs if only_fs else "all"
    out = {"table3": table3, "fpr1": fpr_tables["1%"], "fpr3": fpr_tables["3%"]}
    with open(f"/home/claude/hoba/results_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    np.savez(f"/home/claude/hoba/roc_{tag}.npz",
             **{k: np.array(v) for k, v in roc_store.items()})
    _print_tables(out)


def _store(name, fs, yte, p, table3, roc_store, fpr_tables):
    key = f"{name}+{fs}"
    auc = roc_auc_score(yte, p)
    m = metrics_at_threshold(yte, p, 0.5)
    table3[key] = {**m, "AUC": auc}
    fpr, tpr, _ = roc_curve(yte, p)
    roc_store[key] = np.vstack([fpr, tpr])
    for tol, tag in [(0.01, "1%"), (0.03, "3%")]:
        thr = threshold_for_fpr(yte, p, tol)
        fpr_tables[tag][key] = metrics_at_threshold(yte, p, thr)
    print(f"  {key:12s} F1={m['F1']:.3f} P={m['Precision']:.3f} "
          f"R={m['Recall']:.3f} AUC={auc:.3f}")


def _print_tables(out):
    print("\n================ TABLE 3 (cutoff 0.5) ================")
    df = pd.DataFrame(out["table3"]).T
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
