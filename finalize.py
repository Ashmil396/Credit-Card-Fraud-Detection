"""finalize.py -- merge results, print Tables 3/5a/5b, draw Fig.6 ROC curves."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["BPNN", "SVM", "RF", "DBN", "CNN", "RNN"]


def load(tag):
    with open(f"/home/claude/hoba/results_{tag}.json") as f:
        return json.load(f)


def merge(key):
    r = load("RFM")[key]; h = load("HOBA")[key]
    return {**r, **h}


def as_table(d, cols):
    rows = {}
    for fs in ["RFM", "HOBA"]:
        for m in ORDER:
            k = f"{m}+{fs}"
            if k in d:
                rows[k] = {c: d[k][c] for c in cols}
    return pd.DataFrame(rows).T[cols]


def md_table(df, title, pct_cols=(), float_cols=()):
    out = [f"### {title}\n"]
    hdr = "| Classifier | " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    out += [hdr, sep]
    for idx, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if c in pct_cols:
                cells.append(f"{v*100:.2f}%")
            else:
                cells.append(f"{v:.3f}")
        out.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# ---- Table 3 -------------------------------------------------------------
t3 = merge("table3")
df3 = as_table(t3, ["F1", "Precision", "Recall", "Accuracy", "AUC"])

# ---- Table 5a / 5b -------------------------------------------------------
t5a = merge("fpr1"); t5b = merge("fpr3")
df5a = as_table(t5a, ["F1", "Precision", "Recall", "Accuracy"])
df5b = as_table(t5b, ["F1", "Precision", "Recall", "Accuracy"])

report = ["# Replication results\n"]
report.append(md_table(df3, "Table 3 — performance at 0.5 cutoff",
                       pct_cols=["Precision", "Recall", "Accuracy"]))
report.append(md_table(df5a, "Table 5a — performance at <=1% FPR",
                       pct_cols=["Precision", "Recall", "Accuracy"]))
report.append(md_table(df5b, "Table 5b — performance at <=3% FPR",
                       pct_cols=["Precision", "Recall", "Accuracy"]))

with open("/home/claude/hoba/results_tables.md", "w") as f:
    f.write("\n".join(report))

print("\n".join(report))

# ---- Fig 6 ROC curves ----------------------------------------------------
roc_r = np.load("/home/claude/hoba/roc_RFM.npz")
roc_h = np.load("/home/claude/hoba/roc_HOBA.npz")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, roc, fsname in [(axes[0], roc_h, "HOBA"), (axes[1], roc_r, "RFM")]:
    for m in ORDER:
        k = f"{m}+{fsname}"
        if k in roc.files:
            fpr, tpr = roc[k]
            ax.plot(fpr, tpr, lw=1.6, label=m)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC curves — {fsname} feature set")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
fig.suptitle("Fig. 6 (replicated): ROC curves for different approaches", y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig("/home/claude/hoba/fig6_roc.png", dpi=140, bbox_inches="tight")
print("\nsaved fig6_roc.png and results_tables.md")

# ---- summary deltas ------------------------------------------------------
print("\nHOBA minus RFM (AUC / F1) per classifier:")
for m in ORDER:
    a = t3[f"{m}+HOBA"]["AUC"] - t3[f"{m}+RFM"]["AUC"]
    f1 = t3[f"{m}+HOBA"]["F1"] - t3[f"{m}+RFM"]["F1"]
    print(f"  {m:5s}  dAUC={a:+.3f}  dF1={f1:+.3f}")
