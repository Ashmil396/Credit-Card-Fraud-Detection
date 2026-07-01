# Feature Engineering for Credit-Card Fraud Detection with Deep Learning

A faithful re-implementation of the **methodology** of

> Zhang, X., Han, Y., Xu, W., & Wang, Q. (2021). *HOBA: A novel feature engineering
> methodology for credit card fraud detection with a deep learning architecture.*
> Information Sciences, 557, 302–316.

Built for a course presentation. The slide deck is `Credit_Card_Fraud_Detection.pptx`.

---

## What is and isn't replicated

The original study uses a **proprietary** dataset from a large Chinese commercial
bank (153,685 records). That data was never released, and the only public
credit-card fraud dataset (Kaggle/ULB) is **PCA-anonymised** — it contains none of
the raw fields HOBA is defined on (transaction country, MCC, entry mode, account,
open-to-buy, timestamp). HOBA therefore *cannot* be applied to it.

So this project replicates the **framework**, not the bank's data:

* **Synthetic data** is generated with the exact attributes of the paper's Table 1,
  and the three behaviour-fraud archetypes it names (theft/stolen, counterfeit,
  card-not-present). Fraud is deliberately *ordinary in amount/frequency* but
  *abnormal in location, open-to-buy drain, channel and time* — so the RFM
  benchmark is genuinely handicapped, exactly as the paper argues.
* Resulting dataset: **118,063 transactions, 1.32% fraud** (paper: 1.33%).

Because synthetic fraud is cleaner than real fraud, **absolute scores run higher
than the paper's**. What replicates is the *ordering and qualitative conclusions* —
which is the point of a methodology replication.

---

## Files

| File | Purpose |
|------|---------|
| `data_gen.py` | Synthetic transaction generator (Table 1 schema + 3 fraud archetypes) |
| `feature_engineering.py` | HOBA (518 vars) and RFM (41 vars) feature builders |
| `deep_models.py` | DBN (RBM pre-train + fine-tune), CNN, RNN in PyTorch |
| `run_experiments.py` | 6 classifiers × 2 feature sets; Tables 3 / 5a / 5b; ROC |
| `finalize.py` | Merges runs → `results_tables.md` + `fig6_roc.png` |
| `results_tables.md` | Tables 3, 5a, 5b (markdown) |
| `fig6_roc.png` | Fig. 6 replicated (ROC curves, HOBA vs RFM) |
| `Credit_Card_Fraud_Detection.pptx` | Presentation deck |

## How to run

```bash
pip install torch scikit-learn pandas pyarrow matplotlib --break-system-packages
python data_gen.py                 # -> raw_transactions.parquet
python feature_engineering.py      # -> X_hoba.parquet, X_rfm.parquet, meta.parquet
python run_experiments.py RFM      # -> results_RFM.json,  roc_RFM.npz
python run_experiments.py HOBA     # -> results_HOBA.json, roc_HOBA.npz
python finalize.py                 # -> results_tables.md, fig6_roc.png
```

---

## How the code maps to the paper

| Paper element | Where |
|---------------|-------|
| Table 1 raw transaction schema | `data_gen.py` (`_row`) |
| Behaviour-fraud types (§2.1) | `data_gen.py` (`_inject_fraud`) |
| HOBA: characteristic × period × measure × statistic (§3.3) | `feature_engineering.build_hoba` |
| Geographic distance & open-to-buy measures (§3.3) | `feature_engineering._prep` |
| Rule-based binary variables (§3.3) | `build_hoba` (`R_*` columns) |
| RFM benchmark (Van Vlasselaer) | `feature_engineering.build_rfm` |
| DBN via stacked RBMs + fine-tune (§3.4.1) | `deep_models.RBM`, `DBN` |
| CNN over recent-transaction matrix (§3.4.2) | `deep_models.CNNNet` |
| Elman RNN over the sequence (§3.4.3) | `deep_models.RNNNet` |
| Evaluation metrics + AUC (§4.3) | `run_experiments.metrics_at_threshold` |
| Performance under FPR tolerance (§4.4.2) | `run_experiments.threshold_for_fpr` |

---

## Headline results (this replication)

* **HOBA improves every classifier.** Mean F1 gain ≈ **+0.19** at the 0.5 cutoff;
  largest is DBN (**+0.375**, 0.47 → 0.85).
* **DBN + HOBA is the best overall model**, AUC **0.997** of all 12 (classifier ×
  feature-set) combinations — matching the paper's headline.
* **Practical (≤3% FPR):** DBN+HOBA recovers **~97%** of fraud vs ~89% for DBN+RFM,
  echoing the paper's claim that HOBA captures most fraud at an acceptable alarm rate.
* **Caveat:** CNN under-performed here (AUC 0.93), sensitive to how the
  recent-transaction feature matrix is built and to the CPU training budget.

See `results_tables_1.md` for the full Tables 3 / 5a / 5b.
