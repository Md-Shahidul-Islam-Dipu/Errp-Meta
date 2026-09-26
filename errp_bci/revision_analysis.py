"""Tables and statistics for the camera-ready revision (numpy/pandas/scipy only).

Reads the ``draws.csv`` / ``timing.csv`` / ``burden.csv`` written by
``errp_bci.revision`` and reproduces the same inferential procedure as the
paper: average each subject over draws (and seeds), paired exact Wilcoxon at
N=16, BH-FDR within each family, median paired difference with a 95% bootstrap
CI. Single-draw statistics are computed on PAIRED draws (every method saw the
same support trials in a given draw).

    from errp_bci import revision_analysis as ra
    ra.report("Results/Revision_inria")
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

from .analysis import _bh_fdr, _boot_median_ci

META = ["Reptile", "Full-MAML", "MAML-ANIL", "SubjectConditioned"]

# E1: realistic-calibration family (tested at every protocol, K=4)
E1_PAIRS = [(m, b) for m in META for b in ("Supervised", "EEGNet", "EEGNet-FT")]
# E2/E3: input-representation 2x2 + fine-tuned EEGNet (balanced protocol)
E2_PAIRS = [("MAML-ANIL", "PCA-Probe"),        # meta vs probe, PCA input
            ("ANIL-EEGNet", "EEGNet"),         # meta vs probe, raw input
            ("MAML-ANIL", "ANIL-EEGNet"),      # input effect, ANIL
            ("Reptile", "Reptile-EEGNet"),     # input effect, Reptile
            ("EEGNet-FT", "EEGNet"),           # fine-tuning vs probe
            ("Reptile", "EEGNet-FT"),          # headline vs strongest EEGNet
            ("ANIL-EEGNet", "Supervised"), ("Reptile-EEGNet", "Supervised"),
            ("EEGNet-ZeroShot", "Reptile"),    # unadapted raw CNN vs PCA meta-learner
            ("ANIL-EEGNet", "EEGNet-ZeroShot"),  # meta-training vs plain pretraining
            ("ANIL-EEGNet", "EEGNet-FT"),
            ("ANIL-EEGNet", "ANIL-EEGNet-ZeroShot"),  # what test-time adaptation buys
            ("Reptile-EEGNet", "Reptile-EEGNet-ZeroShot")]
# E5: Riemannian comparators
E5_PAIRS = [("Reptile", "MDWM"), ("Reptile", "xDAWN-MDM"),
            ("MDWM", "Supervised"), ("xDAWN-MDM", "Supervised")]


# Label coding differs by corpus: INRIA keeps TrainLabels.csv's `Prediction`
# (1 = CORRECT feedback, 0 = error); coadaptation uses 1 = error. Balanced
# accuracy / AUROC are symmetric, but error-rate, TPR/TNR and "error trials in
# the support set" are not, so INRIA rows are recoded to error = positive here.
ERROR_LABEL = {"inria": 0, "coadaptation": 1}


def _errors_positive(r: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    d = r["draws"]
    flip = d.dataset.map(ERROR_LABEL).fillna(1).eq(0)
    if flip.any():
        d = d.copy()
        d.loc[flip, "n_err_support"] = d.loc[flip, "k"] - d.loc[flip, "n_err_support"]
        for a, b in (("tpr", "tnr"), ("tp", "tn"), ("fp", "fn")):
            va, vb = d[a].copy(), d[b].copy()
            d[a] = va.where(~flip, vb)
            d[b] = vb.where(~flip, va)
        r["draws"] = d
    if "burden" in r:
        bu = r["burden"].copy()
        f = bu.dataset.map(ERROR_LABEL).fillna(1).eq(0)
        bu.loc[f, "error_rate"] = 1 - bu.loc[f, "error_rate"]   # burden itself is symmetric
        r["burden"] = bu
    return r


def load(out_dir: str) -> Dict[str, pd.DataFrame]:
    r = {n: pd.read_csv(os.path.join(out_dir, f"{n}.csv"))
         for n in ("draws", "timing", "burden") if os.path.exists(os.path.join(out_dir, f"{n}.csv"))}
    return _errors_positive(r)


def subject_means(d: pd.DataFrame, method: str, protocol: str, k: int,
                  metric: str = "balanced_accuracy") -> pd.Series:
    s = d[(d.method == method) & (d.protocol == protocol) & (d.k == k)]
    return s.groupby("subject")[metric].mean()


def _test(d, a, b, protocol, k, metric="balanced_accuracy") -> dict:
    va, vb = subject_means(d, a, protocol, k, metric), subject_means(d, b, protocol, k, metric)
    idx = va.index.intersection(vb.index)
    x, y = va[idx].to_numpy(), vb[idx].to_numpy()
    diff = x - y
    row = dict(a=a, b=b, protocol=protocol, k=k, n=len(idx),
               mean_a=float(np.mean(x)) if len(x) else np.nan,
               mean_b=float(np.mean(y)) if len(y) else np.nan,
               delta=float(np.mean(diff)) if len(x) else np.nan,
               n_better=int((diff > 0).sum()))
    if len(idx) < 5 or np.allclose(diff, 0):
        return {**row, "p": 1.0, "r": 0.0, "median": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    try:
        _, p = wilcoxon(x, y, alternative="two-sided", method="exact")
    except TypeError:
        _, p = wilcoxon(x, y, alternative="two-sided", mode="exact")
    r = float(np.clip(norm.ppf(1 - p / 2) * np.sign(np.mean(diff)) / np.sqrt(len(idx)), -1, 1))
    med, lo, hi = _boot_median_ci(diff)
    return {**row, "p": float(p), "r": r, "median": med, "ci_lo": lo, "ci_hi": hi}


def family(d: pd.DataFrame, pairs, protocols: Sequence[str], ks: Sequence[int]) -> pd.DataFrame:
    rows = [_test(d, a, b, pr, k) for pr in protocols for k in ks for a, b in pairs
            if a in set(d.method) and b in set(d.method)]
    df = pd.DataFrame(rows)
    if len(df):
        df["p_fdr"], df["sig"] = _bh_fdr(df.p.to_numpy())
    return df


def single_draw(d: pd.DataFrame, a: str, b: str, protocol: str, k: int) -> dict:
    """Paired single-draw view: how often does A beat B on the SAME support
    trials, and what does the worst decile of draws look like?"""
    key = ["seed", "subject", "draw"]
    A = d[(d.method == a) & (d.protocol == protocol) & (d.k == k)].set_index(key).balanced_accuracy
    B = d[(d.method == b) & (d.protocol == protocol) & (d.k == k)].set_index(key).balanced_accuracy
    idx = A.index.intersection(B.index)
    diff = (A[idx] - B[idx]).to_numpy()
    return dict(a=a, b=b, protocol=protocol, k=k, n_draws=len(idx),
                p_win=float((diff > 0).mean()), p_tie=float((diff == 0).mean()),
                mean_diff=float(diff.mean()),
                a_p10=float(np.percentile(A[idx], 10)), b_p10=float(np.percentile(B[idx], 10)),
                a_sd_draw=float(A[idx].groupby(level=[0, 1]).std().mean()),
                b_sd_draw=float(B[idx].groupby(level=[0, 1]).std().mean()))


def accuracy_table(d: pd.DataFrame, protocol: str, ks=(4, 10, 20),
                   metric: str = "balanced_accuracy") -> pd.DataFrame:
    out = {}
    for m in sorted(d.method.unique()):
        out[m] = {k: subject_means(d, m, protocol, k, metric).mean() for k in ks}
    t = pd.DataFrame(out).T
    t.columns = [f"K={k}" for k in ks]
    return t.sort_values(t.columns[0], ascending=False)


def zero_error_rate(d: pd.DataFrame) -> pd.DataFrame:
    """Share of calibration draws that contain no error trial."""
    one = d.drop_duplicates(["seed", "subject", "protocol", "k", "draw"])
    return one.assign(no_err=one.n_err_support == 0).groupby(["protocol", "k"]).no_err.mean().unstack()


def application(d: pd.DataFrame, burden: Optional[pd.DataFrame], protocol="balanced", k=4,
                methods=("ANIL-EEGNet", "EEGNet-ZeroShot", "EEGNet-FT", "Reptile",
                         "SubjectConditioned", "EEGNet", "Supervised")) -> pd.DataFrame:
    """Per 100 selections at the dataset's own error rate e: errors caught
    (e*TPR), correct selections falsely flagged ((1-e)*(1-TNR)), and the net
    under undo-on-detection (caught - false flags)."""
    e = float(burden.drop_duplicates("subject").error_rate.mean()) if burden is not None else 0.3
    rows = []
    for m in methods:
        s = d[(d.method == m) & (d.protocol == protocol) & (d.k == k)]
        if not len(s):
            continue
        g = s.groupby("subject")[["tpr", "tnr"]].mean()
        tpr, tnr = g.tpr.mean(), g.tnr.mean()
        caught, false = 100 * e * tpr, 100 * (1 - e) * (1 - tnr)
        rows.append(dict(method=m, error_rate=e, tpr=tpr, tnr=tnr, caught=caught,
                         false_flags=false, net=caught - false,
                         residual_err=100 * e * (1 - tpr)))
    return pd.DataFrame(rows)


def burden_summary(burden: pd.DataFrame) -> pd.DataFrame:
    return burden.groupby("per_class")[["error_rate", "mean", "median", "p90"]].median()


def timing_summary(t: pd.DataFrame) -> pd.DataFrame:
    return t.groupby("group").agg(train_s_median=("train_s", "median"),
                                  train_s_total=("train_s", "sum"),
                                  adapt_ms=("adapt_ms_mean", "median"),
                                  device=("device", "first")).sort_values("train_s_median")


def report(out_dir: str, save: bool = True) -> Dict[str, pd.DataFrame]:
    r = load(out_dir)
    d, burden = r["draws"], r.get("burden")
    res = {}
    for pr in ("balanced", "natural", "contiguous"):
        res[f"acc_{pr}"] = accuracy_table(d, pr)
    res["auroc_balanced"] = accuracy_table(d, "balanced", metric="auroc")
    res["zero_error"] = zero_error_rate(d)
    res["E1"] = family(d, E1_PAIRS, ("balanced", "natural", "contiguous"), (4,))
    res["E1_single_draw"] = pd.DataFrame(
        [single_draw(d, a, b, pr, 4) for pr in ("balanced", "natural", "contiguous")
         for a, b in (("Reptile", "Supervised"), ("Reptile", "EEGNet"),
                      ("SubjectConditioned", "Supervised"), ("SubjectConditioned", "EEGNet"))
         if a in set(d.method) and b in set(d.method)])
    res["E2E3"] = family(d, E2_PAIRS, ("balanced",), (4, 10, 20))
    res["E5"] = family(d, E5_PAIRS, ("balanced", "contiguous"), (4, 10, 20))
    res["application"] = application(d, burden)
    if burden is not None:
        res["burden"] = burden_summary(burden)
    if "timing" in r:
        res["timing"] = timing_summary(r["timing"])
    pd.set_option("display.width", 200)
    for name, t in res.items():
        print(f"\n=== {name} ===")
        print(t.round(3).to_string())
        if save:
            t.to_csv(os.path.join(out_dir, f"table_{name}.csv"))
    return res
