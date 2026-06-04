"""Reproduce the paper's tables and statistics from the saved ``Results/`` CSVs.

This module is intentionally dependency-light (numpy / pandas / scipy only, no
torch, no datasets) so the paper's numbers can be regenerated locally without a
GPU. It implements exactly the inferential procedure described in the paper:

  * point estimates: mean ± std of the per-seed subject-means (3 seeds);
  * inference: average each subject across seeds, then a paired Wilcoxon
    signed-rank test at N = 16 with the EXACT null distribution, with
    Benjamini-Hochberg FDR correction across all comparisons within a dataset;
  * effect size: rank-biserial r = Z / sqrt(N), reported in [-1, 1].

Usage
-----
    from errp_bci import analysis
    analysis.summary("primary")          # table + macro-F1 status + FDR tests
    analysis.summary("validation")
    analysis.summary("ablation_nofilm")  # the FiLM significance test

``analysis.pca_variance("inria")`` computes the retained-variance figure, but that
one DOES need the dataset + package (it is imported lazily only when called).
"""
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

SEEDS = [42, 123, 456]
KS = [5, 10, 20]

# Experiment -> Results/ subfolder (mirrors errp_bci.experiments).
EXPERIMENT_DIRS = {
    "primary": "Primery",
    "validation": "Validation",
    "ablation_classweighting": "Class wighted",
    "ablation_nofilm": "FiLM vs NoFiLM",
    "ablation_feature": "Feature Representation",
    "ablation_preproc": "Preprocessing Depth",
}

_META = ["Reptile", "Full-MAML", "MAML-ANIL", "SubjectConditioned"]
_PRIMARY_ROWS = ["EEGNet", "Reptile", "SubjectConditioned", "Full-MAML", "MAML-ANIL",
                 "Prototypical", "Matching", "Cov. Align.", "Riemannian",
                 "Pretrain-ZeroShot", "Pretrain-FT", "Supervised"]

# Methods to tabulate (CSV method names) per experiment.
_TABLE_METHODS = {
    "primary": ["EEGNet", "Reptile", "SubjectConditioned", "Full-MAML", "MAML-ANIL",
                "Prototypical", "Matching", "CovarianceAlignment", "Riemannian",
                "Pretrain-ZeroShot", "Pretrain-FT", "Supervised"],
}
_TABLE_METHODS["validation"] = _TABLE_METHODS["primary"]

# Comparison sets per experiment (paper-reported).
def _comparisons(experiment: str) -> List[Tuple[str, str]]:
    if experiment in ("primary", "validation"):
        vs_sup = [(m, "Supervised") for m in
                  ["EEGNet"] + _META + ["Prototypical", "Matching", "Riemannian",
                                        "CovarianceAlignment", "Pretrain-FT", "Pretrain-ZeroShot"]]
        vs_eeg = [(m, "EEGNet") for m in _META]
        vs_pre = [(m, b) for m in _META for b in ("Pretrain-FT", "Pretrain-ZeroShot")]
        return vs_sup + vs_eeg + vs_pre
    if experiment == "ablation_classweighting":
        return [("Full-MAML", "Full-MAML-Uniform"), ("MAML-ANIL", "MAML-ANIL-Uniform")]
    if experiment == "ablation_nofilm":
        return [("SubjectConditioned", "SubjectConditioned_NoFiLM")]
    if experiment == "ablation_feature":
        return [("Full-MAML", "Full-MAML-Raw"), ("SubjectConditioned", "SubjectConditioned-Raw")]
    if experiment == "ablation_preproc":
        return [("Supervised-Full", "Supervised-FilterOnly"),
                ("Supervised-Full", "Supervised-ArtifactBaselineOnly"),
                ("Full-MAML-Full", "Full-MAML-FilterOnly"),
                ("Full-MAML-Full", "Full-MAML-ArtifactBaselineOnly")]
    raise KeyError(experiment)


# ── Data loading ────────────────────────────────────────────────────────────
def _resolve_results_dir(experiment: str, results_root: Optional[str]) -> str:
    sub = EXPERIMENT_DIRS[experiment]
    roots = [results_root, "Results", "/kaggle/working",
             os.path.join(os.getcwd(), "Results"), "."]
    for root in roots:
        if not root:
            continue
        cand = os.path.join(root, sub)
        if glob.glob(os.path.join(cand, "seed_*", "*_per_subject.csv")):
            return cand
    raise FileNotFoundError(
        f"No per-subject CSVs found for experiment {experiment!r} "
        f"(looked for '{sub}/seed_*/' under {roots}).")


def _load(results_dir: str) -> pd.DataFrame:
    frames = [pd.read_csv(f) for s in SEEDS
              for f in glob.glob(os.path.join(results_dir, f"seed_{s}", "*_per_subject.csv"))]
    if not frames:
        raise FileNotFoundError(f"No per-subject CSVs under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def _subject_means(df: pd.DataFrame, method: str, k: int, metric: str) -> pd.Series:
    """Per-subject value averaged across seeds (the unit of inference)."""
    sub = df[(df.method == method) & (df.k_shots == k)]
    return sub.groupby("subject_id")[metric].mean()


# ── Statistics ──────────────────────────────────────────────────────────────
def _bh_fdr(pvals: np.ndarray, alpha: float = 0.05):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out, out < alpha


def _boot_median_ci(diff, n_boot=10000, seed=0):
    """Median paired difference + 95% bootstrap CI over subjects (magnitude that,
    unlike p/r, does not saturate when all subjects improve)."""
    rng = np.random.default_rng(seed)
    med = float(np.median(diff))
    boots = np.median(rng.choice(diff, size=(n_boot, len(diff)), replace=True), axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return med, float(lo), float(hi)


def _wilcoxon_pair(df, a, b, k, metric):
    va, vb = _subject_means(df, a, k, metric), _subject_means(df, b, k, metric)
    idx = va.index.intersection(vb.index)
    x, y = va[idx].to_numpy(), vb[idx].to_numpy()
    n = len(idx)
    diff = x - y
    if n < 2 or np.allclose(diff, 0):
        return dict(k=k, n=n, mean1=float(np.mean(x)) if n else np.nan,
                    mean2=float(np.mean(y)) if n else np.nan,
                    delta=float(np.mean(diff)) if n else np.nan, p=1.0, r=0.0,
                    median_diff=0.0, ci_lo=0.0, ci_hi=0.0)
    try:
        _, p = wilcoxon(x, y, alternative="two-sided", mode="exact")
    except Exception:
        _, p = wilcoxon(x, y, alternative="two-sided")
    z = norm.ppf(1 - p / 2) * np.sign(np.mean(diff))
    r = float(np.clip(z / np.sqrt(n), -1, 1))
    med, lo, hi = _boot_median_ci(diff)
    return dict(k=k, n=n, mean1=float(np.mean(x)), mean2=float(np.mean(y)),
                delta=float(np.mean(diff)), p=float(p), r=r,
                median_diff=med, ci_lo=lo, ci_hi=hi)


def pairwise_tests(experiment: str, results_root: Optional[str] = None,
                   metric: str = "balanced_accuracy", ks: List[int] = None) -> pd.DataFrame:
    """Seed-averaged N=16 exact Wilcoxon + BH-FDR (within experiment) for every
    paper comparison at every K. Returns a tidy DataFrame."""
    ks = ks or KS
    df = _load(_resolve_results_dir(experiment, results_root))
    rows = []
    for a, b in _comparisons(experiment):
        for k in ks:
            rows.append({"method1": a, "method2": b, **_wilcoxon_pair(df, a, b, k, metric)})
    out = pd.DataFrame(rows)
    p_fdr, sig = _bh_fdr(out["p"].to_numpy())
    out["p_fdr"] = p_fdr
    out["significant"] = sig
    return out


# ── Tables ──────────────────────────────────────────────────────────────────
def accuracy_table(experiment: str, results_root: Optional[str] = None,
                   metric: str = "balanced_accuracy") -> pd.DataFrame:
    """mean ± std (over the 3 seed-level means) per method × K, plus AUROC@K=5."""
    df = _load(_resolve_results_dir(experiment, results_root))
    methods = _TABLE_METHODS.get(experiment, sorted(df.method.unique()))
    methods = [m for m in methods if m in set(df.method)]
    rows = []
    for m in methods:
        row = {"method": m}
        for k in KS:
            sm = df[(df.method == m) & (df.k_shots == k)].groupby("seed")[metric].mean()
            row[f"K={k}"] = f"{sm.mean():.3f}±{sm.std():.3f}" if len(sm) else "--"
        au = df[(df.method == m) & (df.k_shots == 5)].groupby("seed")["auroc"].mean()
        row["AUROC@5"] = f"{au.mean():.3f}" if len(au) else "--"
        rows.append(row)
    return pd.DataFrame(rows)


def _macro_f1_row(r):
    tp, fp, tn, fn = r["tp"], r["fp"], r["tn"], r["fn"]
    if any(pd.isna(v) for v in (tp, fp, tn, fn)):
        return np.nan

    def f1(pn, pd_, rn, rd):
        p = pn / pd_ if pd_ > 0 else 0.0
        rc = rn / rd if rd > 0 else 0.0
        return 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
    return (f1(tn, tn + fn, tn, tn + fp) + f1(tp, tp + fp, tp, tp + fn)) / 2.0


def macro_f1_table(experiment: str, results_root: Optional[str] = None) -> pd.DataFrame:
    """Macro-F1 from the persisted confusion counts, where available. Neural
    methods averaged the counts away per fold, so they show NaN (need a re-run)."""
    df = _load(_resolve_results_dir(experiment, results_root)).copy()
    df["macro_f1"] = df.apply(_macro_f1_row, axis=1)
    rows = []
    for m in (_TABLE_METHODS.get(experiment) or sorted(df.method.unique())):
        sub = df[df.method == m]
        if sub.empty:
            continue
        avail = sub["macro_f1"].notna().any()
        cell = {}
        for k in KS:
            sm = sub[sub.k_shots == k].groupby("seed")["macro_f1"].mean()
            cell[f"K={k}"] = f"{sm.mean():.3f}" if avail and len(sm) else "n/a (re-run)"
        rows.append({"method": m, **cell})
    return pd.DataFrame(rows)


# ── One-call summary ────────────────────────────────────────────────────────
def summary(experiment: str, results_root: Optional[str] = None,
            save_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Print and return the paper's table + macro-F1 status + FDR tests for one
    experiment. If ``save_dir`` is given, also write the three CSVs there."""
    acc = accuracy_table(experiment, results_root)
    mf1 = macro_f1_table(experiment, results_root)
    tests = pairwise_tests(experiment, results_root)

    bar = "=" * 70
    print(f"\n{bar}\n{experiment} — balanced accuracy (mean±std over seeds), AUROC@K=5\n{bar}")
    print(acc.to_string(index=False))
    print(f"\n-- macro-F1 (from persisted confusion counts; 'n/a' = needs re-run) --")
    print(mf1.to_string(index=False))
    print(f"\n-- paired Wilcoxon, subject-level N=16, exact, BH-FDR within dataset --")
    show = tests[["method1", "method2", "k", "n", "mean1", "mean2",
                  "delta", "p", "p_fdr", "r", "significant"]].copy()
    for c in ("mean1", "mean2", "delta", "r"):
        show[c] = show[c].map(lambda v: f"{v:+.3f}" if c == "delta" else f"{v:.3f}")
    show["p"] = tests["p"].map(lambda v: f"{v:.2e}")
    show["p_fdr"] = tests["p_fdr"].map(lambda v: f"{v:.2e}")
    print(show.to_string(index=False))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        acc.to_csv(os.path.join(save_dir, f"{experiment}_table.csv"), index=False)
        mf1.to_csv(os.path.join(save_dir, f"{experiment}_macro_f1.csv"), index=False)
        tests.to_csv(os.path.join(save_dir, f"{experiment}_wilcoxon_fdr.csv"), index=False)
        print(f"\nSaved CSVs to {save_dir}/")
    return {"table": acc, "macro_f1": mf1, "tests": tests}


def pca_variance(dataset: str, n_components: int = 32,
                 dataset_root: Optional[str] = None) -> Tuple[float, float]:
    """Mean ± std of variance retained by the per-fold PCA. Needs the dataset and
    the full package (torch/mne); imported lazily so this module stays light."""
    from .config import ExperimentConfig            # noqa: lazy (pulls torch)
    from .data.loaders import load_dataset, build_features_and_loso
    cfg = ExperimentConfig(name="pca", dataset=dataset, dataset_root=dataset_root)
    _, loso = build_features_and_loso(load_dataset(cfg), n_components)
    vr = np.array([loso[s]["pca"].pca.explained_variance_ratio_.sum() for s in loso])
    print(f"{dataset}: PCA({n_components}) retains "
          f"{vr.mean()*100:.1f}% ± {vr.std()*100:.1f}% across {len(vr)} folds")
    return float(vr.mean()), float(vr.std())
