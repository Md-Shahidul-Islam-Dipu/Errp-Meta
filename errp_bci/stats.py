"""Statistical tests with effect sizes (ported verbatim; optional BH-FDR added).

``perform_statistical_tests`` gains an opt-in ``fdr`` flag (default False) that
appends Benjamini-Hochberg corrected p-values and a corrected significance flag
(addresses flaw #7 in paper/flaws_to_fix.md). With ``fdr=False`` the CSV is
byte-identical to the original notebooks.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .config import Config


def paired_wilcoxon(results1: Dict, results2: Dict, k: int,
                    metric: str = 'balanced_accuracy') -> Dict:
    """Paired Wilcoxon signed-rank test at a given K, with rank-biserial effect size."""
    from scipy.stats import norm

    def _get(r, sid, k, m):
        v = r.get('subjects', {}).get(sid, {}).get('k_shots', {}).get(k, {}).get(m)
        return None if (v is None or (isinstance(v, float) and v != v)) else float(v)

    subs = sorted(set(results1.get('subjects', {})) & set(results2.get('subjects', {})))
    pairs = [(a, b) for s in subs
             for a, b in [(_get(results1, s, k, metric), _get(results2, s, k, metric))]
             if a is not None and b is not None]

    if len(pairs) < 2:
        return {'error': f'Only {len(pairs)} paired subjects', 'n': len(pairs)}

    a1 = np.array([p[0] for p in pairs])
    a2 = np.array([p[1] for p in pairs])
    diff = a1 - a2

    try:
        stat, p = wilcoxon(a1, a2, alternative='two-sided')
    except ValueError as e:
        return {'error': str(e), 'n': len(pairs)}

    n = len(pairs)
    z = norm.ppf(1 - p / 2) * np.sign(np.mean(diff))
    r = z / np.sqrt(n)
    interp = ('negligible' if abs(r) < 0.1
              else 'small' if abs(r) < 0.3
              else 'medium' if abs(r) < 0.5 else 'large')

    return {'metric': metric, 'k': k, 'n': n,
            'mean1': float(np.mean(a1)), 'mean2': float(np.mean(a2)),
            'mean_diff': float(np.mean(diff)), 'statistic': float(stat),
            'p_value': float(p), 'significant': bool(p < 0.05),
            'effect_r': float(r), 'effect_interp': interp}


def _benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05):
    """Return (rejected_mask, adjusted_pvalues) for BH-FDR."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity (cumulative min from the largest p)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    rejected = out < alpha
    return rejected, out


def perform_statistical_tests(results_dict: Dict[str, Dict],
                               k_shots: List[int],
                               comparisons: List[Tuple[str, str]],
                               output_dir: str = Config.RESULTS_DIR,
                               seed: int = 42,
                               fdr: bool = False) -> pd.DataFrame:
    """Run all pairwise tests and save to CSV.

    If ``fdr`` is True, add ``p_value_fdr`` and ``significant_fdr`` columns
    (Benjamini-Hochberg across every valid comparison in this call).
    """
    rows = []
    for m1, m2 in comparisons:
        if m1 not in results_dict or m2 not in results_dict:
            continue
        for k in k_shots:
            row = paired_wilcoxon(results_dict[m1], results_dict[m2], k)
            rows.append({'method1': m1, 'method2': m2, **row})

    df = pd.DataFrame(rows)
    if fdr and 'p_value' in df.columns:
        valid = df['p_value'].notna()
        if valid.any():
            rej, adj = _benjamini_hochberg(df.loc[valid, 'p_value'].values)
            df.loc[valid, 'p_value_fdr'] = adj
            df.loc[valid, 'significant_fdr'] = rej

    out_dir = Path(output_dir) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "statistical_tests.csv", index=False)
    return df
