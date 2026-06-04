"""Results tables, adaptation-curve / heatmap plots, combined CSV.

Auto-ported verbatim from the legacy notebooks by _build_pkg.py.
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None

from .config import Config

def print_results_table(results_by_seed: dict, k_value: int = 10,
                         metric: str = 'balanced_accuracy') -> pd.DataFrame:
    """Print mean±std for a given K and metric, sorted descending."""
    all_methods = sorted({m for sr in results_by_seed.values() for m in sr})
    rows = []
    for method in all_methods:
        vals = []
        for sr in results_by_seed.values():
            if method not in sr: continue
            for sd in sr[method].get('subjects', {}).values():
                v = sd.get('k_shots', {}).get(k_value, {}).get(metric)
                if v is not None and not (isinstance(v, float) and v != v):
                    vals.append(float(v))
        if not vals: continue
        rows.append({'Method': method,
                     f'Mean': f"{np.mean(vals):.4f}",
                     f'Std':  f"{np.std(vals):.4f}",
                     'N': len(vals)})

    df = pd.DataFrame(rows).sort_values('Mean', ascending=False)
    print(f"\n{'='*55}")
    print(f"Metric: {metric.replace('_',' ').title()}  |  K={k_value}")
    print(f"Chance: 0.5000  |  Seeds={len(results_by_seed)}  |  Subjects≈26")
    print(f"{'='*55}")
    print(df.to_string(index=False))
    return df

METHOD_COLORS = {
    'Supervised':          '#2ECC71',
    'Full-MAML':           '#E74C3C',
    'MAML-ANIL':           '#E67E22',
    'SubjectConditioned':  '#9B59B6',
    'Reptile':             '#3498DB',
    'Prototypical':        '#1ABC9C',
    'Matching':            '#F39C12',
    'Riemannian':          '#27AE60',
    'CovarianceAlignment': '#16A085',
    'EEGNet':              '#8E44AD',
}

METHOD_MARKERS = {
    'Supervised': 'D', 'Full-MAML': 's', 'MAML-ANIL': 'h',
    'SubjectConditioned': '*', 'Reptile': '<', 'Prototypical': '>',
    'Matching': 'X', 'Riemannian': 'd', 'CovarianceAlignment': 'P', 'EEGNet': 'o',
}

def _extract_k_metric(results_by_seed, method, k_shots, metric):
    out = {}
    for k in k_shots:
        vals = []
        for sr in results_by_seed.values():
            if method not in sr: continue
            for sd in sr[method].get('subjects', {}).values():
                v = sd.get('k_shots', {}).get(k, {}).get(metric)
                if v is not None and not (isinstance(v, float) and v != v):
                    vals.append(float(v))
        if vals:
            out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out

def plot_adaptation_curves(results_by_seed, metric='balanced_accuracy', save_path=None):
    fig, ax = plt.subplots(figsize=(12, 7))
    all_methods = sorted({m for sr in results_by_seed.values() for m in sr})

    for method in all_methods:
        kd = _extract_k_metric(results_by_seed, method, Config.K_SHOTS, metric)
        if not kd: continue
        ks = sorted(kd.keys())
        means = [kd[k][0] for k in ks]
        stds  = [kd[k][1] for k in ks]
        color  = METHOD_COLORS.get(method, '#888888')
        marker = METHOD_MARKERS.get(method, 'o')
        ax.plot(ks, means, marker=marker, linewidth=2.2, markersize=9,
                label=method, color=color, zorder=3)
        ax.fill_between(ks,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.12, color=color, zorder=2)

    ax.axhline(0.5, color='red', linestyle=':', linewidth=1.8, alpha=0.6,
               label='Chance (0.50)', zorder=1)
    ax.set_xlabel('K-shots (support examples)', fontsize=14, fontweight='bold')
    ylabel = metric.replace('_', ' ').title()
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(f'{ylabel} vs K-shots\n(mean ± std over 26 subjects × 3 seeds)',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(Config.K_SHOTS)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def plot_per_subject_heatmap(results_by_seed, method1='SubjectConditioned',
                              method2='Supervised', k=10,
                              metric='balanced_accuracy', save_path=None):
    """Heatmap of per-subject performance gain: method1 − method2."""
    m1_s = defaultdict(list)
    m2_s = defaultdict(list)
    for sr in results_by_seed.values():
        for m, store in [(method1, m1_s), (method2, m2_s)]:
            if m not in sr: continue
            for sid, sd in sr[m].get('subjects', {}).items():
                v = sd.get('k_shots', {}).get(k, {}).get(metric)
                if v is not None and not (isinstance(v, float) and v != v):
                    store[sid].append(float(v))

    common = sorted(set(m1_s) & set(m2_s))
    if not common:
        print("No common subjects found for heatmap.")
        return

    diffs = [np.mean(m1_s[s]) - np.mean(m2_s[s]) for s in common]

    fig, ax = plt.subplots(figsize=(max(14, len(common) * 0.55), 3.5))
    im = ax.imshow([diffs], cmap='RdYlGn', aspect='auto', vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(len(common)))
    ax.set_xticklabels(common, rotation=45, ha='right', fontsize=9)
    ax.set_yticks([])
    ax.set_title(
        f'{metric.replace("_"," ").title()} gain: {method1} − {method2}  |  K={k}'
        f'  |  green=method1 better, red=worse',
        fontsize=12)

    for i, (sid, d) in enumerate(zip(common, diffs)):
        ax.text(i, 0, f'{d:+.3f}', ha='center', va='center',
                fontsize=7, color='black', fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def build_combined_results_csv(results_by_seed: dict, k_shots: list,
                                output_dir: str = Config.RESULTS_DIR) -> pd.DataFrame:
    """Build a combined CSV: mean ± std per method × K across ALL seeds."""
    all_methods = sorted({m for sr in results_by_seed.values() for m in sr})
    rows = []
    for method in all_methods:
        for k in k_shots:
            vals = defaultdict(list)
            for sr in results_by_seed.values():
                if method not in sr: continue
                for sd in sr[method].get('subjects', {}).values():
                    km = sd.get('k_shots', {}).get(k, {})
                    for m in ['accuracy', 'balanced_accuracy', 'f1_score', 'auroc']:
                        v = km.get(m)
                        if v is not None and not (isinstance(v, float) and v != v):
                            vals[m].append(float(v))
            row = {'method': method, 'k_shots': k,
                   'n_seeds': len(results_by_seed)}
            for m in ['accuracy', 'balanced_accuracy', 'f1_score', 'auroc']:
                vs = vals[m]
                row[f'{m}_mean'] = float(np.mean(vs)) if vs else float('nan')
                row[f'{m}_std']  = float(np.std(vs))  if vs else float('nan')
                row[f'{m}_n']    = len(vs)
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(['k_shots', 'balanced_accuracy_mean'],
                                         ascending=[True, False])
    out_path = Path(output_dir) / "combined_results_all_seeds.csv"
    df.to_csv(out_path, index=False)
    print(f"Combined results saved to: {out_path}")
    return df
