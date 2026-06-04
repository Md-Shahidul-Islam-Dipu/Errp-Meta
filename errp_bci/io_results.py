"""Results I/O — checkpointing and CSV export (ported verbatim, one audit fix).

AUDIT FIX (see CHANGES.md): ``load_subject_checkpoint`` now normalizes the
``k_shots`` dictionary keys back to ``int``. JSON object keys are always strings,
so a resumed checkpoint came back with string K keys while the rest of the
pipeline (aggregate CSV, Wilcoxon tests) looks them up with integer K — silently
producing NaN aggregates on any resumed run. Normalizing on load makes resume
behavior identical to a fresh run. Checkpoint *write* format is unchanged.
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config


def _ckpt_path(output_dir: str, method: str, seed: int, subject: str) -> Path:
    p = Path(output_dir) / f"seed_{seed}" / "fold_checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{method}_{subject}.json"


def _json_safe(obj):
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (v != v) else v   # NaN → None for JSON
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _normalize_kshot_keys(result: Optional[Dict]) -> Optional[Dict]:
    """Convert integer-like string keys in ``result['k_shots']`` back to int.

    JSON stores dict keys as strings; the rest of the pipeline indexes k_shots
    with integers. Without this, resumed checkpoints yield NaN aggregates/stats.
    """
    if not isinstance(result, dict):
        return result
    ks = result.get('k_shots')
    if isinstance(ks, dict):
        result['k_shots'] = {
            (int(k) if isinstance(k, str) and k.lstrip('-').isdigit() else k): v
            for k, v in ks.items()
        }
    return result


def save_subject_checkpoint(output_dir: str, method: str, seed: int,
                             subject: str, result: Dict) -> None:
    """Save single-subject result immediately after computation (crash-safe)."""
    path = _ckpt_path(output_dir, method, seed, subject)
    with open(path, 'w') as f:
        json.dump(result, f, default=_json_safe)


def load_subject_checkpoint(output_dir: str, method: str, seed: int,
                             subject: str) -> Optional[Dict]:
    """Load previously saved fold result, or None if not found."""
    path = _ckpt_path(output_dir, method, seed, subject)
    if path.exists():
        with open(path) as f:
            return _normalize_kshot_keys(json.load(f))
    return None


def save_per_subject_metrics(results: Dict, method_name: str, seed: int,
                              output_dir: str = Config.RESULTS_DIR) -> None:
    """Write per-subject × per-K metrics to CSV."""
    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, sd in results.get('subjects', {}).items():
        for k, km in sd.get('k_shots', {}).items():
            rows.append({'method': method_name, 'seed': seed, 'subject_id': sid,
                         'k_shots': k,
                         **{m: km.get(m, float('nan'))
                            for m in ['accuracy', 'balanced_accuracy', 'f1_score', 'auroc',
                                      'n_samples', 'n_class_0', 'n_class_1',
                                      'tp', 'fp', 'tn', 'fn']}})
    pd.DataFrame(rows).to_csv(seed_dir / f"{method_name}_per_subject.csv", index=False)


def save_aggregate_metrics(results: Dict, method_name: str, seed: int,
                            k_shots: List[int],
                            output_dir: str = Config.RESULTS_DIR) -> None:
    """Write aggregate (mean ± std across subjects) metrics to CSV."""
    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in k_shots:
        vals = defaultdict(list)
        for sd in results.get('subjects', {}).values():
            km = sd.get('k_shots', {}).get(k, {})
            for m in ['accuracy', 'balanced_accuracy', 'f1_score', 'auroc']:
                v = km.get(m, float('nan'))
                if not (isinstance(v, float) and v != v):  # skip NaN
                    vals[m].append(v)
        row = {'method': method_name, 'seed': seed, 'k_shots': k}
        for m, vs in vals.items():
            row[f'{m}_mean'] = float(np.mean(vs)) if vs else float('nan')
            row[f'{m}_std']  = float(np.std(vs))  if vs else float('nan')
        rows.append(row)
    pd.DataFrame(rows).to_csv(seed_dir / f"{method_name}_aggregate.csv", index=False)
