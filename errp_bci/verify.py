"""LOSO isolation verification.

Auto-ported verbatim from the legacy notebooks by _build_pkg.py.
"""
import numpy as np
from typing import Dict

from .config import Config
from .models.maml import MAML_Encoder

def verify_loso_isolation(loso_splits: dict, verbose: bool = True) -> bool:
    """Comprehensive LOSO isolation check.

    Verifies:
    (a) Test subject not in own training set
    (b) PCA fitted and output dimension correct
    (c) No NaN or Inf in test features
    """
    all_ok = True
    for test_sid, fold in loso_splits.items():
        # (a)
        if test_sid in fold['train_subjects']:
            print(f"FAIL: {test_sid} in own training set!")
            all_ok = False

        # (b)
        tdim = fold['test']['features'].shape[1]
        pdim = fold['pca'].pca.n_components_
        if tdim != pdim:
            print(f"FAIL: {test_sid} dim mismatch {tdim} != {pdim}")
            all_ok = False
        if not fold['pca'].is_fitted:
            print(f"FAIL: {test_sid} PCA not fitted!")
            all_ok = False

        # (c)
        if np.any(~np.isfinite(fold['test']['features'])):
            print(f"FAIL: {test_sid} contains NaN/Inf in features!")
            all_ok = False

    if all_ok and verbose:
        print(f"OK  LOSO isolation verified: {len(loso_splits)} folds, all clean.")

    # Also check support/query disjointness
    first_sid = sorted(loso_splits.keys())[0]
    test_X = loso_splits[first_sid]['test']['features']
    test_y = loso_splits[first_sid]['test']['labels']
    for k in Config.K_SHOTS:
        sx, sy, qx, qy = MAML_Encoder._balanced_episode(test_X, test_y, k, 40)
        sx_set = {tuple(x.round(6)) for x in sx}
        qx_set = {tuple(x.round(6)) for x in qx}
        assert not (sx_set & qx_set), f"K={k}: support/query overlap!"
    if verbose:
        print(f"OK  Support/query disjointness verified for K={Config.K_SHOTS}")

    return all_ok
