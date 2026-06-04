"""Riemannian + covariance-alignment baselines with xDAWN spatial filtering.

Auto-ported verbatim from the legacy notebooks by _build_pkg.py.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..progress import tqdm
from scipy.linalg import sqrtm as mat_sqrt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..config import Config
from ..reproducibility import set_seed
from ..metrics import compute_comprehensive_metrics
from ..io_results import save_subject_checkpoint, load_subject_checkpoint

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.spatialfilters import Xdawn
    PYRIEMANN_AVAILABLE = True
except ImportError:
    PYRIEMANN_AVAILABLE = False

from .prototypical import create_balanced_episode

def _make_xdawn(n_filters: int):
    """Version-agnostic Xdawn constructor.

    Tries every known pyriemann API variation in order:
      1. Xdawn(nfilters=n, estimator='lwf')     [pyriemann < 0.3]
      2. Xdawn(n_components=n, estimator='lwf')  [pyriemann 0.3-0.5]
      3. Xdawn(n_filters=n, estimator='lwf')     [some builds]
      4. Xdawn(n, estimator='lwf')               [positional]
      5. Xdawn(n)                                [positional, no estimator]
    Disables the estimator kwarg variant if TypeError is raised.
    """
    import inspect
    try:
        params = list(inspect.signature(Xdawn.__init__).parameters.keys())
    except Exception:
        params = []

    attempts = []
    for kwarg in ['nfilters', 'n_components', 'n_filters', 'nfilter']:
        attempts.append({kwarg: n_filters, 'estimator': 'lwf'})
        attempts.append({kwarg: n_filters})

    for kwargs in attempts:
        try:
            return Xdawn(**kwargs)
        except TypeError:
            continue

    # Pure positional fallbacks
    for args, kwargs in [
        ((n_filters,), {'estimator': 'lwf'}),
        ((n_filters,), {}),
    ]:
        try:
            return Xdawn(*args, **kwargs)
        except TypeError:
            continue

    raise RuntimeError(
        f"Cannot construct Xdawn with n_filters={n_filters}. "
        f"Tried all known pyriemann API variants. "
        f"Xdawn.__init__ params: {params}"
    )

def run_riemannian_tangent_lda_loso(
        loso_splits: dict, k_shots: list,
        preprocessed_data: dict,
        seed: int = 42, output_dir: str = Config.RESULTS_DIR,
        n_xdawn_filters: int = 4) -> Optional[dict]:
    """xDAWN + Riemannian Tangent Space + LDA with LOSO isolation."""
    if not PYRIEMANN_AVAILABLE:
        print("pyriemann not available. Skipping.")
        return None

    set_seed(seed)
    all_results: dict = {}
    method_name = 'Riemannian'

    for test_sid, fold in tqdm(loso_splits.items(), desc="Riemannian LOSO"):
        ckpt = load_subject_checkpoint(output_dir, method_name, seed, test_sid)
        if ckpt is not None:
            all_results[test_sid] = ckpt; continue

        train_sids   = fold['train_subjects']
        train_ep     = np.concatenate([preprocessed_data[s]['epochs'] for s in train_sids])
        train_lb     = np.concatenate([preprocessed_data[s]['labels'] for s in train_sids])
        test_ep      = preprocessed_data[test_sid]['epochs']
        test_lb      = preprocessed_data[test_sid]['labels']

        # ── xDAWN (LOSO-isolated): fit on training subjects only ───────────
        try:
            xd = _make_xdawn(n_xdawn_filters)
            xd.fit(train_ep, train_lb)
            tr_ep_xd = xd.transform(train_ep)
            ts_ep_xd = xd.transform(test_ep)
        except Exception as e:
            print(f"  xDAWN failed for {test_sid}: {e}. Using raw.")
            tr_ep_xd = train_ep; ts_ep_xd = test_ep

        # ── Covariance matrices ────────────────────────────────────────────
        cov = Covariances(estimator='lwf')
        try:
            tr_cov = cov.transform(tr_ep_xd)
            ts_cov = cov.transform(ts_ep_xd)
        except Exception as e:
            print(f"  Covariance failed for {test_sid}: {e}. Skipping.")
            continue

        # ── Tangent space fitted on TRAINING covariances ───────────────────
        ts = TangentSpace(metric='riemann')
        try:
            ts.fit(tr_cov)
        except Exception as e:
            print(f"  TangentSpace fit failed for {test_sid}: {e}. Skipping.")
            continue

        subject_results = {'subject_id': test_sid, 'k_shots': {}}
        for k in k_shots:
            np.random.seed(seed + hash(test_sid) % 10000 + k)
            try:
                sx, sy, qx, qy = create_balanced_episode(ts_cov, test_lb, k, 40)
                sx_ts = ts.transform(sx); qx_ts = ts.transform(qx)
                lda   = LinearDiscriminantAnalysis()
                lda.fit(sx_ts, sy)
                preds = lda.predict(qx_ts)
                try:
                    probs = lda.predict_proba(qx_ts)
                except Exception:
                    probs = None
                subject_results['k_shots'][k] = compute_comprehensive_metrics(preds, qy, probs)
            except Exception:
                subject_results['k_shots'][k] = {m: float('nan')
                    for m in ['accuracy','balanced_accuracy','f1_score','auroc']}

        all_results[test_sid] = subject_results
        save_subject_checkpoint(output_dir, method_name, seed, test_sid, subject_results)

    return {'method': method_name, 'subjects': all_results,
            'k_shots': k_shots, 'seed': seed}

def run_covariance_alignment_loso(
        loso_splits: dict, k_shots: list,
        preprocessed_data: dict,
        seed: int = 42, output_dir: str = Config.RESULTS_DIR,
        n_xdawn_filters: int = 4) -> Optional[dict]:
    """xDAWN + Riemannian Parallel Transport Alignment + LDA.

    Alignment: C_aligned = M_tgt^{1/2} M_src^{-1/2} C M_src^{-1/2} M_tgt^{1/2}
    where M_src = Riemannian mean of training covariances,
          M_tgt = Riemannian mean of K-shot support covariances.
    """
    if not PYRIEMANN_AVAILABLE:
        return None

    set_seed(seed)
    all_results: dict = {}
    method_name = 'CovarianceAlignment'

    for test_sid, fold in tqdm(loso_splits.items(), desc="CovAlign LOSO"):
        ckpt = load_subject_checkpoint(output_dir, method_name, seed, test_sid)
        if ckpt is not None:
            all_results[test_sid] = ckpt; continue

        train_sids  = fold['train_subjects']
        train_ep    = np.concatenate([preprocessed_data[s]['epochs'] for s in train_sids])
        train_lb    = np.concatenate([preprocessed_data[s]['labels'] for s in train_sids])
        test_ep     = preprocessed_data[test_sid]['epochs']
        test_lb     = preprocessed_data[test_sid]['labels']

        try:
            xd = _make_xdawn(n_xdawn_filters)
            xd.fit(train_ep, train_lb)
            tr_ep_xd = xd.transform(train_ep)
            ts_ep_xd = xd.transform(test_ep)
        except Exception:
            tr_ep_xd = train_ep; ts_ep_xd = test_ep

        cov = Covariances(estimator='lwf')
        try:
            tr_cov = cov.transform(tr_ep_xd)
            ts_cov = cov.transform(ts_ep_xd)
        except Exception as e:
            print(f"  CovAlign Cov failed for {test_sid}: {e}. Skipping."); continue

        try:
            M_src          = mean_riemann(tr_cov)
            M_src_inv_sqrt = np.real(mat_sqrt(np.linalg.inv(M_src)))
        except Exception as e:
            print(f"  Riemannian mean failed for {test_sid}: {e}. Skipping."); continue

        subject_results = {'subject_id': test_sid, 'k_shots': {}}
        for k in k_shots:
            np.random.seed(seed + hash(test_sid) % 10000 + k + 100)
            try:
                sx_cov, sy, qx_cov, qy = create_balanced_episode(ts_cov, test_lb, k, 40)
                M_tgt      = mean_riemann(sx_cov)
                M_tgt_sqrt = np.real(mat_sqrt(M_tgt))

                aligned = np.array([
                    M_tgt_sqrt @ M_src_inv_sqrt @ C @ M_src_inv_sqrt @ M_tgt_sqrt
                    for C in tr_cov])

                ts = TangentSpace(metric='riemann')
                ts.fit(aligned)
                sx_ts = ts.transform(sx_cov)
                qx_ts = ts.transform(qx_cov)

                lda = LinearDiscriminantAnalysis()
                lda.fit(sx_ts, sy)
                preds = lda.predict(qx_ts)
                try:
                    probs = lda.predict_proba(qx_ts)
                except Exception:
                    probs = None
                subject_results['k_shots'][k] = compute_comprehensive_metrics(preds, qy, probs)
            except Exception:
                subject_results['k_shots'][k] = {m: float('nan')
                    for m in ['accuracy','balanced_accuracy','f1_score','auroc']}

        all_results[test_sid] = subject_results
        save_subject_checkpoint(output_dir, method_name, seed, test_sid, subject_results)

    return {'method': method_name, 'subjects': all_results,
            'k_shots': k_shots, 'seed': seed}
