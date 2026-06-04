"""Reptile first-order meta-learner.

Auto-ported verbatim from the legacy notebooks by _build_pkg.py.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from ..progress import tqdm

from ..config import Config
from ..reproducibility import set_seed, freeze_batchnorm
from ..metrics import compute_comprehensive_metrics
from ..io_results import save_subject_checkpoint, load_subject_checkpoint
from .encoders import MetaEEGClassifier
from .maml import MAML_Encoder

class ReptileMetaLearner:
    """
    Reptile (Nichol et al. 2018) — corrected outer update.

    Outer update: θ ← θ + ε * (φ_mean - θ)
    where φ_mean = average of task-adapted parameters across meta-batch.

    NO optimizer in outer step. Direct weight-space interpolation only.
    Inner update: SGD with class-weighted CE.
    """
    def __init__(self, input_dim: int,
                 h1: int = Config.ENCODER_HIDDEN,
                 h2: int = Config.ENCODER_HIDDEN2,
                 enc_out: int = Config.ENCODER_OUTPUT,
                 num_classes: int = 2,
                 inner_lr: float = Config.INNER_LR,
                 outer_lr: float = 0.1,
                 inner_steps: int = Config.INNER_STEPS,
                 device: str = str(Config.DEVICE)):
        self.device      = torch.device(device)
        self.inner_lr    = inner_lr
        self.outer_lr    = outer_lr
        self.inner_steps = inner_steps
        self.model = MetaEEGClassifier(
            input_dim, h1, h2, enc_out, Config.DROPOUT, num_classes).to(self.device)

    @staticmethod
    def _cw(labels: np.ndarray, device: torch.device) -> torch.Tensor:
        c = np.bincount(labels, minlength=2).astype(float)
        c = np.where(c == 0, 1.0, c); w = 1.0/c; w = w/w.sum()*2
        return torch.FloatTensor(w).to(device)

    def _adapt(self, features: np.ndarray, labels: np.ndarray) -> nn.Module:
        """Run n_inner_steps of SGD on a task. Returns adapted model copy."""
        adapted = deepcopy(self.model)
        opt = optim.SGD(adapted.parameters(), lr=self.inner_lr)
        X  = torch.FloatTensor(features).to(self.device)
        y  = torch.LongTensor(labels).to(self.device)
        cw = self._cw(labels, self.device)
        adapted.train()
        for _ in range(self.inner_steps):
            opt.zero_grad()
            F.cross_entropy(adapted(X), y, weight=cw).backward()
            opt.step()
        return adapted

    def meta_update(self, tasks: list) -> None:
        """Correct Reptile outer update: interpolate toward mean of adapted params.

        IMPORTANT: No optimizer call here. This is direct parameter interpolation.
        """
        adapted_models = [self._adapt(sx, sy) for sx, sy in tasks]

        for meta_p, *task_ps in zip(self.model.parameters(),
                                    *[m.parameters() for m in adapted_models]):
            mean_adapted = torch.stack([p.data for p in task_ps]).mean(dim=0)
            # θ ← θ + ε * (φ_mean - θ)
            meta_p.data.add_(self.outer_lr * (mean_adapted - meta_p.data))

        del adapted_models

    def adapt_and_evaluate(self, support_X: np.ndarray, support_y: np.ndarray,
                            query_X: np.ndarray, query_y: np.ndarray) -> dict:
        adapted = self._adapt(support_X, support_y)
        adapted.eval()
        with torch.no_grad():
            lgts  = adapted(torch.FloatTensor(query_X).to(self.device))
            probs = F.softmax(lgts, 1).cpu().numpy()
            preds = lgts.argmax(1).cpu().numpy()
        del adapted
        return compute_comprehensive_metrics(preds, query_y, probs)

def train_reptile_loso(
        loso_splits: dict, k_shots: list,
        n_iterations: int = Config.N_META_ITERATIONS,
        inner_lr: float = Config.INNER_LR, outer_lr: float = 0.1,
        inner_steps: int = Config.INNER_STEPS, meta_batch_size: int = 4,
        n_support: int = Config.N_SUPPORT,
        device: str = str(Config.DEVICE), seed: int = 42,
        output_dir: str = Config.RESULTS_DIR) -> dict:
    """LOSO Reptile with corrected outer update and balanced episodes."""
    set_seed(seed)
    all_results: dict = {}
    method_name = 'Reptile'

    for fold_idx, (test_sid, fold) in enumerate(
            tqdm(loso_splits.items(), desc="Reptile LOSO")):
        fold_seed = seed + fold_idx * 7
        set_seed(fold_seed)

        ckpt = load_subject_checkpoint(output_dir, method_name, seed, test_sid)
        if ckpt is not None:
            all_results[test_sid] = ckpt; continue

        train_sids = fold['train_subjects']
        train_dict = {s: {'features': fold['pca'].transform(
                              fold['subjects_features'][s]['features']),
                          'labels': fold['subjects_features'][s]['labels']}
                      for s in train_sids}
        input_dim = fold['test']['features'].shape[1]
        reptile = ReptileMetaLearner(input_dim=input_dim, inner_lr=inner_lr,
                                     outer_lr=outer_lr, inner_steps=inner_steps, device=device)

        for it in tqdm(range(n_iterations), desc=f'{method_name} {test_sid} meta-train', every_n=250, leave=False):
            batch = np.random.choice(train_sids, size=min(meta_batch_size, len(train_sids)),
                                     replace=False)
            tasks = []
            for s in batch:
                f, l = train_dict[s]['features'], train_dict[s]['labels']
                sx, sy, _, _ = MAML_Encoder._balanced_episode(f, l, n_support, 0)
                tasks.append((sx, sy))
            if tasks:
                reptile.meta_update(tasks)

        test_X = fold['test']['features']; test_y = fold['test']['labels']
        subject_results = {'subject_id': test_sid, 'k_shots': {}}
        for k in k_shots:
            eps = []
            for ep in range(Config.N_EVAL_EPISODES):
                set_seed(fold_seed + k * 100 + ep)
                sx, sy, qx, qy = MAML_Encoder._balanced_episode(
                    test_X, test_y, k, min(len(test_y)-k, 200))
                if len(qy) == 0: continue
                eps.append(reptile.adapt_and_evaluate(sx, sy, qx, qy))
            agg = {}
            for m in ['accuracy', 'balanced_accuracy', 'f1_score', 'auroc']:
                vals = [e[m] for e in eps if not (e[m] != e[m])]
                agg[m] = float(np.mean(vals)) if vals else float('nan')
            subject_results['k_shots'][k] = agg

        all_results[test_sid] = subject_results
        save_subject_checkpoint(output_dir, method_name, seed, test_sid, subject_results)
        del reptile
        if 'cuda' in str(device): torch.cuda.empty_cache()

    return {'method': method_name, 'subjects': all_results,
            'k_shots': k_shots, 'seed': seed}
