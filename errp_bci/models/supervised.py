"""From-scratch supervised few-shot MLP baseline (method_name param added).

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

def run_supervised_baseline_loso(
        loso_splits: Dict, k_shots: List[int],
        hidden_dim: int = 128, lr: float = 1e-3, n_epochs: int = 200,
        device: str = str(Config.DEVICE), seed: int = 42,
        output_dir: str = Config.RESULTS_DIR,
        method_name: str = 'Supervised') -> Dict:
    """
    Supervised few-shot baseline with LOSO evaluation.

    For each subject and each K:
      1. Sample K/n_classes examples per class (balanced support set).
      2. Train 3-layer MLP with LayerNorm + GELU for up to n_epochs.
      3. Early stopping (patience=20) on support set loss.
      4. Evaluate on remaining test subject examples.
    """
    set_seed(seed)
    all_results: Dict = {}

    for test_sid, fold in tqdm(loso_splits.items(), desc="Supervised LOSO"):
        test_X    = fold['test']['features']
        test_y    = fold['test']['labels']
        input_dim = test_X.shape[1]
        subject_results = {'subject_id': test_sid, 'k_shots': {}}

        for k in k_shots:
            fold_seed = seed + hash(test_sid) % 10000 + k
            set_seed(fold_seed)

            # ── Balanced K-shot support set ─────────────────────────────
            unique_cls = np.unique(test_y)
            k_per_cls  = max(1, k // len(unique_cls))
            supp_idx = []
            for cls in unique_cls:
                cls_idx = np.where(test_y == cls)[0]
                n = min(k_per_cls, len(cls_idx))
                supp_idx.extend(np.random.choice(cls_idx, size=n, replace=False))
            supp_idx  = np.array(supp_idx)
            query_idx = np.array([i for i in range(len(test_X)) if i not in supp_idx])
            assert len(set(supp_idx) & set(query_idx)) == 0

            train_X = test_X[supp_idx]; train_y = test_y[supp_idx]
            eval_X  = test_X[query_idx]; eval_y  = test_y[query_idx]
            if len(eval_y) == 0:
                continue

            # ── Class weights ─────────────────────────────────────────
            counts = np.bincount(train_y, minlength=2).astype(float)
            counts = np.where(counts == 0, 1.0, counts)
            w = torch.FloatTensor(2.0 / (counts * counts.sum())).to(device)

            # ── Model: 3-layer MLP with LayerNorm + GELU ──────────────
            model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(hidden_dim, 64),        nn.LayerNorm(64),         nn.GELU(), nn.Dropout(0.3),
                nn.Linear(64, 2)
            ).to(device)

            opt  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss(weight=w)
            tx   = torch.FloatTensor(train_X).to(device)
            ty   = torch.LongTensor(train_y).to(device)

            # ── Train with early stopping ──────────────────────────────
            best_loss, patience_ctr, best_state = float('inf'), 0, None
            model.train()
            for _ in range(n_epochs):
                opt.zero_grad()
                loss = crit(model(tx), ty)
                loss.backward(); opt.step()
                if loss.item() < best_loss - 1e-5:
                    best_loss   = loss.item()
                    best_state  = deepcopy(model.state_dict())
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                if patience_ctr >= 20:
                    break
            if best_state is not None:
                model.load_state_dict(best_state)

            # ── Evaluate ──────────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                lgts  = model(torch.FloatTensor(eval_X).to(device))
                probs = F.softmax(lgts, dim=1).cpu().numpy()
                preds = lgts.argmax(dim=1).cpu().numpy()
            subject_results['k_shots'][k] = compute_comprehensive_metrics(preds, eval_y, probs)

            del model
            if 'cuda' in str(device):
                torch.cuda.empty_cache()

        all_results[test_sid] = subject_results
        save_subject_checkpoint(output_dir, method_name, seed, test_sid, subject_results)

    return {'method': method_name, 'subjects': all_results, 'k_shots': k_shots, 'seed': seed}
