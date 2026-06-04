"""Prototypical Networks (defines create_balanced_episode, reused by others).

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
from .encoders import EEGEncoder, MetaEEGClassifier

def create_balanced_episode(features: np.ndarray, labels: np.ndarray,
                             support_k: int, query_per_class: int = 20,
                             augment: bool = False, aug_factor: int = 3
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a balanced binary few-shot episode.

    Args:
        features       : (N, D)
        labels         : (N,) with values 0 and 1
        support_k      : total support set size (split equally across classes)
        query_per_class: query examples per class
        augment        : if True and support_k < 10, add linear interpolation augmentation
        aug_factor     : synthetic examples per real example
    """
    unique = np.unique(labels)
    k_per  = max(1, support_k // len(unique))
    sxl, syl, qxl, qyl = [], [], [], []

    for cls in unique:
        idx    = np.where(labels == cls)[0]
        n_avail = len(idx)
        sp_n    = min(k_per, n_avail - 1); sp_n = max(1, sp_n)
        qp_n    = min(query_per_class, n_avail - sp_n); qp_n = max(1, qp_n)
        perm    = np.random.permutation(n_avail)
        sxl.append(features[idx[perm[:sp_n]]])
        syl.append(labels  [idx[perm[:sp_n]]])
        qxl.append(features[idx[perm[sp_n:sp_n+qp_n]]])
        qyl.append(labels  [idx[perm[sp_n:sp_n+qp_n]]])

    sx = np.concatenate(sxl); sy = np.concatenate(syl)
    qx = np.concatenate(qxl); qy = np.concatenate(qyl)

    if augment and support_k < 10:
        aug_sx, aug_sy = [], []
        for i in range(len(sx)):
            same = np.where(sy == sy[i])[0]
            for _ in range(aug_factor):
                j = np.random.choice(same)
                a = np.random.uniform(0.3, 0.7)
                aug_sx.append(a * sx[i] + (1-a) * sx[j])
                aug_sy.append(sy[i])
        if aug_sx:
            sx = np.vstack([sx, np.array(aug_sx)])
            sy = np.concatenate([sy, np.array(aug_sy, dtype=int)])

    p = np.random.permutation(len(sx)); sx = sx[p]; sy = sy[p]
    p = np.random.permutation(len(qx)); qx = qx[p]; qy = qy[p]
    return sx, sy, qx, qy

class PrototypicalNetwork:
    """Prototypical Networks for Few-Shot ErrP classification (upgraded)."""
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 embed_dim: int = 64, device: str = str(Config.DEVICE)):
        self.device = torch.device(device)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim)
        ).to(self.device)
        self.opt   = optim.Adam(self.encoder.parameters(), lr=5e-4, weight_decay=1e-5)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=500, eta_min=1e-5)

    def _prototypes(self, emb: torch.Tensor, lbl: torch.Tensor):
        classes = sorted(lbl.unique().tolist())
        protos  = torch.stack([emb[lbl == c].mean(0) for c in classes])
        return classes, protos

    def train_episode(self, sx, sy, qx, qy) -> Tuple[float, float]:
        self.opt.zero_grad()
        s_emb = self.encoder(torch.FloatTensor(sx).to(self.device))
        q_emb = self.encoder(torch.FloatTensor(qx).to(self.device))
        sy_t  = torch.LongTensor(sy).to(self.device)
        qy_t  = torch.LongTensor(qy).to(self.device)
        classes, protos = self._prototypes(s_emb, sy_t)
        logits = -torch.cdist(q_emb, protos)
        tgt    = torch.tensor([classes.index(c.item()) for c in qy_t], device=self.device)
        c      = np.bincount(qy, minlength=2).astype(float)
        c      = np.where(c == 0, 1.0, c)
        w      = torch.FloatTensor(1.0/c * 2 / sum(1.0/c)).to(self.device)
        loss   = F.cross_entropy(logits, tgt, weight=w)
        loss.backward(); self.opt.step()
        return loss.item(), (logits.argmax(1) == tgt).float().mean().item()

    def evaluate_episode(self, sx, sy, qx, qy) -> dict:
        self.encoder.eval()
        with torch.no_grad():
            s_emb = self.encoder(torch.FloatTensor(sx).to(self.device))
            q_emb = self.encoder(torch.FloatTensor(qx).to(self.device))
            sy_t  = torch.LongTensor(sy).to(self.device)
            classes, protos = self._prototypes(s_emb, sy_t)
            dists   = torch.cdist(q_emb, protos)
            pred_i  = dists.argmin(1)
            preds   = torch.tensor([classes[i] for i in pred_i.tolist()], device=self.device)
            probs   = F.softmax(-dists, dim=1).cpu().numpy()
        self.encoder.train()
        return compute_comprehensive_metrics(preds.cpu().numpy(), qy, probs)

def train_prototypical_loso(
        subjects_features: dict, k_shots: list,
        num_epochs: int = 500, query_per_class: int = 20,
        device: str = str(Config.DEVICE), seed: int = 42,
        output_dir: str = Config.RESULTS_DIR) -> dict:
    """LOSO ProtoNets with cross-subject episodes and K<10 augmentation."""
    set_seed(seed)
    all_results: dict = {}
    method_name = 'Prototypical'
    sids = sorted(subjects_features.keys())

    for test_sid in tqdm(sids, desc="Prototypical LOSO"):
        fold_seed = seed + hash(test_sid) % 10000
        set_seed(fold_seed)

        ckpt = load_subject_checkpoint(output_dir, method_name, seed, test_sid)
        if ckpt is not None:
            all_results[test_sid] = ckpt; continue

        train_sids = [s for s in sids if s != test_sid]
        input_dim  = subjects_features[train_sids[0]]['features'].shape[1]
        model = PrototypicalNetwork(input_dim, device=device)

        for epoch in tqdm(range(num_epochs), desc=f'{method_name} {test_sid} train', every_n=100, leave=False):
            for _ in range(len(train_sids) * 2):
                s = np.random.choice(train_sids)
                f, l = subjects_features[s]['features'], subjects_features[s]['labels']
                k_tr = int(np.random.choice(k_shots))
                try:
                    sx, sy, qx, qy = create_balanced_episode(
                        f, l, k_tr, query_per_class, augment=(k_tr < 10))
                    model.train_episode(sx, sy, qx, qy)
                except (ValueError, IndexError):
                    continue
            model.sched.step()

        test_f = subjects_features[test_sid]['features']
        test_l = subjects_features[test_sid]['labels']
        subject_results = {'subject_id': test_sid, 'k_shots': {}}
        for k in k_shots:
            eps = []
            for ep in range(Config.N_EVAL_EPISODES):
                set_seed(fold_seed + k * 100 + ep)
                try:
                    sx, sy, qx, qy = create_balanced_episode(
                        test_f, test_l, k, query_per_class, augment=(k < 10))
                    eps.append(model.evaluate_episode(sx, sy, qx, qy))
                except (ValueError, IndexError):
                    continue
            agg = {}
            for m in ['accuracy', 'balanced_accuracy', 'f1_score', 'auroc']:
                vals = [e[m] for e in eps if not (e[m] != e[m])]
                agg[m] = float(np.mean(vals)) if vals else float('nan')
            subject_results['k_shots'][k] = agg

        all_results[test_sid] = subject_results
        save_subject_checkpoint(output_dir, method_name, seed, test_sid, subject_results)
        del model
        if 'cuda' in str(device): torch.cuda.empty_cache()

    return {'method': method_name, 'subjects': all_results,
            'k_shots': k_shots, 'seed': seed}
