"""Matching Networks with learnable temperature.

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
from .prototypical import create_balanced_episode

class MatchingNetwork:
    """Matching Networks with learnable softmax temperature (upgraded)."""
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
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        # Learnable temperature (log-space for numerical stability)
        self.log_tau = nn.Parameter(torch.zeros(1, device=self.device))

        self.opt   = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.attention.parameters()) + [self.log_tau],
            lr=5e-4, weight_decay=1e-5)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=500, eta_min=1e-5)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(min=0.1, max=10.0)

    def _attn_weights(self, q_emb: torch.Tensor, s_emb: torch.Tensor) -> torch.Tensor:
        Q, S = q_emb.size(0), s_emb.size(0)
        q_exp = q_emb.unsqueeze(1).expand(Q, S, -1)
        s_exp = s_emb.unsqueeze(0).expand(Q, S, -1)
        scores = self.attention(torch.cat([q_exp, s_exp], dim=2).view(-1, q_emb.size(1)*2))
        return F.softmax(scores.view(Q, S) / self.temperature, dim=1)

    def _predict(self, q_emb, s_emb, s_lbl):
        n_cls  = int(s_lbl.max().item()) + 1
        w      = self._attn_weights(q_emb, s_emb)
        s_oh   = F.one_hot(s_lbl, num_classes=n_cls).float()
        probs  = w @ s_oh
        return probs.argmax(1), probs

    def train_episode(self, sx, sy, qx, qy):
        self.opt.zero_grad()
        s_emb = self.encoder(torch.FloatTensor(sx).to(self.device))
        q_emb = self.encoder(torch.FloatTensor(qx).to(self.device))
        sy_t  = torch.LongTensor(sy).to(self.device)
        qy_t  = torch.LongTensor(qy).to(self.device)
        preds, probs = self._predict(q_emb, s_emb, sy_t)
        c = np.bincount(qy, minlength=2).astype(float)
        c = np.where(c == 0, 1.0, c)
        w = torch.FloatTensor(1.0/c * 2 / sum(1.0/c)).to(self.device)
        loss = F.cross_entropy(probs, qy_t, weight=w)
        loss.backward(); self.opt.step()
        return loss.item(), (preds == qy_t).float().mean().item()

    def evaluate_episode(self, sx, sy, qx, qy) -> dict:
        self.encoder.eval()
        with torch.no_grad():
            s_emb = self.encoder(torch.FloatTensor(sx).to(self.device))
            q_emb = self.encoder(torch.FloatTensor(qx).to(self.device))
            sy_t  = torch.LongTensor(sy).to(self.device)
            preds, probs = self._predict(q_emb, s_emb, sy_t)
        self.encoder.train()
        return compute_comprehensive_metrics(
            preds.cpu().numpy(), qy.astype(int), probs.cpu().numpy())

def train_matching_loso(
        subjects_features: dict, k_shots: list,
        num_epochs: int = 500, query_per_class: int = 20,
        device: str = str(Config.DEVICE), seed: int = 42,
        output_dir: str = Config.RESULTS_DIR) -> dict:
    """LOSO Matching Networks with learnable temperature."""
    set_seed(seed)
    all_results: dict = {}
    method_name = 'Matching'
    sids = sorted(subjects_features.keys())

    for test_sid in tqdm(sids, desc="Matching LOSO"):
        fold_seed = seed + hash(test_sid) % 10000 + 11
        set_seed(fold_seed)

        ckpt = load_subject_checkpoint(output_dir, method_name, seed, test_sid)
        if ckpt is not None:
            all_results[test_sid] = ckpt; continue

        train_sids = [s for s in sids if s != test_sid]
        input_dim  = subjects_features[train_sids[0]]['features'].shape[1]
        model = MatchingNetwork(input_dim, device=device)

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
