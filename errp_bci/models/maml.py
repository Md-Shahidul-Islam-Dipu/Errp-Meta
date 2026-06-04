"""MAML / ANIL (with use_weighting ablation flag; default True = primary).

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

class MAML_Encoder:
    """
    MAML / ANIL for EEG-based BCI personalization.

    Outer: Adam + CosineAnnealingLR + gradient clipping (max_norm=1.0)
    Inner: functional gradient updates (correct gradient flow)
    Loss : class-weighted cross-entropy (handles 3:1 ErrP imbalance)
    Episodes: balanced (K/2 per class support and query)
    """
    def __init__(self, input_dim: int,
                 h1: int = Config.ENCODER_HIDDEN,
                 h2: int = Config.ENCODER_HIDDEN2,
                 enc_out: int = Config.ENCODER_OUTPUT,
                 num_classes: int = 2,
                 inner_lr: float = Config.INNER_LR,
                 outer_lr: float = Config.OUTER_LR,
                 inner_steps: int = Config.INNER_STEPS,
                 n_meta_iterations: int = Config.N_META_ITERATIONS,
                 freeze_encoder_inner: bool = False,
                 first_order: bool = True,
                 use_weighting: bool = True,          # ← ABLATION 4 flag
                 device: str = str(Config.DEVICE)):
        self.inner_lr    = inner_lr
        self.inner_steps = inner_steps
        self.freeze_enc  = freeze_encoder_inner
        self.first_order = first_order
        self.use_weighting = use_weighting             # ← ABLATION 4 flag
        self.device      = torch.device(device)

        self.meta_model = MetaEEGClassifier(
            input_dim, h1, h2, enc_out, Config.DROPOUT, num_classes).to(self.device)
        self.meta_opt   = optim.Adam(self.meta_model.parameters(), lr=outer_lr)
        self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(
            self.meta_opt, T_max=n_meta_iterations, eta_min=outer_lr * 0.01)

    # ── Utility: balanced episode sampling ────────────────────────────────
    @staticmethod
    def _balanced_episode(features: np.ndarray, labels: np.ndarray,
                           n_support: int, n_query: int
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a balanced episode with n_support/2 + n_query/2 per class."""
        unique = np.unique(labels)
        sp_per = max(1, n_support // len(unique))
        qp_per = max(1, n_query   // len(unique)) if n_query > 0 else 0
        sxl, syl, qxl, qyl = [], [], [], []
        for cls in unique:
            idx = np.where(labels == cls)[0]
            n   = sp_per + qp_per
            if len(idx) < n:
                sp = max(1, len(idx) // 2); qp = len(idx) - sp
            else:
                sp, qp = sp_per, qp_per
            perm = np.random.permutation(idx)
            sxl.append(features[perm[:sp]]);    syl.append(labels[perm[:sp]])
            qxl.append(features[perm[sp:sp+qp]]); qyl.append(labels[perm[sp:sp+qp]])
        return (np.concatenate(sxl), np.concatenate(syl),
                np.concatenate(qxl), np.concatenate(qyl))

    # ── Utility: class weights ────────────────────────────────────────────
    @staticmethod
    def _class_weights(labels: np.ndarray, device: torch.device,
                       n_classes: int = 2) -> torch.Tensor:
        counts = np.bincount(labels, minlength=n_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        w = 1.0 / counts; w = w / w.sum() * n_classes
        return torch.FloatTensor(w).to(device)

    # ── ABLATION 4: weight selector ───────────────────────────────────────
    def _cw(self, labels: np.ndarray) -> torch.Tensor:
        """Return class weights or uniform weights depending on use_weighting flag.

        use_weighting=True  (default / primary):  inverse-frequency weights
        use_weighting=False (ablation):            uniform weights [0.5, 0.5]

        Called in meta_update (support + query) and adapt_and_evaluate (support).
        Replaces all direct calls to _class_weights throughout the class.
        """
        if self.use_weighting:
            return self._class_weights(labels, self.device)
        else:
            return torch.FloatTensor([1.0, 1.0]).to(self.device)

    # ── Functional forward pass ──────────────────────────────────────────
    def _functional_forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Forward pass using parameter dict (enables functional gradient flow)."""
        h = F.dropout(F.gelu(F.layer_norm(
                F.linear(x, params['encoder.fc1.weight'], params['encoder.fc1.bias']),
                [params['encoder.fc1.weight'].shape[0]])),
                p=Config.DROPOUT, training=self.meta_model.training)
        h = F.dropout(F.gelu(F.layer_norm(
                F.linear(h, params['encoder.fc2.weight'], params['encoder.fc2.bias']),
                [params['encoder.fc2.weight'].shape[0]])),
                p=Config.DROPOUT, training=self.meta_model.training)
        h = F.linear(h, params['encoder.fc3.weight'], params['encoder.fc3.bias'])
        return F.linear(h, params['task_head.fc.weight'], params['task_head.fc.bias'])

    # ── Functional inner loop ─────────────────────────────────────────────
    def _inner_loop(self, sx: torch.Tensor, sy: torch.Tensor,
                    params: dict, cw: torch.Tensor) -> Tuple[dict, float]:
        """Inner-loop gradient adaptation using functional gradients.

        Uses SHALLOW dict copy — tensors are shared, not cloned.
        This preserves the computational graph from query loss back to meta-params.
        """
        adapted   = dict(params)   # shallow copy — DO NOT use deepcopy here
        to_update = ({k: v for k, v in adapted.items() if 'task_head' in k}
                     if self.freeze_enc else adapted)
        total_loss = 0.0
        for _ in range(self.inner_steps):
            logits = self._functional_forward(sx, adapted)
            loss   = F.cross_entropy(logits, sy, weight=cw)
            grads  = torch.autograd.grad(
                loss, to_update.values(),
                create_graph=not self.first_order, allow_unused=True)
            for (name, _), g in zip(list(to_update.items()), grads):
                if g is not None:
                    g_ = g.detach() if self.first_order else g
                    adapted[name] = adapted[name] - self.inner_lr * g_
            to_update = {k: adapted[k] for k in to_update}
            total_loss += loss.item()
        return adapted, total_loss / self.inner_steps

    def meta_update(self, tasks: list) -> dict:
        """Outer-loop meta-update across a batch of tasks."""
        self.meta_opt.zero_grad()
        params = {n: p for n, p in self.meta_model.named_parameters()}
        meta_losses, q_accs = [], []

        for sx, sy, qx, qy in tasks:
            sx = torch.FloatTensor(sx).to(self.device)
            sy = torch.LongTensor(sy).to(self.device)
            qx = torch.FloatTensor(qx).to(self.device)
            qy = torch.LongTensor(qy).to(self.device)
            cw   = self._cw(sy.cpu().numpy())              # ← was _class_weights

            adapted, _ = self._inner_loop(sx, sy, params, cw)
            q_logits   = self._functional_forward(qx, adapted)
            q_cw       = self._cw(qy.cpu().numpy())        # ← was _class_weights
            q_loss     = F.cross_entropy(q_logits, qy, weight=q_cw)
            meta_losses.append(q_loss)
            with torch.no_grad():
                q_accs.append((q_logits.argmax(1) == qy).float().mean().item())

        meta_loss = torch.stack(meta_losses).mean()
        meta_loss.backward()
        nn.utils.clip_grad_norm_(self.meta_model.parameters(), max_norm=1.0)
        self.meta_opt.step()
        self.scheduler.step()
        return {'meta_loss': meta_loss.item(), 'query_acc': float(np.mean(q_accs))}

    def adapt_and_evaluate(self, support_X: np.ndarray, support_y: np.ndarray,
                            query_X: np.ndarray, query_y: np.ndarray) -> dict:
        """Test-time adaptation and evaluation."""
        adapted_model = deepcopy(self.meta_model)
        freeze_batchnorm(adapted_model)

        if self.freeze_enc:
            adapt_params = list(adapted_model.task_head.parameters())
            for p in adapted_model.encoder.parameters():
                p.requires_grad_(False)
        else:
            adapt_params = list(adapted_model.parameters())

        sx  = torch.FloatTensor(support_X).to(self.device)
        sy  = torch.LongTensor(support_y).to(self.device)
        cw  = self._cw(support_y)                          # ← was _class_weights
        opt = optim.SGD(adapt_params, lr=self.inner_lr)

        adapted_model.train()
        for _ in range(self.inner_steps):
            opt.zero_grad()
            F.cross_entropy(adapted_model(sx), sy, weight=cw).backward()
            opt.step()

        adapted_model.eval()
        with torch.no_grad():
            lgts  = adapted_model(torch.FloatTensor(query_X).to(self.device))
            probs = F.softmax(lgts, 1).cpu().numpy()
            preds = lgts.argmax(1).cpu().numpy()
        del adapted_model
        return compute_comprehensive_metrics(preds, query_y, probs)

def train_maml_loso(
        loso_splits: dict, k_shots: list,
        freeze_encoder_inner: bool = False,
        n_meta_iterations: int = Config.N_META_ITERATIONS,
        meta_batch_size: int = Config.META_BATCH_SIZE,
        n_support: int = Config.N_SUPPORT, n_query: int = Config.N_QUERY,
        inner_lr: float = Config.INNER_LR, outer_lr: float = Config.OUTER_LR,
        inner_steps: int = Config.INNER_STEPS,
        use_weighting: bool = True,                        # ← ABLATION 4 flag
        device: str = str(Config.DEVICE), seed: int = 42,
        output_dir: str = Config.RESULTS_DIR,
        method_name: str = 'MAML') -> dict:
    """LOSO MAML/ANIL training with checkpointing and zero-shot evaluation."""
    set_seed(seed)
    all_results: dict = {}

    for fold_idx, (test_sid, fold) in enumerate(
            tqdm(loso_splits.items(), desc=f"{method_name} LOSO")):
        fold_seed = seed + fold_idx * 13
        set_seed(fold_seed)

        ckpt = load_subject_checkpoint(output_dir, method_name, seed, test_sid)
        if ckpt is not None:
            all_results[test_sid] = ckpt
            continue

        train_sids = fold['train_subjects']
        train_dict = {s: {'features': fold['pca'].transform(
                              fold['subjects_features'][s]['features']),
                          'labels'  : fold['subjects_features'][s]['labels']}
                      for s in train_sids}
        input_dim = fold['test']['features'].shape[1]

        agent = MAML_Encoder(input_dim=input_dim, freeze_encoder_inner=freeze_encoder_inner,
                             inner_lr=inner_lr, outer_lr=outer_lr, inner_steps=inner_steps,
                             n_meta_iterations=n_meta_iterations,
                             first_order=freeze_encoder_inner,
                             use_weighting=use_weighting,  # ← ABLATION 4 flag
                             device=device)

        # ── Meta-training ─────────────────────────────────────────────────
        for it in tqdm(range(n_meta_iterations), desc=f'{method_name} {test_sid} meta-train', every_n=250, leave=False):
            batch = np.random.choice(train_sids,
                                     size=min(meta_batch_size, len(train_sids)),
                                     replace=False)
            tasks = []
            for s in batch:
                f, l = train_dict[s]['features'], train_dict[s]['labels']
                if len(f) < n_support + n_query:
                    continue
                sx, sy, qx, qy = MAML_Encoder._balanced_episode(f, l, n_support, n_query)
                tasks.append((sx, sy, qx, qy))
            if tasks:
                agent.meta_update(tasks)

        # ── Zero-shot baseline ─────────────────────────────────────────────
        test_X = fold['test']['features']
        test_y = fold['test']['labels']
        agent.meta_model.eval()
        with torch.no_grad():
            lgts = agent.meta_model(torch.FloatTensor(test_X).to(agent.device))
            zs_probs = F.softmax(lgts, 1).cpu().numpy()
            zs_preds = lgts.argmax(1).cpu().numpy()
        zero_shot = compute_comprehensive_metrics(zs_preds, test_y, zs_probs)

        # ── K-shot evaluation ──────────────────────────────────────────────
        subject_results = {'subject_id': test_sid,
                           'zero_shot': zero_shot, 'k_shots': {}}
        for k in k_shots:
            eps = []
            for ep in range(Config.N_EVAL_EPISODES):
                set_seed(fold_seed + k * 100 + ep)
                sx, sy, qx, qy = MAML_Encoder._balanced_episode(
                    test_X, test_y, k, min(len(test_y) - k, 200))
                if len(qy) == 0:
                    continue
                eps.append(agent.adapt_and_evaluate(sx, sy, qx, qy))
            agg = {}
            for m in ['accuracy', 'balanced_accuracy', 'f1_score', 'auroc']:
                vals = [e[m] for e in eps if not (e[m] != e[m])]
                agg[m] = float(np.mean(vals)) if vals else float('nan')
            subject_results['k_shots'][k] = agg

        all_results[test_sid] = subject_results
        save_subject_checkpoint(output_dir, method_name, seed, test_sid, subject_results)
        del agent
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

    return {'method': method_name, 'subjects': all_results,
            'k_shots': k_shots, 'seed': seed}
