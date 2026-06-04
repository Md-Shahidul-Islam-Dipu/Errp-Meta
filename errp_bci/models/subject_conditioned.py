"""Subject-conditioned meta-learner (+ NoFiLM path). train fn carries method_name.

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
from .encoders import EEGEncoder, SimpleTaskHead, MetaEEGClassifier
from .conditioning import (LabelAwareSubjectEncoder, FiLMLayer,
                           ConditionedEEGEncoder, TaskHead, UnconditionedEEGEncoder)
from .maml import MAML_Encoder

class SubjectConditionedMetaLearner:
    """
    Subject-Conditioned Meta-Learner with label-aware FiLM conditioning.

    SubjectEncoder: LabelAwareSubjectEncoder → z_s (32-dim)
    CondEncoder   : ConditionedEEGEncoder(x, z_s)
    TaskHead      : adapted per-subject in inner loop (ANIL-style)

    Outer loop optimises SubjectEncoder + CondEncoder jointly.
    Inner loop adapts only TaskHead (fast, stable at small K).
    """
    def __init__(self, input_dim: int,
                 embed_dim: int = Config.SUBJECT_EMBED_DIM,
                 enc_hidden: int = Config.ENCODER_HIDDEN2,
                 enc_out: int = Config.ENCODER_OUTPUT,
                 num_classes: int = 2,
                 inner_lr: float = Config.INNER_LR,
                 outer_lr: float = Config.OUTER_LR,
                 inner_steps: int = Config.INNER_STEPS,
                 n_meta_iterations: int = Config.N_META_ITERATIONS,
                 device: str = str(Config.DEVICE)):
        self.device      = torch.device(device)
        self.inner_lr    = inner_lr
        self.inner_steps = inner_steps

        self.subject_encoder = LabelAwareSubjectEncoder(
            input_dim, enc_hidden, embed_dim).to(self.device)
        self.cond_encoder = ConditionedEEGEncoder(
            input_dim, enc_hidden, enc_out, embed_dim).to(self.device)
        self.task_head = TaskHead(enc_out, num_classes).to(self.device)

        meta_params = (list(self.subject_encoder.parameters()) +
                       list(self.cond_encoder.parameters()))
        self.meta_opt  = optim.Adam(meta_params, lr=outer_lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.meta_opt, T_max=n_meta_iterations, eta_min=outer_lr * 0.01)

    @staticmethod
    def _cw(labels: np.ndarray, device: torch.device) -> torch.Tensor:
        c = np.bincount(labels, minlength=2).astype(float)
        c = np.where(c == 0, 1.0, c); w = 1.0/c; w = w/w.sum()*2
        return torch.FloatTensor(w).to(device)

    def _inner_adapt(self, sx: torch.Tensor, sy: torch.Tensor,
                     z_s: torch.Tensor, cw: torch.Tensor) -> dict:
        """Functional inner loop: adapt TaskHead parameters only."""
        tp = {n: p for n, p in self.task_head.named_parameters()}
        for _ in range(self.inner_steps):
            h = self.cond_encoder(sx, z_s)
            logits = F.linear(h, tp['fc.weight'], tp['fc.bias'])
            loss   = F.cross_entropy(logits, sy, weight=cw)
            grads  = torch.autograd.grad(loss, tp.values(),
                                         create_graph=False, allow_unused=True)
            tp = {n: p - self.inner_lr * (g.detach() if g is not None else torch.zeros_like(p))
                  for (n, p), g in zip(tp.items(), grads)}
        return tp

    def meta_update(self, tasks: list) -> dict:
        self.meta_opt.zero_grad()
        meta_losses, q_accs = [], []
        for sx, sy, qx, qy in tasks:
            sx = torch.FloatTensor(sx).to(self.device)
            sy = torch.LongTensor(sy).to(self.device)
            qx = torch.FloatTensor(qx).to(self.device)
            qy = torch.LongTensor(qy).to(self.device)
            cw = self._cw(sy.cpu().numpy(), self.device)

            z_s = self.subject_encoder(sx, sy)
            tp  = self._inner_adapt(sx, sy, z_s, cw)

            q_h     = self.cond_encoder(qx, z_s)
            q_lgts  = F.linear(q_h, tp['fc.weight'], tp['fc.bias'])
            q_cw    = self._cw(qy.cpu().numpy(), self.device)
            q_loss  = F.cross_entropy(q_lgts, qy, weight=q_cw)
            meta_losses.append(q_loss)
            with torch.no_grad():
                q_accs.append((q_lgts.argmax(1) == qy).float().mean().item())

        meta_loss = torch.stack(meta_losses).mean()
        meta_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.subject_encoder.parameters()) + list(self.cond_encoder.parameters()),
            max_norm=1.0)
        self.meta_opt.step(); self.scheduler.step()
        return {'meta_loss': meta_loss.item(), 'query_acc': float(np.mean(q_accs))}

    def adapt_and_evaluate(self, support_X: np.ndarray, support_y: np.ndarray,
                            query_X: np.ndarray, query_y: np.ndarray) -> dict:
        sx  = torch.FloatTensor(support_X).to(self.device)
        sy  = torch.LongTensor(support_y).to(self.device)
        qx  = torch.FloatTensor(query_X).to(self.device)
        cw  = self._cw(support_y, self.device)

        self.subject_encoder.eval(); self.cond_encoder.eval()
        with torch.no_grad():
            z_s = self.subject_encoder(sx, sy)
        tp = self._inner_adapt(sx, sy, z_s, cw)

        with torch.no_grad():
            q_h    = self.cond_encoder(qx, z_s)
            lgts   = F.linear(q_h, tp['fc.weight'], tp['fc.bias'])
            probs  = F.softmax(lgts, 1).cpu().numpy()
            preds  = lgts.argmax(1).cpu().numpy()
        self.subject_encoder.train(); self.cond_encoder.train()
        return compute_comprehensive_metrics(preds, query_y, probs)

N_RAW_PCA_COMPONENTS = 128

def train_subject_conditioned_loso(
        loso_splits: dict, k_shots: list,
        n_meta_iterations: int = Config.N_META_ITERATIONS,
        meta_batch_size: int = Config.META_BATCH_SIZE,
        n_support: int = Config.N_SUPPORT, n_query: int = Config.N_QUERY,
        device: str = str(Config.DEVICE), seed: int = 42,
        output_dir: str = Config.RESULTS_DIR,
        method_name: str = 'SubjectConditioned') -> dict:   # ← added param
    """LOSO training for Subject-Conditioned meta-learner."""
    set_seed(seed)
    all_results: dict = {}

    for fold_idx, (test_sid, fold) in enumerate(
            tqdm(loso_splits.items(), desc=f"{method_name} LOSO")):
        fold_seed = seed + fold_idx * 997
        set_seed(fold_seed)

        ckpt = load_subject_checkpoint(output_dir, method_name, seed, test_sid)
        if ckpt is not None:
            all_results[test_sid] = ckpt; continue

        train_sids = fold['train_subjects']
        train_data = [{'features': fold['pca'].transform(
                           fold['subjects_features'][s]['features']),
                       'labels': fold['subjects_features'][s]['labels']}
                      for s in train_sids]
        input_dim = fold['test']['features'].shape[1]

        learner = SubjectConditionedMetaLearner(
            input_dim=input_dim, device=device,
            n_meta_iterations=n_meta_iterations)

        for it in tqdm(range(n_meta_iterations), desc=f'{method_name} {test_sid} meta-train', every_n=250, leave=False):
            batch_idx = np.random.choice(len(train_data),
                                         size=min(meta_batch_size, len(train_data)),
                                         replace=False)
            tasks = []
            for bi in batch_idx:
                f, l = train_data[bi]['features'], train_data[bi]['labels']
                if len(f) < n_support + n_query:
                    continue
                tasks.append(MAML_Encoder._balanced_episode(f, l, n_support, n_query))
            if tasks:
                learner.meta_update(tasks)

        test_X = fold['test']['features']; test_y = fold['test']['labels']
        subject_results = {'subject_id': test_sid, 'k_shots': {}}
        for k in k_shots:
            eps = []
            for ep in range(Config.N_EVAL_EPISODES):
                set_seed(fold_seed + k * 100 + ep)
                sx, sy, qx, qy = MAML_Encoder._balanced_episode(
                    test_X, test_y, k, min(len(test_y)-k, 200))
                if len(qy) == 0: continue
                eps.append(learner.adapt_and_evaluate(sx, sy, qx, qy))
            agg = {}
            for m in ['accuracy', 'balanced_accuracy', 'f1_score', 'auroc']:
                vals = [e[m] for e in eps if not (e[m] != e[m])]
                agg[m] = float(np.mean(vals)) if vals else float('nan')
            subject_results['k_shots'][k] = agg

        all_results[test_sid] = subject_results
        save_subject_checkpoint(output_dir, method_name, seed, test_sid, subject_results)
        del learner
        if 'cuda' in str(device): torch.cuda.empty_cache()

    return {'method': method_name, 'subjects': all_results,
            'k_shots': k_shots, 'seed': seed}

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

class SubjectConditionedNoFiLM:
    """
    Ablation 1: SubjectConditionedMetaLearner with FiLM conditioning removed.

    Derived directly from SubjectConditionedMetaLearner (cell 29).

    What is removed:
        LabelAwareSubjectEncoder  →  removed entirely (no z_s computed)
        ConditionedEEGEncoder     →  replaced by UnconditionedEEGEncoder
        FiLMLayer x2              →  replaced by plain LayerNorm x2

    What is identical (copied from SubjectConditionedMetaLearner):
        TaskHead architecture
        _cw() — unchanged
        _inner_adapt() — one change: cond_encoder(sx, z_s) → cond_encoder(sx)
        meta_update() — one change: no subject_encoder call, no z_s
        adapt_and_evaluate() — one change: no subject_encoder call, no z_s
        meta_opt: Adam on cond_encoder params only
        scheduler: CosineAnnealingLR, same T_max and eta_min
        grad clip: max_norm=1.0

    Scientific question answered:
        SubjectConditioned > NoFiLM  →  FiLM subject-conditioning is the mechanism
        SubjectConditioned ≈ NoFiLM  →  architecture depth/width explains the gain
        NoFiLM > MAML               →  ANIL-style head adaptation helps regardless
    """
    def __init__(self, input_dim: int,
                 embed_dim: int = Config.SUBJECT_EMBED_DIM,
                 enc_hidden: int = Config.ENCODER_HIDDEN2,
                 enc_out: int = Config.ENCODER_OUTPUT,
                 num_classes: int = 2,
                 inner_lr: float = Config.INNER_LR,
                 outer_lr: float = Config.OUTER_LR,
                 inner_steps: int = Config.INNER_STEPS,
                 n_meta_iterations: int = Config.N_META_ITERATIONS,
                 device: str = str(Config.DEVICE)):
        self.device      = torch.device(device)
        self.inner_lr    = inner_lr
        self.inner_steps = inner_steps

        # Ablation: UnconditionedEEGEncoder replaces ConditionedEEGEncoder
        # No subject_encoder — z_s is never computed
        self.cond_encoder = UnconditionedEEGEncoder(
            input_dim, enc_hidden, enc_out, embed_dim).to(self.device)
        self.task_head = TaskHead(enc_out, num_classes).to(self.device)

        # Outer loop: cond_encoder only — identical to SubjectConditioned
        meta_params = list(self.cond_encoder.parameters())
        self.meta_opt  = optim.Adam(meta_params, lr=outer_lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.meta_opt, T_max=n_meta_iterations, eta_min=outer_lr * 0.01)

    @staticmethod
    def _cw(labels: np.ndarray, device: torch.device) -> torch.Tensor:
        """Identical to SubjectConditionedMetaLearner._cw."""
        c = np.bincount(labels, minlength=2).astype(float)
        c = np.where(c == 0, 1.0, c); w = 1.0/c; w = w/w.sum()*2
        return torch.FloatTensor(w).to(device)

    def _inner_adapt(self, sx: torch.Tensor, sy: torch.Tensor,
                     cw: torch.Tensor) -> dict:
        """
        Identical to SubjectConditionedMetaLearner._inner_adapt
        except: cond_encoder(sx, z_s) → cond_encoder(sx)
        """
        tp = {n: p for n, p in self.task_head.named_parameters()}
        for _ in range(self.inner_steps):
            h      = self.cond_encoder(sx)           # ← no z_s
            logits = F.linear(h, tp['fc.weight'], tp['fc.bias'])
            loss   = F.cross_entropy(logits, sy, weight=cw)
            grads  = torch.autograd.grad(loss, tp.values(),
                                         create_graph=False, allow_unused=True)
            tp = {n: p - self.inner_lr * (g.detach() if g is not None else torch.zeros_like(p))
                  for (n, p), g in zip(tp.items(), grads)}
        return tp

    def meta_update(self, tasks: list) -> dict:
        """
        Identical to SubjectConditionedMetaLearner.meta_update
        except: no subject_encoder forward, no z_s threading.
        """
        self.meta_opt.zero_grad()
        meta_losses, q_accs = [], []
        for sx, sy, qx, qy in tasks:
            sx = torch.FloatTensor(sx).to(self.device)
            sy = torch.LongTensor(sy).to(self.device)
            qx = torch.FloatTensor(qx).to(self.device)
            qy = torch.LongTensor(qy).to(self.device)
            cw = self._cw(sy.cpu().numpy(), self.device)

            tp     = self._inner_adapt(sx, sy, cw)
            q_h    = self.cond_encoder(qx)           # ← no z_s
            q_lgts = F.linear(q_h, tp['fc.weight'], tp['fc.bias'])
            q_cw   = self._cw(qy.cpu().numpy(), self.device)
            q_loss = F.cross_entropy(q_lgts, qy, weight=q_cw)
            meta_losses.append(q_loss)
            with torch.no_grad():
                q_accs.append((q_lgts.argmax(1) == qy).float().mean().item())

        meta_loss = torch.stack(meta_losses).mean()
        meta_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.cond_encoder.parameters()), max_norm=1.0)
        self.meta_opt.step(); self.scheduler.step()
        return {'meta_loss': meta_loss.item(), 'query_acc': float(np.mean(q_accs))}

    def adapt_and_evaluate(self, support_X: np.ndarray, support_y: np.ndarray,
                            query_X: np.ndarray, query_y: np.ndarray) -> dict:
        """
        Identical to SubjectConditionedMetaLearner.adapt_and_evaluate
        except: no subject_encoder forward, no z_s.
        """
        sx  = torch.FloatTensor(support_X).to(self.device)
        sy  = torch.LongTensor(support_y).to(self.device)
        qx  = torch.FloatTensor(query_X).to(self.device)
        cw  = self._cw(support_y, self.device)

        self.cond_encoder.eval()
        tp = self._inner_adapt(sx, sy, cw)

        with torch.no_grad():
            q_h   = self.cond_encoder(qx)            # ← no z_s
            lgts  = F.linear(q_h, tp['fc.weight'], tp['fc.bias'])
            probs = F.softmax(lgts, 1).cpu().numpy()
            preds = lgts.argmax(1).cpu().numpy()
        self.cond_encoder.train()
        return compute_comprehensive_metrics(preds, query_y, probs)

def train_no_film_loso(
        loso_splits: dict, k_shots: list,
        n_meta_iterations: int = Config.N_META_ITERATIONS,
        meta_batch_size: int = Config.META_BATCH_SIZE,
        n_support: int = Config.N_SUPPORT, n_query: int = Config.N_QUERY,
        device: str = str(Config.DEVICE), seed: int = 42,
        output_dir: str = Config.RESULTS_DIR) -> dict:
    """
    LOSO training for SubjectConditionedNoFiLM ablation.

    Copy of train_subject_conditioned_loso (cell 29) with exactly two changes:
      1. method_name = 'SubjectConditioned_NoFiLM'
      2. Instantiates SubjectConditionedNoFiLM instead of SubjectConditionedMetaLearner

    Everything else is identical: fold loop, episode sampling, K-shot
    evaluation, checkpointing, seed management.
    """
    set_seed(seed)
    all_results: dict = {}
    method_name = 'SubjectConditioned_NoFiLM'        # ← change 1

    for fold_idx, (test_sid, fold) in enumerate(
            tqdm(loso_splits.items(), desc="SubjectConditioned_NoFiLM LOSO")):
        fold_seed = seed + fold_idx * 997
        set_seed(fold_seed)

        ckpt = load_subject_checkpoint(output_dir, method_name, seed, test_sid)
        if ckpt is not None:
            all_results[test_sid] = ckpt; continue

        train_sids = fold['train_subjects']
        train_data = [{'features': fold['pca'].transform(
                           fold['subjects_features'][s]['features']),
                       'labels': fold['subjects_features'][s]['labels']}
                      for s in train_sids]
        input_dim = fold['test']['features'].shape[1]

        learner = SubjectConditionedNoFiLM(           # ← change 2
            input_dim=input_dim, device=device,
            n_meta_iterations=n_meta_iterations)

        for it in tqdm(range(n_meta_iterations), desc=f'{method_name} {test_sid} meta-train', every_n=250, leave=False):
            batch_idx = np.random.choice(len(train_data),
                                         size=min(meta_batch_size, len(train_data)),
                                         replace=False)
            tasks = []
            for bi in batch_idx:
                f, l = train_data[bi]['features'], train_data[bi]['labels']
                if len(f) < n_support + n_query:
                    continue
                tasks.append(_balanced_episode(f, l, n_support, n_query))
            if tasks:
                learner.meta_update(tasks)

        test_X = fold['test']['features']; test_y = fold['test']['labels']
        subject_results = {'subject_id': test_sid, 'k_shots': {}}
        for k in k_shots:
            eps = []
            for ep in range(Config.N_EVAL_EPISODES):
                set_seed(fold_seed + k * 100 + ep)
                sx, sy, qx, qy = _balanced_episode(
                    test_X, test_y, k, min(len(test_y)-k, 200))
                if len(qy) == 0: continue
                eps.append(learner.adapt_and_evaluate(sx, sy, qx, qy))
            agg = {}
            for m in ['accuracy', 'balanced_accuracy', 'f1_score', 'auroc']:
                vals = [e[m] for e in eps if not (e[m] != e[m])]
                agg[m] = float(np.mean(vals)) if vals else float('nan')
            subject_results['k_shots'][k] = agg

        all_results[test_sid] = subject_results
        save_subject_checkpoint(output_dir, method_name, seed, test_sid, subject_results)
        del learner
        if 'cuda' in str(device): torch.cuda.empty_cache()

    return {'method': method_name, 'subjects': all_results,
            'k_shots': k_shots, 'seed': seed}
