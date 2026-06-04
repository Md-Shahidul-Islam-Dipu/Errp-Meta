"""Pretrain-FT — fair-transfer baseline (FIXED; addresses flaw #1).

Pools all N-1 training subjects to pretrain an MLP backbone, then fits a
closed-form logistic-regression head on the K new-user trials (Pretrain-FT), and
also reports the no-adaptation transfer (Pretrain-ZeroShot). This is the
decision-critical baseline that disentangles "meta-learning works" from "uses
more data".

FIX vs the original notebook cell (see CHANGES.md): the buggy version moved the
ENTIRE pooled training set onto the GPU at once (``torch.FloatTensor(tr_X).to
(device)``), which OOMs on large folds, and omitted float32 casts. This version
keeps training data on CPU and moves each batch to the device, casts features to
float32, builds the model on-device, and uses ``set(supp_idx)`` membership.
"""
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from ..progress import tqdm

from ..config import Config
from ..reproducibility import set_seed
from ..metrics import compute_comprehensive_metrics
from ..io_results import save_subject_checkpoint, load_subject_checkpoint


def _make_pretrain_backbone(in_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    """3-layer MLP backbone matching the Supervised / MAML architecture."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(hidden_dim, 64),    nn.LayerNorm(64),          nn.GELU(), nn.Dropout(0.3),
    )


def run_pretrain_ft_baseline_loso(
        loso_splits: Dict,
        k_shots: List[int],
        hidden_dim: int = 128,
        lr: float = 1e-3,
        pretrain_epochs: int = 200,
        device: str = str(Config.DEVICE),
        seed: int = 42,
        output_dir: str = Config.RESULTS_DIR) -> Tuple[Dict, Dict]:
    """Fair-transfer baseline: pretrain MLP backbone on N-1 subjects, then
    fit a closed-form logistic-regression head on K new-user trials.

    Two variants are saved:
      Pretrain-FT       -- backbone pretrained on N-1 subjects, head fit on K shots
      Pretrain-ZeroShot -- backbone pretrained on N-1 subjects, no K-shot adaptation

    Both use the same data access as meta-learners (all N-1 training subjects),
    but no meta-learning objective. Comparing meta-learners against Pretrain-FT
    tests whether the meta-learning objective itself is responsible for gains,
    not just the larger training set.
    """
    set_seed(seed)
    all_ft: Dict = {}
    all_zs: Dict = {}

    for test_sid, fold in tqdm(loso_splits.items(), desc='Pretrain-FT LOSO'):
        fold_seed = seed + hash(test_sid) % 10000 + 77
        set_seed(fold_seed)

        # Resume from checkpoint if both variants already computed
        ckpt_ft = load_subject_checkpoint(output_dir, 'Pretrain-FT',       seed, test_sid)
        ckpt_zs = load_subject_checkpoint(output_dir, 'Pretrain-ZeroShot', seed, test_sid)
        if ckpt_ft is not None and ckpt_zs is not None:
            all_ft[test_sid] = ckpt_ft
            all_zs[test_sid] = ckpt_zs
            continue

        # Gather PCA-reduced features for train subjects and test subject
        train_sids = fold['train_subjects']
        pca        = fold['pca']
        subj_feats = fold['subjects_features']  # raw (pre-PCA) features

        tr_X = np.concatenate([
            pca.transform(subj_feats[s]['features']) for s in train_sids
        ]).astype(np.float32)
        tr_y = np.concatenate([subj_feats[s]['labels'] for s in train_sids])
        ts_X = fold['test']['features'].astype(np.float32)  # already PCA-transformed
        ts_y = fold['test']['labels']
        input_dim = tr_X.shape[1]

        # Build backbone + head, both explicitly on device
        backbone = _make_pretrain_backbone(input_dim, hidden_dim).to(device)
        head     = nn.Linear(64, 2).to(device)
        model    = nn.Sequential(backbone, head).to(device)

        counts = np.bincount(tr_y, minlength=2).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        cw     = torch.FloatTensor(2.0 / (counts * counts.sum())).to(device)
        crit   = nn.CrossEntropyLoss(weight=cw)
        opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # Keep training data on CPU to avoid large GPU allocations across folds
        tx = torch.FloatTensor(tr_X)
        ty = torch.LongTensor(tr_y)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tx, ty),
            batch_size=128, shuffle=True, drop_last=False)

        # Pretrain with early stopping
        best_loss, patience_ctr, best_state = float('inf'), 0, None
        model.train()
        for _ in tqdm(range(pretrain_epochs),
                      desc=f'Pretrain-FT {test_sid} pretrain', every_n=25, leave=False):
            epoch_loss = 0.0
            for bx, by in dl:
                bx, by = bx.to(device), by.to(device)  # move batch to device
                opt.zero_grad()
                loss = crit(model(bx), by)
                if not torch.isnan(loss):
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()
            epoch_loss /= max(len(dl), 1)
            if epoch_loss < best_loss - 1e-4:
                best_loss    = epoch_loss
                best_state   = deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= 15:
                break
        if best_state is not None:
            model.load_state_dict(best_state)

        # Freeze backbone; extract test-subject features
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            ts_feat = backbone(
                torch.FloatTensor(ts_X).to(device)
            ).cpu().numpy()

        # Zero-shot evaluation: use the N-1-subject head without any adaptation
        head.eval()
        with torch.no_grad():
            lgts  = head(torch.FloatTensor(ts_feat).to(device))
            probs = F.softmax(lgts, dim=1).cpu().numpy()
            preds = lgts.argmax(dim=1).cpu().numpy()
        zs_metrics = compute_comprehensive_metrics(preds, ts_y, probs)

        subj_zs = {'subject_id': test_sid, 'k_shots': {}}
        for k in k_shots:
            subj_zs['k_shots'][k] = zs_metrics  # identical for all K (no adaptation)
        all_zs[test_sid] = subj_zs
        save_subject_checkpoint(output_dir, 'Pretrain-ZeroShot', seed, test_sid, subj_zs)

        # K-shot fine-tuning: fit logistic regression on K support features
        subj_ft = {'subject_id': test_sid, 'k_shots': {}}
        for k in k_shots:
            set_seed(fold_seed + k)

            # Balanced K-shot support set from test subject
            unique_cls = np.unique(ts_y)
            k_per_cls  = max(1, k // len(unique_cls))
            supp_idx: List[int] = []
            for cls in unique_cls:
                cls_idx = np.where(ts_y == cls)[0]
                n = min(k_per_cls, len(cls_idx))
                supp_idx.extend(np.random.choice(cls_idx, size=n, replace=False).tolist())
            supp_idx  = np.array(supp_idx)
            query_idx = np.array([i for i in range(len(ts_X)) if i not in set(supp_idx)])

            sx_feat = ts_feat[supp_idx]
            sy      = ts_y[supp_idx]
            qx_feat = ts_feat[query_idx]
            qy      = ts_y[query_idx]

            if len(np.unique(sy)) < 2 or len(qy) == 0:
                subj_ft['k_shots'][k] = {
                    m: float('nan')
                    for m in ['accuracy', 'balanced_accuracy', 'f1_score', 'auroc']
                }
                continue

            clf = LogisticRegression(
                C=1.0, max_iter=1000, solver='lbfgs',
                class_weight='balanced', random_state=seed)
            clf.fit(sx_feat, sy)
            preds = clf.predict(qx_feat)
            probs = clf.predict_proba(qx_feat)
            subj_ft['k_shots'][k] = compute_comprehensive_metrics(preds, qy, probs)

        all_ft[test_sid] = subj_ft
        save_subject_checkpoint(output_dir, 'Pretrain-FT', seed, test_sid, subj_ft)

        del model, backbone, head
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

    ft_result = {'method': 'Pretrain-FT',       'subjects': all_ft,
                 'k_shots': k_shots, 'seed': seed}
    zs_result = {'method': 'Pretrain-ZeroShot', 'subjects': all_zs,
                 'k_shots': k_shots, 'seed': seed}
    return ft_result, zs_result
