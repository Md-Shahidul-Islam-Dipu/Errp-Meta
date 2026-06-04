"""EEGNet baseline with closed-form logistic-regression head.

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
from sklearn.linear_model import LogisticRegression
from .prototypical import create_balanced_episode

class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al. 2018) — corrected kernel_length.

    Input  : (batch, 1, n_channels, n_times) — raw filtered baseline-corrected waveform
    kernel_length = Config.KERNEL_LENGTH = sfreq // 2 (half-period temporal filter)
    """
    def __init__(self, n_channels: int, n_times: int, n_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 kernel_length: int = 100, dropout: float = 0.25):
        super().__init__()
        # Block 1: temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length),
                               padding=(0, kernel_length // 2), bias=False)
        self.bn1   = nn.BatchNorm2d(F1)
        # Block 2: depthwise spatial convolution
        self.dw    = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2   = nn.BatchNorm2d(F1 * D)
        self.act1  = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dp1   = nn.Dropout(dropout)
        # Block 3: separable convolution
        self.sep_dw = nn.Conv2d(F1*D, F1*D, (1, 16), groups=F1*D, padding=(0, 8), bias=False)
        self.sep_pw = nn.Conv2d(F1*D, F2, (1, 1), bias=False)
        self.bn3    = nn.BatchNorm2d(F2)
        self.act2   = nn.ELU()
        self.pool2  = nn.AvgPool2d((1, 8))
        self.dp2    = nn.Dropout(dropout)

        # Compute feature dim dynamically
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self.dp1(self.pool1(self.act1(self.bn2(self.dw(self.bn1(self.conv1(x)))))))
            x = self.dp2(self.pool2(self.act2(self.bn3(self.sep_pw(self.sep_dw(x))))))
            self.feat_dim = x.view(1, -1).shape[1]

        self.classifier = nn.Linear(self.feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._features(x))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dp1(self.pool1(self.act1(self.bn2(self.dw(self.bn1(self.conv1(x)))))))
        x = self.dp2(self.pool2(self.act2(self.bn3(self.sep_pw(self.sep_dw(x))))))
        return x.view(x.size(0), -1)

def run_eegnet_loso(
        preprocessed_data: dict, loso_splits: dict, k_shots: list,
        pretrain_epochs: int = 100, lr: float = 1e-3,
        device: str = str(Config.DEVICE),
        seed: int = 42, output_dir: str = Config.RESULTS_DIR) -> dict:
    """EEGNet: pretrain on N-1 subjects → freeze → closed-form linear probe.

    Fixes vs original:
    - kernel_length = Config.KERNEL_LENGTH = sfreq//2 (not hardcoded 32)
    - Linear probe = LogisticRegression (not gradient descent fine-tuning)
    - Class-weighted CE for pretraining
    - Early stopping (patience=15)
    - Backbone on raw filtered waveforms (not PCA bandpower features)
    - GPU memory freed after each fold
    """
    set_seed(seed)
    all_results: dict = {}
    method_name = 'EEGNet'
    n_ch, n_t, kl = Config.N_CHANNELS, Config.N_TIMES, Config.KERNEL_LENGTH

    for test_sid in tqdm(sorted(loso_splits.keys()), desc="EEGNet LOSO"):
        fold_seed = seed + hash(test_sid) % 10000 + 99
        set_seed(fold_seed)

        ckpt = load_subject_checkpoint(output_dir, method_name, seed, test_sid)
        if ckpt is not None:
            all_results[test_sid] = ckpt; continue

        train_sids = [s for s in loso_splits.keys() if s != test_sid]
        tr_ep  = np.concatenate([preprocessed_data[s]['epochs'] for s in train_sids])
        tr_lb  = np.concatenate([preprocessed_data[s]['labels'] for s in train_sids])
        ts_ep  = preprocessed_data[test_sid]['epochs']
        ts_lb  = preprocessed_data[test_sid]['labels']

        model = EEGNet(n_ch, n_t, 2, Config.EEGNET_F1, Config.EEGNET_D, Config.EEGNET_F2,
                       kl, Config.EEGNET_DROPOUT).to(device)

        # Class-weighted CE for pretraining
        counts = np.bincount(tr_lb, minlength=2).astype(float)
        cw     = torch.FloatTensor(2.0 / (counts * counts.sum())).to(device)
        crit   = nn.CrossEntropyLoss(weight=cw)
        opt    = optim.Adam(model.parameters(), lr=lr)

        # DataLoader
        tx = torch.FloatTensor(tr_ep[:, np.newaxis, :, :])
        ty = torch.LongTensor(tr_lb)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tx, ty),
            batch_size=64, shuffle=True, drop_last=False)

        # Pretrain with early stopping
        best_loss, patience_ctr, best_state = float('inf'), 0, None
        model.train()
        for epoch in tqdm(range(pretrain_epochs), desc=f'{method_name} {test_sid} pretrain', every_n=25, leave=False):
            epoch_loss = 0.0
            for bx, by in dl:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = crit(model(bx), by)
                if not torch.isnan(loss):
                    loss.backward(); opt.step()
                    epoch_loss += loss.item()
            epoch_loss /= max(len(dl), 1)
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                best_state = deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= 15:
                break
        if best_state:
            model.load_state_dict(best_state)

        # Freeze backbone
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        # Extract features from test subject
        ts_t = torch.FloatTensor(ts_ep[:, np.newaxis, :, :])
        with torch.no_grad():
            ts_feat = model._features(ts_t.to(device)).cpu().numpy()

        # K-shot linear probe (closed-form LogisticRegression)
        subject_results = {'subject_id': test_sid, 'k_shots': {}}
        for k in k_shots:
            eps = []
            for ep in range(Config.N_EVAL_EPISODES):
                set_seed(fold_seed + k * 100 + ep)
                try:
                    sx_f, sy, qx_f, qy = create_balanced_episode(
                        ts_feat, ts_lb, k, 40, augment=(k < 10))
                    if len(np.unique(sy)) < 2: continue
                    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs',
                                             class_weight='balanced', random_state=seed)
                    clf.fit(sx_f, sy)
                    preds = clf.predict(qx_f)
                    probs = clf.predict_proba(qx_f)
                    eps.append(compute_comprehensive_metrics(preds, qy, probs))
                except Exception:
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
