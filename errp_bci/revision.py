"""Camera-ready revision experiments (reviewer requests E1-E5).

One self-contained runner that trains every method ONCE per LOSO fold and then
evaluates it under several calibration protocols on IDENTICAL support draws, so
single-draw comparisons between methods are paired. Nothing here changes the
code paths behind the published tables; the model classes are reused as-is.

    E1  realistic calibration   protocols: balanced | natural | contiguous
    E2  input representation    ANIL / Reptile on an EEGNet backbone (raw epochs)
    E3  fine-tuned EEGNet       end-to-end fine-tuning on the K support trials
    E4  compute cost            source-training seconds + per-user adaptation ms
    E5  Riemannian comparators  xDAWN-MDM (calibration only) and MDWM

Calibration protocols (the same support indices are used by every method):
    balanced    K//2 trials per class, random  (the paper's protocol; K=4 -> 2+2)
    natural     K random trials, class ignored (errors at the user's own rate)
    contiguous  K consecutive trials from a random start; draw 0 = the first K
                trials of the recording (what a real user actually supplies)
The query set is every trial of the held-out subject not in the support set
(natural prevalence; balanced accuracy is prevalence-invariant).

If a draw contains a single class, gradient methods adapt on it anyway (that is
what the deployed method would do); methods that cannot fit a one-class
classifier (logistic probe, MDM, MDWM) fall back to their source-only model.
Every row records ``n_err_support`` and ``fallback`` so this is auditable.

Hyperparameters are the published ones (``Config``); the few new ones
(EEGNet fine-tuning, MDWM trade-off) are fixed a priori below and never tuned.

Usage (Kaggle):
    from errp_bci.revision import run_revision
    run_revision("inria", seeds=[42])                       # everything
    run_revision("coadaptation", seeds=[42], batch=(0, 8))  # folds 0..7 only
Outputs (under /kaggle/working/Revision_<dataset>/):
    ckpt/<group>_<seed>_<subject>.json   per-fold checkpoint (resume-safe)
    draws.csv    one row per (method, seed, subject, protocol, K, draw)
    timing.csv   one row per (method group, seed, subject)
    burden.csv   raw trials needed to collect K/2 errors, per subject
"""
import json
import os
import time
import zlib
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression

from .config import Config, ExperimentConfig
from .data.loaders import load_dataset, build_features_and_loso
from .metrics import compute_comprehensive_metrics
from .progress import tqdm
from .reproducibility import set_seed, freeze_batchnorm
from .models.encoders import MetaEEGClassifier
from .models.maml import MAML_Encoder
from .models.reptile import ReptileMetaLearner
from .models.subject_conditioned import SubjectConditionedMetaLearner
from .models.pretrain_ft import _make_pretrain_backbone
from .models.eegnet import EEGNet
from .models.prototypical import create_balanced_episode

PROTOCOLS = ("balanced", "natural", "contiguous")
K_SHOTS = (4, 10, 20)
N_DRAWS = 10

# New hyperparameters, fixed a priori (not tuned on any subject).
EEGNET_FT_LR = 1e-4        # end-to-end fine-tuning: Adam, all layers, BN frozen
EEGNET_FT_STEPS = 50
MDWM_TRADEOFF = 0.5        # MDWM source/target trade-off (midpoint, not tuned)
N_XDAWN = 4

GROUPS = ["Supervised", "PCA-Pretrain", "Full-MAML", "MAML-ANIL", "Reptile",
          "SubjectConditioned", "EEGNet", "ANIL-EEGNet", "Reptile-EEGNet",
          "Riemann"]


# ── Helpers ────────────────────────────────────────────────────────────────
def _stable(*parts) -> int:
    """Process-independent integer seed (Python's str hash is salted)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def _metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    probs = np.asarray(probs, dtype=float)
    preds = probs.argmax(1)
    m = compute_comprehensive_metrics(preds, y, probs)
    npos, nneg = m["tp"] + m["fn"], m["tn"] + m["fp"]
    m["tpr"] = m["tp"] / npos if npos else float("nan")
    m["tnr"] = m["tn"] / nneg if nneg else float("nan")
    return m


def _cw(labels: np.ndarray, device) -> torch.Tensor:
    c = np.bincount(labels, minlength=2).astype(float)
    c = np.where(c == 0, 1.0, c)
    w = 1.0 / c
    return torch.FloatTensor(w / w.sum() * 2).to(device)


def _sync(device):
    if "cuda" in str(device):
        torch.cuda.synchronize()


def make_draws(labels: np.ndarray, sid: str, seed: int) -> List[dict]:
    """Support index sets shared by all methods for one subject."""
    n = len(labels)
    draws = []
    for proto in PROTOCOLS:
        for k in K_SHOTS:
            for d in range(N_DRAWS):
                rng = np.random.RandomState(_stable(seed, sid, proto, k, d))
                if proto == "balanced":
                    idx = []
                    for c in (0, 1):
                        ci = np.where(labels == c)[0]
                        idx += list(rng.choice(ci, size=min(k // 2, len(ci)), replace=False))
                    idx = np.array(idx)
                elif proto == "natural":
                    idx = rng.choice(n, size=k, replace=False)
                else:
                    start = 0 if d == 0 else rng.randint(0, n - k + 1)
                    idx = np.arange(start, start + k)
                draws.append({"protocol": proto, "k": k, "draw": d,
                              "idx": np.sort(idx).astype(int)})
    return draws


def calibration_burden(labels: np.ndarray, per_class: Sequence[int] = (1, 2, 5)) -> List[dict]:
    """Raw trials a user must perform (from each start point, in recording
    order) until at least ``m`` error AND ``m`` correct trials are collected."""
    out = []
    y = np.asarray(labels)
    for m in per_class:
        need = []
        for s in range(len(y)):
            e = c = 0
            for j in range(s, len(y)):
                e += y[j] == 1; c += y[j] == 0
                if e >= m and c >= m:
                    need.append(j - s + 1); break
        need = np.array(need) if need else np.array([np.nan])
        out.append({"per_class": m, "error_rate": float(y.mean()),
                    "mean": float(np.mean(need)), "median": float(np.median(need)),
                    "p90": float(np.percentile(need, 90)),
                    "n_starts": int(len(need))})
    return out


# ── EEGNet source pretraining (verbatim protocol of models/eegnet.py) ──────
def _pretrain_eegnet(tr_ep, tr_lb, device, epochs=100, lr=1e-3) -> EEGNet:
    model = EEGNet(Config.N_CHANNELS, Config.N_TIMES, 2, Config.EEGNET_F1,
                   Config.EEGNET_D, Config.EEGNET_F2, Config.KERNEL_LENGTH,
                   Config.EEGNET_DROPOUT).to(device)
    counts = np.bincount(tr_lb, minlength=2).astype(float)
    crit = nn.CrossEntropyLoss(weight=torch.FloatTensor(2.0 / (counts * counts.sum())).to(device))
    opt = optim.Adam(model.parameters(), lr=lr)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.FloatTensor(tr_ep[:, None]), torch.LongTensor(tr_lb)),
        batch_size=64, shuffle=True)
    best, pat, best_state = float("inf"), 0, None
    model.train()
    for _ in range(epochs):
        tot = 0.0
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            if not torch.isnan(loss):
                loss.backward(); opt.step(); tot += loss.item()
        tot /= max(len(dl), 1)
        if tot < best - 1e-4:
            best, pat, best_state = tot, 0, deepcopy(model.state_dict())
        else:
            pat += 1
        if pat >= 15:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _eegnet_forward(model, x, device, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(torch.FloatTensor(x[i:i + bs, None]).to(device)).cpu())
    return F.softmax(torch.cat(out), 1).numpy()


def _eegnet_feats(model, x, device, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model._features(torch.FloatTensor(x[i:i + bs, None]).to(device)).cpu())
    return torch.cat(out).numpy()


# ── Method groups: each returns (train_seconds, evaluator) ─────────────────
# An evaluator maps (support_idx, query_idx) -> {method_name: (probs, fallback)}.

def _episodes(train, n_support, n_query, meta_batch):
    sids = list(train)
    batch = np.random.choice(len(sids), size=min(meta_batch, len(sids)), replace=False)
    tasks = []
    for b in batch:
        f, l = train[sids[b]]
        if len(f) >= n_support + n_query:
            tasks.append(MAML_Encoder._balanced_episode(f, l, n_support, n_query))
    return tasks


def _group_supervised(ctx, device):
    X, y = ctx["test_X"], ctx["test_y"]

    def ev(s, q):
        sy = y[s]
        model = nn.Sequential(
            nn.Linear(X.shape[1], 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 2)).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(weight=_cw(sy, device))
        tx, ty = torch.FloatTensor(X[s]).to(device), torch.LongTensor(sy).to(device)
        best, pat, best_state = float("inf"), 0, None
        model.train()
        for _ in range(200):
            opt.zero_grad(); loss = crit(model(tx), ty); loss.backward(); opt.step()
            if loss.item() < best - 1e-5:
                best, pat, best_state = loss.item(), 0, deepcopy(model.state_dict())
            else:
                pat += 1
            if pat >= 20:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p = F.softmax(model(torch.FloatTensor(X[q]).to(device)), 1).cpu().numpy()
        return {"Supervised": (p, False)}
    return 0.0, ev


def _group_pca_pretrain(ctx, device):
    """PCA-input, pretrain + frozen LR probe (= the paper's Pretrain-FT)."""
    tr_X = np.concatenate([f for f, _ in ctx["train_pca"].values()])
    tr_y = np.concatenate([l for _, l in ctx["train_pca"].values()])
    t0 = time.perf_counter()
    backbone = _make_pretrain_backbone(tr_X.shape[1], 128).to(device)
    head = nn.Linear(64, 2).to(device)
    model = nn.Sequential(backbone, head)
    counts = np.bincount(tr_y, minlength=2).astype(float)
    crit = nn.CrossEntropyLoss(weight=torch.FloatTensor(2.0 / (counts * counts.sum())).to(device))
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.FloatTensor(tr_X), torch.LongTensor(tr_y)), batch_size=128, shuffle=True)
    best, pat, best_state = float("inf"), 0, None
    model.train()
    for _ in range(200):
        tot = 0.0
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = crit(model(bx), by); loss.backward(); opt.step()
            tot += loss.item()
        tot /= max(len(dl), 1)
        if tot < best - 1e-4:
            best, pat, best_state = tot, 0, deepcopy(model.state_dict())
        else:
            pat += 1
        if pat >= 15:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval(); _sync(device)
    train_s = time.perf_counter() - t0
    with torch.no_grad():
        feats = backbone(torch.FloatTensor(ctx["test_X"]).to(device)).cpu().numpy()
        zs = F.softmax(head(torch.FloatTensor(feats).to(device)), 1).cpu().numpy()
    y = ctx["test_y"]

    def ev(s, q):
        if len(np.unique(y[s])) < 2:
            return {"PCA-Probe": (zs[q], True), "PCA-ZeroShot": (zs[q], False)}
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        clf.fit(feats[s], y[s])
        return {"PCA-Probe": (clf.predict_proba(feats[q]), False),
                "PCA-ZeroShot": (zs[q], False)}
    return train_s, ev


def _group_maml(ctx, device, first_order_head_only: bool, name: str):
    t0 = time.perf_counter()
    agent = MAML_Encoder(input_dim=ctx["test_X"].shape[1],
                         freeze_encoder_inner=first_order_head_only,
                         first_order=first_order_head_only, device=device)
    for _ in tqdm(range(Config.N_META_ITERATIONS), desc=f"{name} {ctx['sid']}",
                  every_n=500, leave=False):
        tasks = _episodes(ctx["train_pca"], Config.N_SUPPORT, Config.N_QUERY, Config.META_BATCH_SIZE)
        if tasks:
            agent.meta_update(tasks)
    _sync(device)
    train_s = time.perf_counter() - t0
    X, y = ctx["test_X"], ctx["test_y"]

    def ev(s, q):
        # mirrors MAML_Encoder.adapt_and_evaluate, returning probabilities
        m = deepcopy(agent.meta_model); freeze_batchnorm(m)
        if agent.freeze_enc:
            params = list(m.task_head.parameters())
            for p in m.encoder.parameters():
                p.requires_grad_(False)
        else:
            params = list(m.parameters())
        opt = optim.SGD(params, lr=agent.inner_lr)
        sx, sy = torch.FloatTensor(X[s]).to(device), torch.LongTensor(y[s]).to(device)
        cw = agent._cw(y[s]); m.train()
        for _ in range(agent.inner_steps):
            opt.zero_grad(); F.cross_entropy(m(sx), sy, weight=cw).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = F.softmax(m(torch.FloatTensor(X[q]).to(device)), 1).cpu().numpy()
        return {name: (p, False)}
    return train_s, ev


def _group_reptile(ctx, device):
    t0 = time.perf_counter()
    rep = ReptileMetaLearner(input_dim=ctx["test_X"].shape[1], device=device)
    for _ in tqdm(range(Config.N_META_ITERATIONS), desc=f"Reptile {ctx['sid']}",
                  every_n=500, leave=False):
        tasks = [(sx, sy) for sx, sy, _, _ in
                 [MAML_Encoder._balanced_episode(*ctx["train_pca"][s], Config.N_SUPPORT, 0)
                  for s in np.random.choice(list(ctx["train_pca"]), size=4, replace=False)]]
        rep.meta_update(tasks)
    _sync(device)
    train_s = time.perf_counter() - t0
    X, y = ctx["test_X"], ctx["test_y"]

    def ev(s, q):
        m = rep._adapt(X[s], y[s]); m.eval()
        with torch.no_grad():
            p = F.softmax(m(torch.FloatTensor(X[q]).to(device)), 1).cpu().numpy()
        return {"Reptile": (p, False)}
    return train_s, ev


def _group_subjcond(ctx, device):
    t0 = time.perf_counter()
    L = SubjectConditionedMetaLearner(input_dim=ctx["test_X"].shape[1], device=device)
    for _ in tqdm(range(Config.N_META_ITERATIONS), desc=f"SubjCond {ctx['sid']}",
                  every_n=500, leave=False):
        tasks = _episodes(ctx["train_pca"], Config.N_SUPPORT, Config.N_QUERY, Config.META_BATCH_SIZE)
        if tasks:
            L.meta_update(tasks)
    _sync(device)
    train_s = time.perf_counter() - t0
    X, y = ctx["test_X"], ctx["test_y"]

    def ev(s, q):
        # mirrors SubjectConditionedMetaLearner.adapt_and_evaluate
        sx, sy = torch.FloatTensor(X[s]).to(device), torch.LongTensor(y[s]).to(device)
        L.subject_encoder.eval(); L.cond_encoder.eval()
        with torch.no_grad():
            z = L.subject_encoder(sx, sy)
        tp = L._inner_adapt(sx, sy, z, L._cw(y[s], device))
        with torch.no_grad():
            h = L.cond_encoder(torch.FloatTensor(X[q]).to(device), z)
            p = F.softmax(F.linear(h, tp["fc.weight"], tp["fc.bias"]), 1).cpu().numpy()
        L.subject_encoder.train(); L.cond_encoder.train()
        return {"SubjectConditioned": (p, False)}
    return train_s, ev


def _group_eegnet(ctx, device):
    """E3: source-pretrained EEGNet -> LR probe (paper), end-to-end FT, zero-shot."""
    t0 = time.perf_counter()
    model = _pretrain_eegnet(ctx["train_ep"], ctx["train_lb"], device)
    _sync(device)
    train_s = time.perf_counter() - t0
    E, y = ctx["test_ep"], ctx["test_y"]
    feats = _eegnet_feats(model, E, device)
    zs = _eegnet_forward(model, E, device)

    def ev(s, q):
        out = {"EEGNet-ZeroShot": (zs[q], False)}
        sy = y[s]
        if len(np.unique(sy)) < 2:
            out["EEGNet"] = (zs[q], True)
        else:
            # the paper's probe, incl. its mixup augmentation of the support at K<10
            sx_f = feats[s]
            if len(s) < 10:
                aug_x, aug_y = [], []
                for i in range(len(sx_f)):
                    same = np.where(sy == sy[i])[0]
                    for _ in range(3):
                        j = np.random.choice(same); a = np.random.uniform(0.3, 0.7)
                        aug_x.append(a * sx_f[i] + (1 - a) * sx_f[j]); aug_y.append(sy[i])
                sx_f = np.concatenate([sx_f, np.array(aug_x)]); sy_f = np.concatenate([sy, aug_y])
            else:
                sy_f = sy
            clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
            clf.fit(sx_f, sy_f)
            out["EEGNet"] = (clf.predict_proba(feats[q]), False)
        # end-to-end fine-tuning of every layer, starting from the source head
        m = deepcopy(model); m.train(); freeze_batchnorm(m)   # BN stays on source stats
        opt = optim.Adam(m.parameters(), lr=EEGNET_FT_LR)
        sx = torch.FloatTensor(E[s][:, None]).to(device)
        sy_t = torch.LongTensor(sy).to(device); cw = _cw(sy, device)
        for _ in range(EEGNET_FT_STEPS):
            opt.zero_grad(); F.cross_entropy(m(sx), sy_t, weight=cw).backward(); opt.step()
        m.eval()
        out["EEGNet-FT"] = (_eegnet_forward(m, E[q], device), False)
        del m
        return out
    return train_s, ev


def _group_anil_eegnet(ctx, device):
    """E2: ANIL with an EEGNet backbone on raw epochs (head-only inner loop)."""
    t0 = time.perf_counter()
    model = EEGNet(Config.N_CHANNELS, Config.N_TIMES, 2, Config.EEGNET_F1, Config.EEGNET_D,
                   Config.EEGNET_F2, Config.KERNEL_LENGTH, Config.EEGNET_DROPOUT).to(device)
    opt = optim.Adam(model.parameters(), lr=Config.OUTER_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, Config.N_META_ITERATIONS,
                                                 eta_min=Config.OUTER_LR * 0.01)

    def inner(feat, sy, cw, W, b, create_graph):
        for _ in range(Config.INNER_STEPS):
            loss = F.cross_entropy(F.linear(feat, W, b), sy, weight=cw)
            gW, gb = torch.autograd.grad(loss, (W, b), create_graph=create_graph)
            W, b = W - Config.INNER_LR * gW, b - Config.INNER_LR * gb
        return W, b

    model.train()
    for _ in tqdm(range(Config.N_META_ITERATIONS), desc=f"ANIL-EEGNet {ctx['sid']}",
                  every_n=500, leave=False):
        tasks = _episodes(ctx["train_raw"], Config.N_SUPPORT, Config.N_QUERY, Config.META_BATCH_SIZE)
        if not tasks:
            continue
        opt.zero_grad(); losses = []
        for sx, sy, qx, qy in tasks:
            x = torch.FloatTensor(np.concatenate([sx, qx])[:, None]).to(device)
            feat = model._features(x)
            fs, fq = feat[:len(sx)], feat[len(sx):]
            sy_t, qy_t = torch.LongTensor(sy).to(device), torch.LongTensor(qy).to(device)
            W, b = inner(fs, sy_t, _cw(sy, device), model.classifier.weight,
                         model.classifier.bias, True)
            losses.append(F.cross_entropy(F.linear(fq, W, b), qy_t, weight=_cw(qy, device)))
        torch.stack(losses).mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval(); _sync(device)
    train_s = time.perf_counter() - t0
    E, y = ctx["test_ep"], ctx["test_y"]
    feats = torch.FloatTensor(_eegnet_feats(model, E, device)).to(device)
    with torch.no_grad():   # meta-learned initialization, no test-time adaptation
        zs = F.softmax(model.classifier(feats), 1).cpu().numpy()

    def ev(s, q):
        W = model.classifier.weight.detach().clone().requires_grad_(True)
        b = model.classifier.bias.detach().clone().requires_grad_(True)
        W, b = inner(feats[s], torch.LongTensor(y[s]).to(device), _cw(y[s], device), W, b, False)
        with torch.no_grad():
            p = F.softmax(F.linear(feats[q], W, b), 1).cpu().numpy()
        return {"ANIL-EEGNet": (p, False), "ANIL-EEGNet-ZeroShot": (zs[q], False)}
    return train_s, ev


def _group_reptile_eegnet(ctx, device):
    """E2: Reptile with an EEGNet backbone on raw epochs (full inner loop)."""
    t0 = time.perf_counter()
    model = EEGNet(Config.N_CHANNELS, Config.N_TIMES, 2, Config.EEGNET_F1, Config.EEGNET_D,
                   Config.EEGNET_F2, Config.KERNEL_LENGTH, Config.EEGNET_DROPOUT).to(device)
    eps = 0.1

    def adapt(base, sx, sy, freeze_bn):
        m = deepcopy(base)
        m.train()
        if freeze_bn:          # after .train(), which would re-enable batch stats
            freeze_batchnorm(m)
        o = optim.SGD(m.parameters(), lr=Config.INNER_LR)
        x, t = torch.FloatTensor(sx[:, None]).to(device), torch.LongTensor(sy).to(device)
        cw = _cw(sy, device)
        for _ in range(Config.INNER_STEPS):
            o.zero_grad(); F.cross_entropy(m(x), t, weight=cw).backward(); o.step()
        return m

    for _ in tqdm(range(Config.N_META_ITERATIONS), desc=f"Reptile-EEGNet {ctx['sid']}",
                  every_n=500, leave=False):
        sids = np.random.choice(list(ctx["train_raw"]), size=4, replace=False)
        adapted = []
        for s in sids:
            sx, sy, _, _ = MAML_Encoder._balanced_episode(*ctx["train_raw"][s], Config.N_SUPPORT, 0)
            adapted.append(adapt(model, sx, sy, freeze_bn=False))
        with torch.no_grad():
            for mp, *tps in zip(model.parameters(), *[a.parameters() for a in adapted]):
                mp.add_(eps * (torch.stack([p for p in tps]).mean(0) - mp))
            # BatchNorm running statistics are buffers, not parameters: move them too
            for mb, *tbs in zip(model.buffers(), *[a.buffers() for a in adapted]):
                if mb.dtype.is_floating_point:
                    mb.add_(eps * (torch.stack([t for t in tbs]).mean(0) - mb))
        del adapted
    model.eval(); _sync(device)
    train_s = time.perf_counter() - t0
    E, y = ctx["test_ep"], ctx["test_y"]
    zs = _eegnet_forward(model, E, device)   # meta-learned initialization, unadapted

    def ev(s, q):
        m = adapt(model, E[s], y[s], freeze_bn=True); m.eval()
        p = _eegnet_forward(m, E[q], device); del m
        return {"Reptile-EEGNet": (p, False), "Reptile-EEGNet-ZeroShot": (zs[q], False)}
    return train_s, ev


def _group_riemann(ctx, device):
    """E5: xDAWN super-trial covariances; MDM on the K trials, and MDWM."""
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.classification import MDM
    try:
        from pyriemann.transfer import MDWM, encode_domains
    except ImportError:
        MDWM = None
    t0 = time.perf_counter()
    xc = None
    for kw in ({"nfilter": N_XDAWN}, {"n_filter": N_XDAWN}, {"n_components": N_XDAWN}):
        try:
            xc = XdawnCovariances(estimator="lwf", xdawn_estimator="lwf", **kw); break
        except TypeError:
            continue
    xc.fit(ctx["train_ep"].astype(np.float64), ctx["train_lb"])
    src_cov = xc.transform(ctx["train_ep"].astype(np.float64))
    tst_cov = xc.transform(ctx["test_ep"].astype(np.float64))
    src_y = ctx["train_lb"]
    mdm_src = MDM(metric="riemann").fit(src_cov, src_y)
    zs = mdm_src.predict_proba(tst_cov)
    train_s = time.perf_counter() - t0
    y = ctx["test_y"]

    def ev(s, q):
        out = {}
        if len(np.unique(y[s])) < 2:
            out["xDAWN-MDM"] = (zs[q], True)
            if MDWM is not None:
                out["MDWM"] = (zs[q], True)
            return out
        out["xDAWN-MDM"] = (MDM(metric="riemann").fit(tst_cov[s], y[s]).predict_proba(tst_cov[q]), False)
        if MDWM is not None:
            Xa = np.concatenate([src_cov, tst_cov[s]])
            ya = np.concatenate([src_y, y[s]])
            dom = np.array(["source"] * len(src_y) + ["target"] * len(s))
            try:
                Xe, ye = encode_domains(Xa, ya, dom)
                clf = MDWM(domain_tradeoff=MDWM_TRADEOFF, target_domain="target", metric="riemann")
                clf.fit(Xe, ye)
                p = clf.predict_proba(tst_cov[q])
                order = np.argsort(np.asarray(clf.classes_).astype(int))
                out["MDWM"] = (p[:, order], False)
            except Exception as exc:
                print(f"    MDWM failed ({exc}); source-model fallback")
                out["MDWM"] = (zs[q], True)
        return out
    return train_s, ev


_BUILDERS = {
    "Supervised": _group_supervised,
    "PCA-Pretrain": _group_pca_pretrain,
    "Full-MAML": lambda c, d: _group_maml(c, d, False, "Full-MAML"),
    "MAML-ANIL": lambda c, d: _group_maml(c, d, True, "MAML-ANIL"),
    "Reptile": _group_reptile,
    "SubjectConditioned": _group_subjcond,
    "EEGNet": _group_eegnet,
    "ANIL-EEGNet": _group_anil_eegnet,
    "Reptile-EEGNet": _group_reptile_eegnet,
    "Riemann": _group_riemann,
}


# ── Runner ─────────────────────────────────────────────────────────────────
def _out_root(dataset: str) -> str:
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(os.getcwd(), "Results")
    return os.path.join(base, f"Revision_{dataset}")


def run_revision(dataset: str, seeds: Sequence[int] = (42,),
                 groups: Optional[Sequence[str]] = None,
                 batch: Optional[Tuple[int, int]] = None,
                 dataset_root: Optional[str] = None,
                 out_dir: Optional[str] = None) -> str:
    """Run the revision experiments. ``batch=(start, end)`` restricts the
    held-out subjects to sorted indices [start, end) so the 16 folds can be
    split across Kaggle sessions; finished folds are skipped on re-run."""
    groups = list(groups or GROUPS)
    unknown = set(groups) - set(_BUILDERS)
    if unknown:
        raise ValueError(f"Unknown groups {unknown}; choose from {GROUPS}")
    device = str(Config.DEVICE)
    out = out_dir or _out_root(dataset)
    ck = os.path.join(out, "ckpt"); os.makedirs(ck, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"=== Revision | {dataset} | groups={groups} | seeds={list(seeds)} | {gpu} ===")

    ecfg = ExperimentConfig(name="revision", dataset=dataset, dataset_root=dataset_root)
    ppd = load_dataset(ecfg)
    feats, loso = build_features_and_loso(ppd, 32)
    sids = sorted(loso)
    todo = sids[batch[0]:batch[1]] if batch else sids

    # calibration burden only depends on the labels
    rows = []
    for sid in sids:
        for r in calibration_burden(ppd[sid]["labels"]):
            rows.append({"dataset": dataset, "subject": sid, **r})
    pd.DataFrame(rows).to_csv(os.path.join(out, "burden.csv"), index=False)

    for seed in seeds:
        for sid in todo:
            fold = loso[sid]
            ctx = None
            for g in groups:
                path = os.path.join(ck, f"{g}_{seed}_{sid}.json")
                if os.path.exists(path):
                    continue
                if ctx is None:
                    tr = fold["train_subjects"]
                    ctx = {"sid": sid,
                           "test_X": fold["test"]["features"], "test_y": fold["test"]["labels"],
                           "test_ep": ppd[sid]["epochs"],
                           "train_pca": {s: (fold["pca"].transform(feats[s]["features"]),
                                             feats[s]["labels"]) for s in tr},
                           "train_raw": {s: (ppd[s]["epochs"], ppd[s]["labels"]) for s in tr},
                           "train_ep": np.concatenate([ppd[s]["epochs"] for s in tr]),
                           "train_lb": np.concatenate([ppd[s]["labels"] for s in tr])}
                    draws = make_draws(ctx["test_y"], sid, seed)
                print(f"  [{seed}] {sid} {g} ...", flush=True)
                set_seed(_stable(seed, sid, g))
                train_s, ev = _BUILDERS[g](ctx, device)
                recs, adapt_ms = [], []
                y = ctx["test_y"]
                for d in draws:
                    s = d["idx"]; q = np.setdiff1d(np.arange(len(y)), s)
                    np.random.seed(_stable(seed, sid, g, d["protocol"], d["k"], d["draw"]))
                    t1 = time.perf_counter()
                    res = ev(s, q)
                    _sync(device)
                    adapt_ms.append(1000 * (time.perf_counter() - t1))
                    for meth, (probs, fb) in res.items():
                        m = _metrics(y[q], probs)
                        recs.append({"dataset": dataset, "method": meth, "seed": seed,
                                     "subject": sid, "protocol": d["protocol"], "k": d["k"],
                                     "draw": d["draw"], "n_err_support": int(y[s].sum()),
                                     "fallback": bool(fb),
                                     **{x: m[x] for x in ("balanced_accuracy", "auroc", "tpr",
                                                          "tnr", "tp", "fp", "tn", "fn")}})
                timing = {"dataset": dataset, "group": g, "seed": seed, "subject": sid,
                          "train_s": train_s, "adapt_ms_mean": float(np.mean(adapt_ms)),
                          "n_evals": len(adapt_ms), "device": gpu,
                          "n_train_trials": int(len(ctx["train_lb"]))}
                with open(path + ".tmp", "w") as f:
                    json.dump({"rows": recs, "timing": timing}, f, default=float)
                os.replace(path + ".tmp", path)
                if "cuda" in device:
                    torch.cuda.empty_cache()
    collect(out)
    return out


def collect(out: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge all fold checkpoints under ``out`` into draws.csv / timing.csv."""
    ck = os.path.join(out, "ckpt")
    rows, tim = [], []
    for fn in sorted(os.listdir(ck)):
        if fn.endswith(".json"):
            with open(os.path.join(ck, fn)) as f:
                d = json.load(f)
            rows += d["rows"]; tim.append(d["timing"])
    draws, timing = pd.DataFrame(rows), pd.DataFrame(tim)
    draws.to_csv(os.path.join(out, "draws.csv"), index=False)
    timing.to_csv(os.path.join(out, "timing.csv"), index=False)
    print(f"collected {len(draws)} draw rows, {len(timing)} timing rows -> {out}")
    return draws, timing
