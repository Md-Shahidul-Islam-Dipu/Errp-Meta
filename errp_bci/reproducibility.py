"""Determinism and reproducibility helpers (ported verbatim from the notebooks)."""
import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Enforce full determinism across all random sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def freeze_batchnorm(model: nn.Module) -> None:
    """Put all BatchNorm layers in eval mode and disable running stat updates.

    Essential during inner-loop adaptation: batch sizes of K=5-20 produce
    unreliable batch statistics that corrupt running mean/variance.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            m.track_running_stats = False
