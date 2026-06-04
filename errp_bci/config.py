"""Configuration — single source of truth.

``Config`` holds global hyperparameters and runtime signal fields exactly as in
the original notebooks (model functions read e.g. ``Config.INNER_LR`` as default
arguments, so these attribute names must stay stable). ``ExperimentConfig`` is a
lightweight per-experiment descriptor consumed by the runner; it does NOT
replace ``Config`` — the runner applies its values (paths, pca_components,
preprocessing depth, runtime signal fields) onto ``Config`` and/or passes them
explicitly to the method runners.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


# ── Dataset-root resolution (Kaggle-first, local fallback) ─────────────────
def _resolve_dataset_root(default="/kaggle/input/inria-bci-challenge/inria-bci-challenge"):
    candidates = [default]
    cwd = os.getcwd()
    candidates += [
        os.path.join(cwd, "Code", "inria-bci-challenge"),
        os.path.join(cwd, "inria-bci-challenge"),
        os.path.join(cwd, "..", "Code", "inria-bci-challenge"),
    ]
    seen, uniq = set(), []
    for p in candidates:
        n = os.path.normpath(p)
        if n not in seen:
            seen.add(n); uniq.append(n)
    for root in uniq:
        if (os.path.isfile(os.path.join(root, "TrainLabels.csv"))
                and os.path.isdir(os.path.join(root, "train"))):
            return root
    return os.path.normpath(default)


def _resolve_coadaptation_dir(default="/kaggle/input/datasets/dipuislam/errp-coadaption/data"):
    """Best-effort resolution of the ErrP-Coadaptation .set directory.

    Mirrors the validation notebook's resolver: try the Kaggle default, then a
    few local candidates. The coadaptation loader also accepts an explicit dir.
    """
    candidates = [default]
    cwd = os.getcwd()
    candidates += [
        os.path.join(cwd, "errp-coadaption", "data"),
        os.path.join(cwd, "Code", "errp-coadaption", "data"),
        os.path.join(cwd, "..", "errp-coadaption", "data"),
    ]
    for p in candidates:
        n = os.path.normpath(p)
        if os.path.isdir(n):
            return n
    return os.path.normpath(default)


def _resolve_output_root(default="/kaggle/working/results_v2"):
    return default if os.path.isdir("/kaggle/working") else os.path.normpath(
        os.path.join(os.getcwd(), "Results", "results_v2"))


class Config:
    # ── Paths ──────────────────────────────────────────────────────────
    DATASET_ROOT  = _resolve_dataset_root()
    TRAIN_DIR     = os.path.join(DATASET_ROOT, "train")
    LABELS_FILE   = os.path.join(DATASET_ROOT, "TrainLabels.csv")

    OUTPUT_ROOT    = _resolve_output_root()
    RESULTS_DIR    = OUTPUT_ROOT
    FIGURES_DIR    = os.path.join(OUTPUT_ROOT, "figures")
    METRICS_DIR    = os.path.join(OUTPUT_ROOT, "metrics")
    CSV_DIR        = os.path.join(OUTPUT_ROOT, "csv")
    CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")

    # ── Preprocessing ──────────────────────────────────────────────────
    TMIN             = -0.2      # epoch start relative to feedback onset (s)
    TMAX             =  0.6      # epoch end relative to feedback onset (s)
    BASELINE         = (-0.2, 0.0)
    LOWCUT           =  1.0      # bandpass lower cutoff (Hz)
    HIGHCUT          = 40.0      # bandpass upper cutoff (Hz)
    NOTCH_FREQ       = 50.0      # power-line notch (Hz)
    FILTER_ORDER     =  4        # Butterworth order
    ART_THRESHOLD_UV = 100.0     # artifact rejection threshold µV (peak-to-peak)

    # ── Feature extraction ─────────────────────────────────────────────
    PCA_COMPONENTS = 32

    # ── Experiment ─────────────────────────────────────────────────────
    K_SHOTS         = [5, 10, 20]   # K=50 removed (not few-shot)
    N_SEEDS         = 3
    RANDOM_SEEDS    = [42, 123, 456]
    N_EVAL_EPISODES = 10

    # ── Meta-learning ──────────────────────────────────────────────────
    N_META_ITERATIONS = 2000
    META_BATCH_SIZE   = 4
    N_SUPPORT         = 10
    N_QUERY           = 40
    INNER_LR          = 0.01
    OUTER_LR          = 5e-4
    INNER_STEPS       = 5

    # ── Encoder architecture ───────────────────────────────────────────
    ENCODER_HIDDEN  = 256   # 3-layer MLP
    ENCODER_HIDDEN2 = 128
    ENCODER_OUTPUT  = 64
    DROPOUT         = 0.3

    # ── Subject-conditioned meta-learner ───────────────────────────────
    SUBJECT_EMBED_DIM = 32   # consistent across SubjectEncoder and FiLM

    # ── EEGNet ─────────────────────────────────────────────────────────
    EEGNET_F1      = 8
    EEGNET_D       = 2
    EEGNET_F2      = 16
    EEGNET_DROPOUT = 0.25
    # KERNEL_LENGTH, SFREQ, N_CHANNELS, N_TIMES set at runtime after data load
    SFREQ         = None
    N_CHANNELS    = None
    N_TIMES       = None
    KERNEL_LENGTH = None

    # ── Coadaptation (validation) dataset ──────────────────────────────
    DATA_DIR              = None   # set by the coadaptation loader at runtime
    ERRP_EVENT_LABEL_MAP  = None   # auto-detected (or manual) {event_code: label}
    MANUAL_ERRP_EVENT_LABEL_MAP = None  # set to override auto-detection

    # ── Device ─────────────────────────────────────────────────────────
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_output_dirs(output_root: str) -> None:
    """Create the standard output directory tree under ``output_root``."""
    for sub in ("", "figures", "metrics", "csv", "checkpoints"):
        os.makedirs(os.path.join(output_root, sub) if sub else output_root,
                    exist_ok=True)


@dataclass
class ExperimentConfig:
    """Per-experiment descriptor. The runner applies these onto ``Config`` and
    passes the run-list to the method runners.

    Attributes
    ----------
    name             : experiment key (e.g. "primary").
    dataset          : "inria" | "coadaptation".
    results_dir      : output directory for this experiment's Results.
    pca_components    : PCA dimensionality for the default feature pipeline.
    preprocessing_depth : "full" | "filteronly" | "minimal".
    runs             : list of run specs. Each run spec is a dict:
                         {"method": <registry key>,
                          "save_name": <checkpoint/method name>,
                          "overrides": {<runner-kwarg>: value, ...}}
                       For primary/validation the runs are auto-derived from the
                       selected methods; ablations enumerate explicit variant runs.
    comparisons      : list of (method_a, method_b) for the Wilcoxon tests.
    dataset_root     : optional explicit dataset path (else auto-resolved).
    """
    name: str
    dataset: str = "inria"
    results_dir: str = ""
    pca_components: int = 32
    preprocessing_depth: str = "full"
    runs: List[dict] = field(default_factory=list)
    comparisons: List[Tuple[str, str]] = field(default_factory=list)
    dataset_root: Optional[str] = None
