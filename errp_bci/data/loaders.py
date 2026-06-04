"""Dataset entry points (hand-written glue around the verbatim loader modules).

These functions replace the notebooks' module-level "execute" tails: they run the
INRIA or Coadaptation loading pipeline, set the runtime signal fields on
``Config`` (SFREQ / N_CHANNELS / N_TIMES / KERNEL_LENGTH), and build the LOSO PCA
splits. The two dataset pipelines are kept entirely separate (``inria`` vs
``coadaptation`` modules) — they share no loader code, only the output contract:
a ``preprocessed_subjects_data`` dict ``{sid: {'epochs','labels','ch_names',
'times','sfreq', ...}}``.
"""
import os
from typing import Dict, Tuple

from ..progress import tqdm

from ..config import Config, ExperimentConfig
from . import inria, coadaptation, features

_DEPTH_FUNCS = {
    "full": "process_subject_data",                       # = primary pipeline (verbatim)
    "filter_only": "process_subject_data_filter_only",
    "artifact_baseline_only": "process_subject_data_artifact_baseline_only",
}


def _set_runtime_signal_fields(ppd: Dict[str, Dict]) -> None:
    if not ppd:
        raise RuntimeError("No subjects were successfully preprocessed.")
    first = next(iter(ppd.values()))
    Config.SFREQ = first["sfreq"]
    Config.N_CHANNELS = len(first["ch_names"])
    Config.N_TIMES = first["epochs"].shape[2]
    Config.KERNEL_LENGTH = int(Config.SFREQ // 2)
    print(f"Signal: sfreq={Config.SFREQ:.1f} Hz | n_ch={Config.N_CHANNELS} | "
          f"n_times={Config.N_TIMES} | kernel_length={Config.KERNEL_LENGTH}")


def load_inria(cfg: ExperimentConfig) -> Dict[str, Dict]:
    """Load + preprocess the INRIA NER 2015 dataset under the configured depth."""
    root = cfg.dataset_root or Config.DATASET_ROOT
    train_dir = os.path.join(root, "train")
    labels_file = os.path.join(root, "TrainLabels.csv")
    Config.DATASET_ROOT, Config.TRAIN_DIR, Config.LABELS_FILE = root, train_dir, labels_file

    train_index = inria.index_dataset(train_dir)
    labels_dict = inria.load_labels(labels_file)
    mapping = inria.build_dataset_mapping(train_index, labels_dict)
    print(f"INRIA subjects found: {len(mapping)}")

    proc = getattr(inria, _DEPTH_FUNCS[cfg.preprocessing_depth])
    ppd: Dict[str, Dict] = {}
    for sid in tqdm(sorted(mapping), desc=f"Preprocess INRIA ({cfg.preprocessing_depth})"):
        r = proc(sid, mapping[sid])
        if r is not None:
            ppd[sid] = r
    _set_runtime_signal_fields(ppd)
    return ppd


def load_coadaptation(cfg: ExperimentConfig) -> Dict[str, Dict]:
    """Load + preprocess the ErrP-Coadaptation dataset (EEGLAB .set + ErrP event
    auto-detection). Honors ``Config.MANUAL_ERRP_EVENT_LABEL_MAP`` if set."""
    root = cfg.dataset_root or coadaptation._resolve_coadaptation_dir()
    Config.DATA_DIR = root
    print(f"Coadaptation data dir: {root} (exists={os.path.isdir(root)})")

    manual = Config.MANUAL_ERRP_EVENT_LABEL_MAP
    if manual:
        Config.ERRP_EVENT_LABEL_MAP = dict(manual)
        print(f"Using MANUAL ERRP_EVENT_LABEL_MAP = {Config.ERRP_EVENT_LABEL_MAP}")
    else:
        Config.ERRP_EVENT_LABEL_MAP = coadaptation.autodetect_event_map(root)

    train_index = coadaptation.index_coadaptation_dataset(root)
    mapping = coadaptation.build_coadaptation_mapping(train_index, Config.ERRP_EVENT_LABEL_MAP)
    if len(mapping) < 5:
        raise RuntimeError(
            f"Only {len(mapping)} coadaptation subjects usable — check the "
            "auto-detected ERRP_EVENT_LABEL_MAP or set "
            "Config.MANUAL_ERRP_EVENT_LABEL_MAP before running.")

    ppd: Dict[str, Dict] = {}
    for sid in tqdm(sorted(mapping), desc="Preprocess Coadaptation"):
        r = coadaptation.process_subject_data(sid, mapping[sid])
        if r is not None:
            ppd[sid] = r
    _set_runtime_signal_fields(ppd)
    return ppd


def load_dataset(cfg: ExperimentConfig) -> Dict[str, Dict]:
    """Dispatch to the correct dataset loader based on ``cfg.dataset``."""
    if cfg.dataset == "inria":
        return load_inria(cfg)
    if cfg.dataset == "coadaptation":
        return load_coadaptation(cfg)
    raise ValueError(f"Unknown dataset: {cfg.dataset!r}")


def build_features_and_loso(ppd: Dict[str, Dict],
                            pca_components: int) -> Tuple[Dict, Dict]:
    """Extract temporal features and build the LOSO-isolated PCA splits."""
    subjects_features = features.extract_features_all_subjects(ppd)
    loso_splits = features.apply_pca_loso(subjects_features, pca_components)
    return subjects_features, loso_splits
