"""Experiment presets.

Each preset is an :class:`ExperimentConfig` carrying a *run-list*: the concrete
(method, save_name, runner_kwargs, pca_components, preprocessing_depth) tuples to
execute. Primary/validation enumerate all methods once; the four ablations
enumerate explicit variant runs. Default ``results_dir`` names match the existing
``Results/`` subfolders so prior checkpoints resume.
"""
from typing import List, Optional

from .config import ExperimentConfig
from .registry import ALL_METHODS

# ── Comparisons ─────────────────────────────────────────────────────────────
_PRIMARY_COMPARISONS = [
    ("SubjectConditioned", "Supervised"), ("Full-MAML", "Supervised"),
    ("MAML-ANIL", "Supervised"), ("Reptile", "Supervised"),
    ("Prototypical", "Supervised"), ("Matching", "Supervised"),
    ("SubjectConditioned", "MAML-ANIL"), ("Riemannian", "Supervised"),
    ("CovarianceAlignment", "Supervised"), ("EEGNet", "Supervised"),
    ("Pretrain-FT", "Supervised"), ("Pretrain-FT", "Reptile"),
    ("Pretrain-ZeroShot", "Supervised"),
    # flaw #2: every gradient meta-learner vs the strongest neural baseline.
    ("Full-MAML", "EEGNet"), ("MAML-ANIL", "EEGNet"),
    ("Reptile", "EEGNet"), ("SubjectConditioned", "EEGNet"),
    ("Reptile", "Pretrain-FT"),
]


def _simple_run(method: str, pca: int, depth: str) -> dict:
    return {"method": method, "save_name": method, "runner_kwargs": {},
            "pca_components": pca, "preprocessing_depth": depth}


def _primary_like(name: str, dataset: str, results_dir: str) -> ExperimentConfig:
    cfg = ExperimentConfig(name=name, dataset=dataset, results_dir=results_dir,
                           pca_components=32, preprocessing_depth="full")
    cfg.runs = [_simple_run(m, 32, "full") for m in ALL_METHODS]
    cfg.comparisons = list(_PRIMARY_COMPARISONS)
    return cfg


def _build(name: str) -> ExperimentConfig:
    if name == "primary":
        return _primary_like("primary", "inria", "Primery")

    if name == "validation":
        return _primary_like("validation", "coadaptation", "Validation")

    if name == "ablation_classweighting":
        cfg = ExperimentConfig(name=name, dataset="inria", results_dir="Class wighted")
        cfg.runs = [
            {"method": "Full-MAML", "save_name": "Full-MAML",
             "runner_kwargs": {"use_weighting": True}, "pca_components": 32, "preprocessing_depth": "full"},
            {"method": "Full-MAML", "save_name": "Full-MAML-Uniform",
             "runner_kwargs": {"use_weighting": False}, "pca_components": 32, "preprocessing_depth": "full"},
            {"method": "MAML-ANIL", "save_name": "MAML-ANIL",
             "runner_kwargs": {"use_weighting": True}, "pca_components": 32, "preprocessing_depth": "full"},
            {"method": "MAML-ANIL", "save_name": "MAML-ANIL-Uniform",
             "runner_kwargs": {"use_weighting": False}, "pca_components": 32, "preprocessing_depth": "full"},
        ]
        cfg.comparisons = [("Full-MAML", "Full-MAML-Uniform"),
                           ("MAML-ANIL", "MAML-ANIL-Uniform")]
        return cfg

    if name == "ablation_nofilm":
        cfg = ExperimentConfig(name=name, dataset="inria", results_dir="FiLM vs NoFiLM")
        cfg.runs = [
            _simple_run("SubjectConditioned", 32, "full"),
            {"method": "SubjectConditioned-NoFiLM", "save_name": "SubjectConditioned_NoFiLM",
             "runner_kwargs": {}, "pca_components": 32, "preprocessing_depth": "full"},
        ]
        cfg.comparisons = [("SubjectConditioned", "SubjectConditioned_NoFiLM")]
        return cfg

    if name == "ablation_feature":
        cfg = ExperimentConfig(name=name, dataset="inria", results_dir="Feature Representation")
        cfg.runs = [
            {"method": "Full-MAML", "save_name": "Full-MAML",
             "runner_kwargs": {}, "pca_components": 32, "preprocessing_depth": "full"},
            {"method": "Full-MAML", "save_name": "Full-MAML-Raw",
             "runner_kwargs": {}, "pca_components": 128, "preprocessing_depth": "full"},
            {"method": "SubjectConditioned", "save_name": "SubjectConditioned",
             "runner_kwargs": {}, "pca_components": 32, "preprocessing_depth": "full"},
            {"method": "SubjectConditioned", "save_name": "SubjectConditioned-Raw",
             "runner_kwargs": {}, "pca_components": 128, "preprocessing_depth": "full"},
            {"method": "EEGNet", "save_name": "EEGNet",
             "runner_kwargs": {}, "pca_components": 32, "preprocessing_depth": "full"},
        ]
        cfg.comparisons = [("Full-MAML", "Full-MAML-Raw"),
                           ("SubjectConditioned", "SubjectConditioned-Raw")]
        return cfg

    if name == "ablation_preproc":
        cfg = ExperimentConfig(name=name, dataset="inria", results_dir="Preprocessing Depth")
        depths = [("full", "Full"),
                  ("filter_only", "FilterOnly"),
                  ("artifact_baseline_only", "ArtifactBaselineOnly")]
        runs = []
        for depth, label in depths:
            runs.append({"method": "Supervised", "save_name": f"Supervised-{label}",
                         "runner_kwargs": {}, "pca_components": 32, "preprocessing_depth": depth})
            runs.append({"method": "Full-MAML", "save_name": f"Full-MAML-{label}",
                         "runner_kwargs": {}, "pca_components": 32, "preprocessing_depth": depth})
        cfg.runs = runs
        cfg.comparisons = [
            ("Supervised-Full", "Supervised-FilterOnly"),
            ("Supervised-Full", "Supervised-ArtifactBaselineOnly"),
            ("Full-MAML-Full", "Full-MAML-FilterOnly"),
            ("Full-MAML-Full", "Full-MAML-ArtifactBaselineOnly"),
        ]
        return cfg

    raise KeyError(f"Unknown experiment: {name!r}. Choose from {list(EXPERIMENTS)}.")


EXPERIMENTS = [
    "primary", "validation", "ablation_classweighting", "ablation_nofilm",
    "ablation_feature", "ablation_preproc",
]


def get_experiment(name: str) -> ExperimentConfig:
    """Return a fresh ExperimentConfig preset for ``name``."""
    return _build(name)


def list_experiments() -> List[str]:
    return list(EXPERIMENTS)
