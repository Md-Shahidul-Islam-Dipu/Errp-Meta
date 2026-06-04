"""Method registry — maps a method key to its runner and calling convention.

Each entry records:
    fn              : the LOSO runner callable.
    kind            : how to pass the data context to ``fn``:
                        "loso"    -> fn(loso_splits, k_shots, ...)
                        "features"-> fn(subjects_features, k_shots, ...)
                        "eegnet"  -> fn(preprocessed, loso_splits, k_shots, ...)
                        "riemann" -> fn(loso_splits, k_shots, preprocessed, ...)   (no device)
                        "pretrain"-> fn(loso_splits, k_shots, ...) returns (ft, zs) pair
    has_device      : whether ``fn`` accepts a ``device`` kwarg.
    has_method_name : whether ``fn`` accepts a ``method_name`` kwarg (controls
                      checkpoint + result naming).
    fixed           : kwargs always passed to ``fn`` (e.g. freeze_encoder_inner).
    produces        : the result/checkpoint name(s) this entry yields by default.

The runner (``errp_bci.runner``) uses this metadata to build the call.
"""
from .models import (supervised, pretrain_ft, maml, subject_conditioned, reptile,
                     prototypical, matching, riemannian, eegnet)


def _entry(fn, kind, has_device=True, has_method_name=True, fixed=None, produces=None):
    return {"fn": fn, "kind": kind, "has_device": has_device,
            "has_method_name": has_method_name, "fixed": fixed or {},
            "produces": produces}


METHOD_REGISTRY = {
    "Supervised": _entry(
        supervised.run_supervised_baseline_loso, "loso",
        produces=["Supervised"]),

    "Pretrain": _entry(
        pretrain_ft.run_pretrain_ft_baseline_loso, "pretrain",
        has_method_name=False, produces=["Pretrain-FT", "Pretrain-ZeroShot"]),

    "Full-MAML": _entry(
        maml.train_maml_loso, "loso",
        fixed={"freeze_encoder_inner": False}, produces=["Full-MAML"]),

    "MAML-ANIL": _entry(
        maml.train_maml_loso, "loso",
        fixed={"freeze_encoder_inner": True}, produces=["MAML-ANIL"]),

    "SubjectConditioned": _entry(
        subject_conditioned.train_subject_conditioned_loso, "loso",
        produces=["SubjectConditioned"]),

    "SubjectConditioned-NoFiLM": _entry(
        subject_conditioned.train_no_film_loso, "loso",
        produces=["SubjectConditioned_NoFiLM"]),

    "Reptile": _entry(
        reptile.train_reptile_loso, "loso", produces=["Reptile"]),

    "Prototypical": _entry(
        prototypical.train_prototypical_loso, "features", produces=["Prototypical"]),

    "Matching": _entry(
        matching.train_matching_loso, "features", produces=["Matching"]),

    "Riemannian": _entry(
        riemannian.run_riemannian_tangent_lda_loso, "riemann",
        has_device=False, produces=["Riemannian"]),

    "CovarianceAlignment": _entry(
        riemannian.run_covariance_alignment_loso, "riemann",
        has_device=False, produces=["CovarianceAlignment"]),

    "EEGNet": _entry(
        eegnet.run_eegnet_loso, "eegnet", produces=["EEGNet"]),
}

# Default method order for primary / validation experiments (mirrors the
# original execution pipeline order).
ALL_METHODS = [
    "Supervised", "Pretrain", "Full-MAML", "MAML-ANIL", "SubjectConditioned",
    "Reptile", "Prototypical", "Matching", "Riemannian", "CovarianceAlignment",
    "EEGNet",
]


def list_methods(experiment: str = None):
    """Return the valid method keys. If ``experiment`` is given, return the
    methods that experiment actually runs (else every registered method)."""
    if experiment is None:
        return list(METHOD_REGISTRY.keys())
    from .experiments import get_experiment
    cfg = get_experiment(experiment)
    seen, out = set(), []
    for run in cfg.runs:
        m = run["method"]
        if m not in seen:
            seen.add(m); out.append(m)
    return out
