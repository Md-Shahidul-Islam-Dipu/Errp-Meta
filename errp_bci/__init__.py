"""errp_bci — Meta-learning for rapid personalization of ErrP-driven BCIs.

A single Python package backing one orchestrator notebook. Pick an experiment
(primary / validation / one of four ablations) and which model(s) to run.

Public API
----------
    from errp_bci import run_experiment, list_experiments, list_methods
    results = run_experiment("primary", methods=["EEGNet"], seeds=[42])

See ``errp_bci.experiments`` for the available experiment presets and
``errp_bci.registry`` for the available methods.
"""
from .config import Config, ExperimentConfig
from .experiments import EXPERIMENTS, get_experiment, list_experiments
from .registry import METHOD_REGISTRY, list_methods
from .runner import run_experiment

__all__ = [
    "Config",
    "ExperimentConfig",
    "EXPERIMENTS",
    "get_experiment",
    "list_experiments",
    "METHOD_REGISTRY",
    "list_methods",
    "run_experiment",
]
