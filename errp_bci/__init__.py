"""errp_bci — Meta-learning for rapid personalization of ErrP-driven BCIs.

A single Python package backing one orchestrator notebook. Pick an experiment
(primary / validation / one of four ablations) and which model(s) to run.

Public API
----------
    from errp_bci import run_experiment, list_experiments, list_methods
    results = run_experiment("primary", methods=["EEGNet"], seeds=[42])

    # Reproduce every paper number from saved Results/ CSVs (no torch needed):
    from errp_bci import analysis
    analysis.summary("primary")

Heavy symbols (run_experiment, the registry, Config) are imported lazily so that
``from errp_bci import analysis`` works in a torch-free environment — useful for
regenerating the paper's tables and statistics locally without a GPU/PyTorch.
See ``errp_bci.experiments`` for presets and ``errp_bci.registry`` for methods.
"""
import importlib

__all__ = [
    "Config", "ExperimentConfig", "EXPERIMENTS", "get_experiment",
    "list_experiments", "METHOD_REGISTRY", "list_methods", "run_experiment",
    "analysis",
]

# attribute -> (submodule, symbol); imported on first access (keeps `import
# errp_bci` and `from errp_bci import analysis` from pulling in torch).
_LAZY = {
    "Config": ("config", "Config"),
    "ExperimentConfig": ("config", "ExperimentConfig"),
    "EXPERIMENTS": ("experiments", "EXPERIMENTS"),
    "get_experiment": ("experiments", "get_experiment"),
    "list_experiments": ("experiments", "list_experiments"),
    "METHOD_REGISTRY": ("registry", "METHOD_REGISTRY"),
    "list_methods": ("registry", "list_methods"),
    "run_experiment": ("runner", "run_experiment"),
}


def __getattr__(name):
    if name == "analysis":
        return importlib.import_module(".analysis", __name__)
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module("." + mod, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
