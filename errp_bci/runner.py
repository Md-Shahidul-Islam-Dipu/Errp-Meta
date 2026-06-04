"""Experiment runner — the single orchestration entry point.

``run_experiment(name, methods=None, seeds=None, k_shots=None)`` reproduces what
the six legacy execution cells did, but driven by config:

    1. resolve the experiment preset + output directory,
    2. for each (preprocessing_depth, pca_components) needed by the run-list,
       load the dataset and build LOSO PCA splits (cached, built once),
    3. for each seed, execute each selected run (method + variant kwargs),
       checkpointing per subject and writing per-subject / aggregate CSVs,
    4. run the Wilcoxon comparisons and save the combined results CSV.

Data loading happens lazily and is cached, so selecting a single method
(e.g. ``methods=["EEGNet"]``) does the minimum work.
"""
import os
from typing import Dict, List, Optional

from .config import Config, ensure_output_dirs
from .experiments import get_experiment
from .registry import METHOD_REGISTRY
from .stats import perform_statistical_tests
from .io_results import save_per_subject_metrics, save_aggregate_metrics
from .reproducibility import set_seed
from .data.loaders import load_dataset, build_features_and_loso


def _resolve_results_dir(results_dir: str) -> str:
    if os.path.isabs(results_dir):
        return results_dir
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else \
        os.path.join(os.getcwd(), "Results")
    return os.path.join(base, results_dir)


def _invoke(entry: dict, save_name: str, ctx: dict, k_shots: List[int],
            seed: int, output_dir: str, device: str, runner_kwargs: dict):
    """Call a method runner according to its registry calling convention."""
    fn, kind = entry["fn"], entry["kind"]
    kw = {"seed": seed, "output_dir": output_dir}
    if entry["has_device"]:
        kw["device"] = device
    if entry["has_method_name"]:
        kw["method_name"] = save_name
    kw.update(entry["fixed"])
    kw.update(runner_kwargs)

    if kind == "loso":
        return fn(ctx["loso"], k_shots, **kw)
    if kind == "features":
        return fn(ctx["features"], k_shots, **kw)
    if kind == "eegnet":
        return fn(ctx["preprocessed"], ctx["loso"], k_shots, **kw)
    if kind == "riemann":
        return fn(ctx["loso"], k_shots, ctx["preprocessed"], **kw)
    if kind == "pretrain":
        return fn(ctx["loso"], k_shots, **kw)  # returns (ft, zs)
    raise ValueError(f"Unknown method kind: {kind!r}")


def run_experiment(name: str,
                   methods: Optional[List[str]] = None,
                   seeds: Optional[List[int]] = None,
                   k_shots: Optional[List[int]] = None,
                   dataset_root: Optional[str] = None,
                   fdr: bool = False) -> Dict[int, Dict[str, dict]]:
    """Run an experiment end-to-end. Returns ``{seed: {save_name: result_dict}}``.

    ``dataset_root`` optionally overrides the auto-resolved dataset path (useful
    when the Kaggle input folder name differs from the defaults).
    """
    cfg = get_experiment(name)
    seeds = seeds or Config.RANDOM_SEEDS
    k_shots = k_shots or Config.K_SHOTS
    device = str(Config.DEVICE)

    # Filter the run-list by the requested method keys (None = all preset runs).
    runs = cfg.runs if methods is None else [r for r in cfg.runs if r["method"] in methods]
    if not runs:
        raise ValueError(
            f"No runs selected for experiment {name!r} with methods={methods}. "
            f"Valid methods: {sorted({r['method'] for r in cfg.runs})}")

    output_dir = _resolve_results_dir(cfg.results_dir)
    ensure_output_dirs(output_dir)
    Config.RESULTS_DIR = output_dir
    Config.CSV_DIR = os.path.join(output_dir, "csv")
    Config.FIGURES_DIR = os.path.join(output_dir, "figures")
    print(f"\n=== Experiment: {name} | dataset={cfg.dataset} | "
          f"output={output_dir} ===")
    print(f"Runs: {[r['save_name'] for r in runs]}")
    print(f"Seeds: {seeds} | K: {k_shots}")

    # Lazily build + cache data contexts keyed by (preprocessing_depth, pca).
    _ppd_cache: Dict[str, dict] = {}
    _ctx_cache: Dict[tuple, dict] = {}

    def get_ctx(depth: str, pca: int) -> dict:
        key = (depth, pca)
        if key in _ctx_cache:
            return _ctx_cache[key]
        if depth not in _ppd_cache:
            run_cfg = get_experiment(name)
            run_cfg.preprocessing_depth = depth
            run_cfg.dataset_root = dataset_root or cfg.dataset_root
            _ppd_cache[depth] = load_dataset(run_cfg)
        ppd = _ppd_cache[depth]
        feats, loso = build_features_and_loso(ppd, pca)
        ctx = {"preprocessed": ppd, "features": feats, "loso": loso}
        _ctx_cache[key] = ctx
        return ctx

    all_results_by_seed: Dict[int, Dict[str, dict]] = {}

    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        set_seed(seed)
        seed_results: Dict[str, dict] = {}

        for run in runs:
            method = run["method"]
            save_name = run["save_name"]
            entry = METHOD_REGISTRY[method]
            ctx = get_ctx(run.get("preprocessing_depth", cfg.preprocessing_depth),
                          run.get("pca_components", cfg.pca_components))
            print(f"  -> {save_name} ({method}) ...")

            result = _invoke(entry, save_name, ctx, k_shots, seed, output_dir,
                             device, run.get("runner_kwargs", {}))

            if entry["kind"] == "pretrain":
                ft, zs = result
                for nm, res in (("Pretrain-FT", ft), ("Pretrain-ZeroShot", zs)):
                    seed_results[nm] = res
                    save_per_subject_metrics(res, nm, seed, output_dir)
                    save_aggregate_metrics(res, nm, seed, k_shots, output_dir)
            elif result is None:
                print(f"     (skipped — runner returned None, e.g. pyriemann missing)")
            else:
                seed_results[save_name] = result
                save_per_subject_metrics(result, save_name, seed, output_dir)
                save_aggregate_metrics(result, save_name, seed, k_shots, output_dir)
            print(f"     {save_name} done")

        active = [(a, b) for (a, b) in cfg.comparisons
                  if a in seed_results and b in seed_results]
        if active:
            stats_df = perform_statistical_tests(
                seed_results, k_shots, active, output_dir, seed, fdr=fdr)
            cols = ['method1', 'method2', 'k', 'mean1', 'mean2',
                    'p_value', 'significant', 'effect_r', 'effect_interp']
            print("\nStatistical tests:")
            print(stats_df[[c for c in cols if c in stats_df.columns]].to_string(index=False))

        all_results_by_seed[seed] = seed_results

    # Combined cross-seed CSV (best-effort; reporting handles missing pieces).
    try:
        from .reporting import build_combined_results_csv
        build_combined_results_csv(all_results_by_seed, k_shots,
                                   output_dir=output_dir)
    except Exception as exc:  # pragma: no cover - reporting is non-critical
        print(f"(combined CSV skipped: {exc})")

    print("\n✓ Experiment complete.")
    return all_results_by_seed
