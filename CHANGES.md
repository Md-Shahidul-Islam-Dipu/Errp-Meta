# CHANGES — refactor to the `errp_bci` package + bug fixes

This refactor replaces six near-identical notebooks with one Python package
(`errp_bci/`) driven by a single orchestrator notebook (`ErrP_Orchestrator.ipynb`).
The model/architecture/training code was **ported verbatim** from the notebooks
(via AST extraction — see `_build_pkg.py`); only the items below changed.

## Architecture

- **`errp_bci/`** — package. Shared infra (`config`, `reproducibility`, `metrics`,
  `io_results`, `stats`), data loaders (`data/inria.py`, `data/coadaptation.py`,
  `data/features.py`, `data/loaders.py`), models (`models/*.py`), and the
  orchestration layer (`registry.py`, `experiments.py`, `runner.py`, `reporting.py`,
  `verify.py`).
- **`ErrP_Orchestrator.ipynb`** — pick `EXPERIMENT` + `METHODS`, run.
- The two dataset loaders are kept **completely separate** — INRIA (CSV +
  `TrainLabels.csv`) and Coadaptation (EEGLAB `.set` + ErrP event auto-detection)
  share no loader code, only the output contract (`preprocessed_subjects_data`).
- Experiment → results-dir defaults match the existing `Results/` subfolders
  (`Primery`, `Validation`, `Class wighted`, `FiLM vs NoFiLM`,
  `Feature Representation`, plus new `Preprocessing Depth`), and the per-subject
  JSON **checkpoint format is unchanged**, so existing runs resume and reproduce.

## Bug fixes (correctness)

1. **Pretrain-FT fair baseline (paper flaw #1, decision-critical).**
   `models/pretrain_ft.py` is the corrected `run_pretrain_ft_baseline_loso`
   (from the in-progress `_pretrain_src.py`). The notebook cell 23 version moved
   the **entire** pooled training set onto the GPU at once
   (`torch.FloatTensor(tr_X).to(device)`) — OOMs on large folds — and omitted
   `float32` casts. The fixed version keeps training data on CPU and moves each
   batch to the device, casts to `float32`, builds the model on-device, and uses
   `set(supp_idx)` membership. It is wired into `primary`/`validation` (produces
   `Pretrain-FT` and `Pretrain-ZeroShot`) and into the comparisons.

2. **Resumed-checkpoint K-key mismatch.** `io_results.load_subject_checkpoint`
   now normalizes the `k_shots` dict keys back to `int`. JSON stores object keys
   as strings, so a resumed checkpoint returned string K keys while aggregation
   and the Wilcoxon tests look them up with integer K — silently yielding NaN
   aggregates/stats on any resumed run. Fresh runs were unaffected; resume now
   matches a fresh run. Checkpoint **write** format is unchanged.

## Additions (opt-in / non-numeric)

3. **Meta-vs-EEGNet comparisons (paper flaw #2).** `primary`/`validation` now also
   run Wilcoxon `Full-MAML/MAML-ANIL/Reptile/SubjectConditioned vs EEGNet` (and
   `Reptile vs Pretrain-FT`). These are extra rows in `statistical_tests.csv`.

4. **Optional BH-FDR correction (paper flaw #7).** `run_experiment(..., fdr=True)`
   adds `p_value_fdr` / `significant_fdr` columns. Default `fdr=False` keeps the
   stats CSV identical to before.

5. **`run_supervised_baseline_loso` gained a `method_name` param** (default
   `'Supervised'`) so the preprocessing-depth ablation can write distinct
   checkpoints per condition. Default behavior is unchanged.

6. **Kaggle-visible progress (`progress.py`).** `tqdm.auto` renders a Jupyter
   widget that does NOT appear in Kaggle's committed run logs, so long folds
   looked frozen. Every module now imports `tqdm` from `errp_bci.progress`, a
   drop-in that prints periodic flushed text lines to stdout instead, e.g.
   `[Full-MAML S05 meta-train] 500/2000 (25%) 42s eta 126s`. The long inner
   meta-training loops (≈2000 iters/fold for MAML/Reptile/SubjectConditioned,
   500 for ProtoNet/Matching, the EEGNet/Pretrain pretrain loops) are now wrapped
   so you can see progress *within* a fold, not just per-method. Updates are rate-
   limited (every ~250 iters and/or 15s) so the log never floods.

## Equivalence

Apart from fix #2 (which only matters when resuming partial checkpoints), every
method runs the same code path with the same seeding order as the legacy
notebooks. Attaching an existing `Results/<experiment>/` and re-running resumes
from checkpoints and reproduces the saved aggregates.

## Removed / archived

- Legacy notebooks moved to `legacy_notebooks/` (kept for reference).
- Obsolete in-progress scaffolding removed: `_pretrain_src.py`, `_replace_cell23.pl`.
- Build/verify tooling kept at repo root: `_build_pkg.py` (regenerates the
  verbatim modules), `_check_pkg.py` (syntax + undefined-name check),
  `_import_test.py` (stubbed import-graph check).
