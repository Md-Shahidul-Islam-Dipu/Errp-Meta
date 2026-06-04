# Running experiments on Kaggle

Everything runs from **`ErrP_Orchestrator.ipynb`** backed by the **`errp_bci/`**
package. You pick one experiment and which models to run, then execute.

## One-time setup

1. **Upload `errp_bci/` as a Kaggle Dataset.**
   - Zip the `errp_bci/` folder (keep the folder itself inside the zip).
   - Kaggle → *Datasets* → *New Dataset* → upload the zip → Create.
   - Re-upload (new version) whenever you change a module.
2. **Create/open the orchestrator notebook** on Kaggle (upload
   `ErrP_Orchestrator.ipynb`).
3. **Attach inputs** (notebook → *Add Input*):
   - the `errp_bci` dataset you just uploaded,
   - the **INRIA BCI Challenge** dataset (for `primary` + ablations),
   - the **ErrP-Coadaptation** dataset (for `validation`).
4. Enable **GPU** in the notebook settings (meta-learners train much faster).

The setup cell finds `errp_bci` automatically by scanning `/kaggle/input/*/`.

## Choosing what to run (cell 2)

```python
EXPERIMENT = "primary"        # primary | validation | ablation_classweighting |
                              #   ablation_nofilm | ablation_feature | ablation_preproc
METHODS    = None             # None = all; or e.g. ["EEGNet"], ["Full-MAML", "Reptile"]
SEEDS      = [42, 123, 456]   # [42] for a quick smoke test
K_SHOTS    = [5, 10, 20]
DATASET_ROOT = None           # set if your Kaggle input path differs from defaults
FDR          = False          # True -> add BH-FDR corrected p-values to the stats CSV
```

- `errp_bci.list_methods(EXPERIMENT)` prints the valid model keys for an experiment.
- Selecting a subset only does the necessary work (data is loaded lazily/cached).

### Validation event map
If the ErrP event auto-detection on the coadaptation dataset misfires, force it
before running:
```python
from errp_bci.config import Config
Config.MANUAL_ERRP_EVENT_LABEL_MAP = {33035: 1, 33036: 0}   # {event_code: label}
```

## Outputs & resuming

Results are written under `/kaggle/working/<experiment-folder>/seed_<seed>/`:
per-subject JSON checkpoints (`fold_checkpoints/`), per-subject and aggregate
CSVs, and `statistical_tests.csv`. Runs are **crash-safe and resumable** — if you
attach a previous `Results/<experiment>/` (or keep working dir), completed folds
are loaded from checkpoints instead of recomputed.

## Recommended verification on Kaggle

1. **Smoke test:** `EXPERIMENT="primary", METHODS=["EEGNet"], SEEDS=[42]` — fast.
2. **Equivalence:** attach existing `Results/Primery/`, run `EXPERIMENT="primary",
   SEEDS=[42]` (all methods) — checkpoint resume should reproduce saved aggregates.
3. **Full runs:** all six experiments across `SEEDS=[42, 123, 456]`.

See `CHANGES.md` for the bug fixes applied during the refactor.
