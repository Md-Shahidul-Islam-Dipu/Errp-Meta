# Meta-Learning for Rapid Personalization of ErrP-Driven BCIs

A cross-dataset benchmark of few-shot personalization methods for **error-related
potential (ErrP)** brain–computer interfaces. We compare gradient- and metric-based
meta-learners, a FiLM subject-conditioned encoder, and classical/transfer baselines
under one leave-one-subject-out (LOSO) protocol on two independent ErrP datasets,
asking how few calibration trials a new user needs.

This repository accompanies the paper (see [`paper/main.tex`](paper/main.tex)). It
contains the full pipeline as an importable Python package plus a single
orchestrator notebook, and the saved per-fold results needed to regenerate every
table and figure.

## Repository layout

```
errp_bci/              # the pipeline as a Python package (data, models, runner, stats)
ErrP_Orchestrator.ipynb# one notebook: pick an experiment + models, run
Generate_Figures.ipynb # regenerates all paper figures from Results/
Results/               # per-fold checkpoints + per-subject / aggregate CSVs (3 seeds)
figures/               # generated figures (PDF + PNG)
legacy_notebooks/      # the original six monolithic notebooks (reference only)
paper/                 # LaTeX source, bibliography, figures
README_orchestrator.md # how to run experiments on Kaggle
CHANGES.md             # package/refactor changelog
requirements.txt
```

## Datasets (not included — download separately)

The raw EEG is not redistributed here. Obtain it from the original sources:

- **INRIA BCI Challenge (NER 2015)** — P300-speller feedback ErrP (primary).
  Available via Kaggle: *"BCI Challenge @ NER 2015"*.
- **ErrP-Coadaptation** (Ehrlich & Cheng, 2018) — observation ErrP (validation).

## Install

```bash
pip install -r requirements.txt
```

Everything except `pyriemann` is preinstalled on Kaggle's GPU image; the
orchestrator installs `pyriemann` in its first cell.

## Run an experiment

The intended workflow is on Kaggle (GPU) — see
[`README_orchestrator.md`](README_orchestrator.md). In short, open
`ErrP_Orchestrator.ipynb`, attach the `errp_bci` package and the datasets, and edit
one cell:

```python
EXPERIMENT = "primary"   # primary | validation | ablation_classweighting |
                         #   ablation_nofilm | ablation_feature | ablation_preproc
METHODS    = None        # None = all; or e.g. ["EEGNet"], ["Full-MAML", "Reptile"]
SEEDS      = [42, 123, 456]
K_SHOTS    = [5, 10, 20]
```

Methods: `Supervised`, `Pretrain` (FT + ZeroShot transfer controls), `Full-MAML`,
`MAML-ANIL`, `Reptile`, `SubjectConditioned` (FiLM), `Prototypical`, `Matching`,
`Riemannian`, `CovarianceAlignment`, `EEGNet`.

Runs are checkpointed per subject under `Results/<experiment>/seed_<seed>/`, so they
resume after interruption.

## Reproduce the tables and figures (no GPU or datasets needed)

The saved `Results/` CSVs are enough to regenerate every paper number locally:

```bash
pip install numpy pandas matplotlib seaborn   # subset of requirements
jupyter nbconvert --to notebook --execute Generate_Figures.ipynb
```

Figures are written to `figures/`. To regenerate the paper's **statistics** (the
tables and every pairwise test — vs Supervised, vs EEGNet, vs Pretrain-FT/ZeroShot):

```python
from errp_bci import analysis        # torch-free; reads only Results/ CSVs
analysis.summary("primary")          # prints table + macro-F1 + Wilcoxon/FDR
analysis.summary("validation", save_dir="stats")   # also writes CSVs
```

Comparisons use paired Wilcoxon signed-rank tests at the **subject level**
(N=16, seeds averaged, exact null, Benjamini–Hochberg FDR; rank-biserial r). Each
experiment's `Results/<dir>/` already ships the precomputed
`statistical_tests_paper.csv`, `summary_table.csv`, and `macro_f1.csv`.

## Citation

```bibtex
@inproceedings{dipu2026errp,
  title     = {Meta-Learning for Rapid Personalization of Error-Related Potential
               Brain-Computer Interfaces: A Cross-Dataset Benchmark},
  author    = {Dipu, Md Shahidul Islam and Sen, Barshon},
  booktitle = {Proc. SPICSCON},
  year      = {2026}
}
```

## License

MIT — see [`LICENSE`](LICENSE). Dataset usage is governed by the respective dataset
licenses.
