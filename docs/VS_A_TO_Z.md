# This workshop vs the original A-to-Z Colab

Original: https://colab.research.google.com/github/ArnovanHilten/GenNet/blob/master/examples/A_to_Z/GenNet_A_to_Z.ipynb

That notebook is the **software path** from PLINK to a trained GenNet. This hour is the **scientific path** you can finish on free Colab in 2026.

## What A-to-Z actually does

1. `git clone` ArnovanHilten/GenNet and `pip install -r requirements_GenNet.txt` (TensorFlow 2.11 pin).
2. Smoke train `examples/example_classification/` (`python GenNet.py train -ID 9999 -epochs 50`). SNP → gene only. They quote AUC ≈ 0.90.
3. `python GenNet.py convert -step all` on bundled PLINK (`examples/A_to_Z/plink/`).
4. `topology -type create_annovar_input`, then **precomputed** Annovar gene output (the Colab does not run `perl annotate_variation.pl`).
5. `topology -type create_gene_network` → `topology.csv` with genes such as RBFOX1, WWOX, PTPRD, FHIT. Still **no pathway layer**.
6. Train again, then `plot -type manhattan_relative_importance`.

It never calls `python GenNet.py interpret` (no NID, DFIM, PathExplain, RLIPP). It never fits a lasso/PRS baseline.

## Bundled examples in the GitHub repo

| Folder | Topology | Use |
|--------|----------|-----|
| `examples/example_classification/` | SNP → gene (`HERC2`, `BRCA2`, `ApoE`, `VEGFA`, …) | Fast CLI smoke test |
| `examples/example_regression/` | SNP → gene → pathway (`Path1`) | Regression flag |
| `examples/example_regression_cov/` | Same plus covariates | Covariate input |
| `examples/A_to_Z/` | Simulated PLINK + Annovar gene map, ~14k people | Convert tutorial |
| `examples/example_plink/` | PLINK toy | Convert only |

`genotype.h5` files are generated, not stored in git. A-to-Z ships PLINK + Annovar text; classification/regression ship `subjects.csv` + `topology.csv` and expect you to already have hdf5 from a prior convert (or the image).

## What this hour does instead

| | A-to-Z Colab | Neurepiomics hour |
|--|----------------|-------------------|
| Install | Full GenNet + TF 2.11 | Stock Colab TensorFlow |
| Data | Bundled PLINK / hdf5 | Simulated 1,600 × 192, planted APOE × COL4A1 |
| Topology | SNP → gene (Annovar) | SNP → gene → **pathway** |
| Train | `python GenNet.py train` | Keras `DirectedLayer` (same mask idea as `LocallyDirected1D`) |
| Prediction | Train/val curves, AUC | AUC **and** L1 logistic |
| Plots | Manhattan of SNP weights | Gene bars **and** Manhattan |
| Interactions | Not run | Simplified NID |
| Time on current Colab | Often dies at pip | Tens of seconds |

## Why we did not just run A-to-Z on Wednesday

Current Colab Python/TensorFlow will not cleanly install `requirements_GenNet.txt`. Convert + Annovar is the right lesson for a half-day methods course, not a 60-minute mixed CHARGE room. Homework remains the A-to-Z notebook on a machine with the GenNet conda env.

## What we took from A-to-Z into this hour

- `File → Save a copy in Drive`
- The three-file contract (`genotype`, `subjects.csv`, `topology.csv`)
- Weight-path importance
- A Manhattan of relative SNP importance (same formula as `plot -type manhattan_relative_importance`)
- The exact CLI commands as the way home
