# GenNet in one hour — Neurepiomics 2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshchupkin/gennet-neurepiomics-workshop-2026/blob/main/notebooks/01_gennet_in_one_hour.ipynb)

**Open the practical:** [Launch in Google Colab](https://colab.research.google.com/github/roshchupkin/gennet-neurepiomics-workshop-2026/blob/main/notebooks/01_gennet_in_one_hour.ipynb)

Hands-on module for **Neurepiomics 2026** (San Antonio, 5–7 October), inside Wednesday’s
optional session:

> **Genetic and Multiomic Analyses** · 11:45–15:00 · Center for Brain Health  
> Xueqiu Jian, Feiyang Zhao, Aniket Mishra, and Gennady Roshchupkin

This repository is a **one-hour GenNet block**: a short talk on how a biologically
structured network is configured, then a Google Colab practical that trains a tiny
model, checks prediction, and looks at gene importance and SNP–SNP interaction.

It is **not** a release of [ALIEN](https://www.roshchupkin.org/alien/). ALIEN is the
research programme around GenNet. What participants run here is a teaching-scale
replica of the GenNet idea, plus pointers to the real CLI.

## Why a teaching replica instead of the full CLI in Colab?

[GenNet](https://github.com/ArnovanHilten/GenNet) pins TensorFlow 2.11. Free Colab
currently ships a much newer TensorFlow, so a full `pip install -r requirements_GenNet.txt`
often fails in the first ten minutes of a workshop. The notebook keeps the same
scientific structure (masked SNP → gene → pathway connections, L1 sparsity, weight-path
importance, NID-style candidate scores plus an additive-only control) with code that runs on stock Colab **CPU**.

After the hour, the [A-to-Z Colab](https://colab.research.google.com/github/ArnovanHilten/GenNet/blob/master/examples/A_to_Z/GenNet_A_to_Z.ipynb)
and `python GenNet.py --help` are the path to real data.

## One-hour timetable

| Min | Block | What happens |
|-----|--------|----------------|
| 0–5 | Open Colab | Click the README badge. Runtime → CPU. Run the Imports and Simulate code cells. |
| 5–20 | Talk | Paper figures, interpretation, NID, ALIEN, then the toy. |
| 20–50 | Practical | Simulate → inspect topology → train → AUC vs lasso → gene ranks → NID candidates → additive-only control. |
| 50–60 | Wrap | How this maps to CHARGE-scale GenNet, caveats, Q&A. |

The presenter uses their own introductory slides; this repository contains the practical and supporting notes.

Instructor notes: [`docs/INSTRUCTOR.md`](docs/INSTRUCTOR.md)  
Cheat sheet: [`docs/CHEATSHEET.md`](docs/CHEATSHEET.md)  
Vs original A-to-Z Colab: [`docs/VS_A_TO_Z.md`](docs/VS_A_TO_Z.md)

## Practical (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshchupkin/gennet-neurepiomics-workshop-2026/blob/main/notebooks/01_gennet_in_one_hour.ipynb)

That badge opens [`notebooks/01_gennet_in_one_hour.ipynb`](notebooks/01_gennet_in_one_hour.ipynb) in Colab from `main`. Participants do not upload a file.

You do **not** need a GPU. The simulated cohort is 1,600 people × 192 SNPs.
Training is tens of seconds on CPU.

**Instructor answer key (participants: inspect the rankings first):**

1. Strong additive **APOE** (two SNPs), loosely analogous to ε4.
2. A **within-APOE interaction** of those two SNPs — a candidate for NID and response-surface checks.
3. Weaker additive **COL4A1** (vascular-matrix pathway).
4. All other genes are noise.

The phenotype is a **simulated binary high WMH burden**. Gene names are a story
scaffold for a CHARGE-style SVD/AD endophenotype. This is not 1000 Genomes, UK
Biobank, or a real GWAS.

## Real GenNet (after the workshop)

```text
git clone https://github.com/ArnovanHilten/GenNet
conda create -n env_GenNet python=3.10.12
conda activate env_GenNet
pip install -r requirements_GenNet.txt

python GenNet.py train -path ./examples/example_classification/ -ID 1
python GenNet.py interpret -type get_weight_scores -resultpath results/GenNet_experiment_1_/
python GenNet.py interpret -type NID -resultpath results/GenNet_experiment_1_/
```

Paper: [Communications Biology 2021](https://www.nature.com/articles/s42003-021-02622-z)  
Site: [ALIEN / GenNet](https://www.roshchupkin.org/alien/)  
Canonical code: [ArnovanHilten/GenNet](https://github.com/ArnovanHilten/GenNet)

## Reproducibility and interpretation

The data seed and model seed are separate (`SEED=7`, `MODEL_SEED=7`). The training cell rebuilds the model each time, and preprocessing uses training participants only. Deterministic operations are enabled in the notebook, but results may differ across TensorFlow versions. A fixed seed is a teaching configuration, not evidence of robustness.

The teaching layer uses dense matrices with 216 allowed hidden edges and 4,704 allocated hidden weights (4,737 total trainable parameters). L1 applies only to allowed edges. The original GenNet uses sparse storage.

NID ranks candidates; it does not establish interaction effects or significance. The notebook now fits an additive-only control and compares known and fitted genotype response surfaces on the logit scale. Even the additive control can score a pair highly and learn spurious non-additivity. Good prediction and a correct pair ranking do not establish recovery of the mechanism.

Optional section 10 explores shuffled topology, initialization and sparsity for ALIEN benchmarking. It is outside the live hour.

**Projector / connection fallback:** [completed notebook with outputs](notebooks/01_gennet_in_one_hour_completed.ipynb). Download it before the session; it includes the computed practical plots and tables. Paper figures in markdown still use online URLs.

## Local checks

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name python3
PYTHONPATH=src python -m pytest -q
python scripts/build_notebook.py
python scripts/execute_notebook.py
# Optional: refresh the completed backup or test the topology extension.
python scripts/execute_notebook.py --output notebooks/01_gennet_in_one_hour_completed.ipynb
python scripts/execute_notebook.py --extensions
# If the environment disallows kernel sockets:
python scripts/execute_notebook.py --in-process
```

GitHub Actions regenerates and executes the self-contained notebook in a fresh kernel, in addition to testing the helper modules. See [expected outputs](docs/EXPECTED.md) for the recorded environment.

## Audience notes

Neurepiomics mixes neurologists, epidemiologists, imaging scientists, and
bioinformaticians. The talk assumes no TensorFlow. The practical assumes people
can click Run in Colab. Critical points to leave in the room:

- GenNet is built for **interpretation under biology**, not for beating a modern PRS.
- Topology is a **scientific choice** (Annovar genes, KEGG, GTEx, your own mask).
- Attribution is a **hypothesis generator**. It is not a diagnosis.
