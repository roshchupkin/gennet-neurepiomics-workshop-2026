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
importance, NID-style interactions) with code that runs on stock Colab **CPU**.

After the hour, the [A-to-Z Colab](https://colab.research.google.com/github/ArnovanHilten/GenNet/blob/master/examples/A_to_Z/GenNet_A_to_Z.ipynb)
and `python GenNet.py --help` are the path to real data.

## One-hour timetable

| Min | Block | What happens |
|-----|--------|----------------|
| 0–5 | Open Colab | Click the README badge. Runtime → CPU. Run the first two cells. |
| 5–23 | Talk | Paper figures, interpretation, NID, ALIEN, then the toy. |
| 20–50 | Practical | Simulate → inspect topology → train → AUC vs lasso → gene ranks → NID. |
| 50–60 | Wrap | How this maps to CHARGE-scale GenNet, caveats, Q&A. |

Instructor notes: [`docs/INSTRUCTOR.md`](docs/INSTRUCTOR.md)  
Slide script: [`docs/SLIDES.md`](docs/SLIDES.md)  
Presenter deck: [`docs/slides.html`](docs/slides.html) (open in a browser; paper figures, offline)  
Cheat sheet: [`docs/CHEATSHEET.md`](docs/CHEATSHEET.md)  
Vs original A-to-Z Colab: [`docs/VS_A_TO_Z.md`](docs/VS_A_TO_Z.md)

## Practical (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshchupkin/gennet-neurepiomics-workshop-2026/blob/main/notebooks/01_gennet_in_one_hour.ipynb)

That badge opens [`notebooks/01_gennet_in_one_hour.ipynb`](notebooks/01_gennet_in_one_hour.ipynb) in Colab from `main`. Participants do not upload a file.

You do **not** need a GPU. The simulated cohort is 1,600 people × 192 SNPs.
Training is tens of seconds on CPU.

**What is planted (do not tell the room until they have ranked genes):**

1. Strong additive **APOE** (two SNPs), loosely analogous to ε4.
2. A **within-APOE interaction** of those two SNPs — the NID target.
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

## Local checks (this workstation)

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_GenNet
export CUDA_VISIBLE_DEVICES=-1
cd ~/work/gennet-neurepiomics-workshop
PYTHONPATH=src pytest tests/test_workshop.py -q
PYTHONPATH=src pytest tests/test_workshop.py -q -m slow
```

## Audience notes

Neurepiomics mixes neurologists, epidemiologists, imaging scientists, and
bioinformaticians. The talk assumes no TensorFlow. The practical assumes people
can click Run in Colab. Critical points to leave in the room:

- GenNet is built for **interpretation under biology**, not for beating a modern PRS.
- Topology is a **scientific choice** (Annovar genes, KEGG, GTEx, your own mask).
- Attribution is a **hypothesis generator**. It is not a diagnosis.
