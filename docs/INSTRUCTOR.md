# Instructor notes — GenNet, 60 minutes

**Event:** Neurepiomics 2026, San Antonio  
**Slot:** Wednesday 7 October, Genetic and Multiomic Analyses (you share 11:45–15:00 with Jian / Zhao / Mishra). This document is the **one-hour GenNet module**.  
**Room constraint:** mixed audience, hotel/campus Wi‑Fi, laptops of uneven quality. Colab CPU only.

## Goal

After 60 minutes a participant should be able to:

1. Explain that GenNet is a neural net whose **edges are chosen by biology**, not learned as a fully connected layer.
2. Name the three files the CLI needs: `genotype.h5`, `subjects.csv`, `topology.csv`.
3. Train a tiny model in Colab and read a **gene ranking**, a **Manhattan of SNP weights**, and an **interaction-candidate** table, then check an additive-only control.
4. Explain that this toy does not establish performance against modern PRS methods or on highly polygenic traits, and that one hour cannot train UK Biobank.

## Before the session

- [ ] Open the introductory slides: [`docs/GenNet_intro.pdf`](GenNet_intro.pdf). The PowerPoint stays local and is not in the repository.
- [ ] Open https://colab.research.google.com/github/roshchupkin/gennet-neurepiomics-workshop-2026/blob/main/notebooks/01_gennet_in_one_hour.ipynb and **Run all** once on Colab CPU. Keep that tab as the projector backup.
- [ ] Download the completed backup notebook for connection failures.
- [ ] Put that Colab link on a slide and in the chat (the README badge is the same URL).
- [ ] Confirm the room can reach `colab.research.google.com`. If not, you present the completed notebook; they follow on paper.
- [ ] Do not promise 1000 Genomes in this hour. Download + QC will eat the practical.

## Minute-by-minute

### 0:00–0:05 — landing

“Open the Colab link (README badge or the URL on the slide). Runtime → Change runtime type → CPU. Run the Imports and Simulate code cells. You should see 1600 people and 192 SNPs.”

If Colab is compiling TensorFlow slowly, start the talk anyway.

### 0:05–0:20 — introductory talk

Use your own presentation to introduce the biological topology, prediction and interpretation. Before starting the notebook, briefly explain the difference between additive SNP effects, non-additivity and NID candidate ranking. The notebook contains the relevant paper figures; sources: [`FIGURES.md`](FIGURES.md).

**Do say:** compare prediction against suitable baselines. GenNet incorporates biological structure during learning; weights and NID scores are candidates for further investigation.

**Do not say:** “Deep learning outperforms GWAS.” It does not, here.

The notebook is longer than the live hour: theory, paper figures, and a take-home CLI live next to the runnable cells. **Run all** still trains in ~40 s. Tell the room to skim while it runs; do not walk every markdown cell.

### 0:20–0:50 — practical

Walk the room through these checkpoints. If someone is stuck, they can skip to the next markdown header; later cells do not depend on plots.

| Checkpoint | Cell | Success looks like |
|------------|------|--------------------|
| A | Simulate / topology | `APOE` maps to `lipid_endocytosis` |
| B | Train | val AUC clearly above 0.5 (expect ~0.75–0.90) |
| C | Lasso vs GenNet | both decent; both can use gene annotation, GenNet uses it during training |
| D | Gene importance + Manhattan | APOE in the top 5; APOE SNPs spike on the Manhattan |
| E | NID candidates | `APOE_s0`–`APOE_s1` high in the table |
| F | Additive-only control | NID can score additive effects; compare known and fitted logit surfaces |

**Planted answers** (reveal after D, not before):

- Main gene: **APOE**
- Secondary gene: **COL4A1**
- Interaction: **APOE_s0 × APOE_s1**

The training cell restarts from model seed 7. If recovery fails, keep the failure visible and switch to the completed backup; do not rerun until a favorable result appears. Rerun downstream cells after any deliberate retraining.

At checkpoint F, ask: “The additive control has no planted interaction. Why does NID still rank this pair highly?” The fixed configuration also produces more fitted non-additivity in the additive control than in the interaction-trained model. This is the teaching point: prediction and candidate ranking do not establish mechanism recovery. One unit per gene limits the toy’s expressiveness.

### 0:50–1:00 — wrap

Three sentences:

1. On real data you clone GenNet, convert PLINK/VCF, build topology with Annovar/KEGG/GTEx, then `train` / `interpret`.
2. ALIEN is the longer programme (multi-omics, cohorts, configuration) — not a second package they install today.
3. Questions.

The notebook repeats one sequence: Concept, Think before running, code, Interpretation. Core blocks are topology, prediction, gene ranking, and the additive-only control. Before the NID table, ask: if two SNPs are strongly additive and do not interact, can they still score highly? After the control, ask which result is prediction, which is a candidate pair, and which examines fitted non-additivity.

Homework if they ask: the take-home section (real CLI checklist), the A-to-Z Colab, and the Communications Biology paper. The ADDITIONAL block is after the hour.

Recorded dry-run (TensorFlow 2.20, data/model seeds 7): test AUC **0.790**, gene ranks **APOE then COL4A1**, candidate pair rank 1. See [`EXPECTED.md`](EXPECTED.md). Download the [completed backup](../notebooks/01_gennet_in_one_hour_completed.ipynb) before class.

## Failure modes

| What happens | What you do |
|--------------|-------------|
| Colab “connecting” forever | Projector notebook; they watch |
| `import tensorflow` dies | Runtime → Restart; CPU not TPU |
| AUC ~0.5 | Check configuration and environment; optimization can fail. Use the completed backup, preserving the failed result |
| Someone wants UKB | “That’s a cluster job, not a Colab hour” |
| “Is this diagnostic?” | No. Attribution is a hypothesis. |

## Shared session etiquette

You are one of four faculty in Genetic and Multiomic Analyses. Keep this module to **60 minutes** even if the room is hungry for more. Point ambitious people at the cheat sheet and the full CLI.

## What this is not

- Not a contribution back to `ArnovanHilten/GenNet` (the ComPopBio fork stays independent).
- Not 1000 Genomes. Real public genotypes are too large and too messy for this slot.
- The student README does not explain [ALIEN](https://www.roshchupkin.org/alien/). If someone asks: ALIEN is the broader research programme; this hour runs the GenNet practical.

Comparison with the original A-to-Z Colab, including why this hour uses a simulated cohort: [`VS_A_TO_Z.md`](VS_A_TO_Z.md).

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

The data seed and model seed are separate (`SEED=7`, `MODEL_SEED=7`). The training cell rebuilds the model each time, and preprocessing uses training participants only. See [expected outputs](EXPECTED.md).
