# Instructor notes — GenNet, 60 minutes

**Event:** Neurepiomics 2026, San Antonio  
**Slot:** Wednesday 7 October, Genetic and Multiomic Analyses (you share 11:45–15:00 with Jian / Zhao / Mishra). This document is the **one-hour GenNet module**.  
**Room constraint:** mixed audience, hotel/campus Wi‑Fi, laptops of uneven quality. Colab CPU only.

## Goal

After 60 minutes a participant should be able to:

1. Explain that GenNet is a neural net whose **edges are chosen by biology**, not learned as a fully connected layer.
2. Name the three files the CLI needs: `genotype.h5`, `subjects.csv`, `topology.csv`.
3. Train a tiny model in Colab and read a **gene ranking**, a **Manhattan of SNP weights**, and a **pairwise interaction** table.
4. Say out loud that this does **not** beat a PRS on a highly polygenic trait, and that 1 hour cannot train UK Biobank.

## Before the session

- [ ] Open [`docs/slides.html`](slides.html) full screen; have [`SLIDES.md`](SLIDES.md) as speaker notes.
- [ ] Open https://colab.research.google.com/github/roshchupkin/gennet-neurepiomics-workshop-2026/blob/main/notebooks/01_gennet_in_one_hour.ipynb and **Run all** once on Colab CPU. Keep that tab as the projector backup.
- [ ] Put that Colab link on a slide and in the chat (the README badge is the same URL).
- [ ] Confirm the room can reach `colab.research.google.com`. If not, you present the completed notebook; they follow on paper.
- [ ] Do not promise 1000 Genomes in this hour. Download + QC will eat the practical.

## Minute-by-minute

### 0:00–0:05 — landing

“Open the Colab link (README badge or the URL on the slide). Runtime → Change runtime type → CPU. Run the first two cells. You should see 1600 people and 192 SNPs.”

If Colab is compiling TensorFlow slowly, start the talk anyway.

### 0:05–0:20 — talk (slides 1–8)

Stay at the idea level. The topology table is the one slide to linger on.

**Do say:** PRS is the right tool if the job is *rank people by risk*. GenNet is the right tool if the job is *which annotated genes/pathways the net used*.

**Do not say:** “Deep learning outperforms GWAS.” It does not, here.

### 0:20–0:50 — practical

Walk the room through these checkpoints. If someone is stuck, they can skip to the next markdown header; later cells do not depend on plots.

| Checkpoint | Cell | Success looks like |
|------------|------|--------------------|
| A | Simulate / topology | `APOE` maps to `lipid_endocytosis` |
| B | Train | val AUC clearly above 0.5 (expect ~0.75–0.90) |
| C | Lasso vs GenNet | both decent; lasso hits APOE SNPs, GenNet names the gene |
| D | Gene importance + Manhattan | APOE in the top 5; APOE SNPs spike on the Manhattan |
| E | NID | `APOE_s0`–`APOE_s1` high in the table |

**Planted answers** (reveal after D, not before):

- Main gene: **APOE**
- Secondary gene: **COL4A1**
- Interaction: **APOE_s0 × APOE_s1**

If APOE is not in the top 5 (rare; seed is fixed), re-run the training cell. Do not debug TensorFlow in front of 40 people — switch to your completed tab.

### 0:50–1:00 — wrap

Three sentences:

1. On real data you clone GenNet, convert PLINK/VCF, build topology with Annovar/KEGG/GTEx, then `train` / `interpret`.
2. ALIEN is the longer programme (multi-omics, cohorts, configuration) — not a second package they install today.
3. Questions.

Homework if they ask: A-to-Z Colab, and the Communications Biology paper.

Local dry-run on this workstation (seed 7): test AUC **0.77**, gene ranks **APOE then COL4A1**, NID rank 1 **APOE_s0 × APOE_s1**, ~6 seconds. See [`EXPECTED.md`](EXPECTED.md).

## Failure modes

| What happens | What you do |
|--------------|-------------|
| Colab “connecting” forever | Projector notebook; they watch |
| `import tensorflow` dies | Runtime → Restart; CPU not TPU |
| AUC ~0.5 | They trained 1 epoch or used the wrong split; re-run train cell |
| Someone wants UKB | “That’s a cluster job, not a Colab hour” |
| “Is this diagnostic?” | No. Attribution is a hypothesis. |

## Shared session etiquette

You are one of four faculty in Genetic and Multiomic Analyses. Keep this module to **60 minutes** even if the room is hungry for more. Point ambitious people at the cheat sheet and the full CLI.

## What this is not

- Not a contribution back to `ArnovanHilten/GenNet` (the ComPopBio fork stays independent).
- Not 1000 Genomes. Real public genotypes are too large and too messy for this slot.
- Not ALIEN software. The website is the research map; GenNet is the code that exists.
