#!/usr/bin/env python3
"""Write notebooks/01_gennet_in_one_hour.ipynb (self-contained Colab practical)."""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_gennet_in_one_hour.ipynb"

CELLS = []


def md(source: str) -> None:
    CELLS.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip("\n").split("\n")],
        }
    )


def code(source: str) -> None:
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip("\n").split("\n")],
        }
    )


md(
    """# GenNet in one hour

**Neurepiomics 2026** · Genetic and Multiomic Analyses · teaching practical

This notebook is a **Colab-CPU** walk through the GenNet idea:

1. Biology decides which SNP may talk to which gene and pathway.
2. A small network is trained to predict a simulated “high WMH burden” label.
3. We read **prediction**, **gene importance**, and **SNP–SNP interaction**.

It is **not** a GWAS, not UK Biobank, and not the full GenNet CLI. Gene names (APOE, COL4A1, …) are a story scaffold. Allele frequencies and effects are invented so the hour can finish.

**Runtime:** `Runtime → Change runtime type → CPU` (GPU not needed). Then `Runtime → Run all`.

Links: [GenNet paper](https://www.nature.com/articles/s42003-021-02622-z) · [GitHub](https://github.com/ArnovanHilten/GenNet) · [ALIEN](https://www.roshchupkin.org/alien/) · [A-to-Z Colab](https://colab.research.google.com/github/ArnovanHilten/GenNet/blob/master/examples/A_to_Z/GenNet_A_to_Z.ipynb)
"""
)

md(
    """## 0. Imports

Stock Colab already has TensorFlow, scikit-learn, pandas, and matplotlib. No GenNet pip install.
"""
)

code(
    """import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

tf.get_logger().setLevel("ERROR")
print("tensorflow", tf.__version__)
rng_global = np.random.default_rng(7)"""
)

md(
    """## 1. What GenNet is configuring

A fully connected net would let every SNP influence every hidden unit. GenNet forbids that. You supply a **topology**: each row is one allowed path.

In the real tool those paths usually come from Annovar (SNP→gene) and KEGG or GTEx (gene→pathway). Here we write the same table by hand for 24 named genes and four pathways.

The CLI then wants three files:

| File | Role |
|------|------|
| `genotype.h5` | people × SNPs |
| `subjects.csv` | id, label, genotype row, train/val/test (`set` = 1/2/3) |
| `topology.csv` | allowed connections |

We keep the matrices in memory. The tables are printed so you can see what you would write on disk.
"""
)

md(
    """## 2. Simulate a tiny SVD-flavored cohort

1,600 people, 24 genes × 8 SNPs = 192 variants. Independent genotypes (no LD), Hardy–Weinberg draws.

Pathways:

- `lipid_endocytosis`: APOE, ABCA7, CLU, SORL1
- `immune`: TREM2, CD33, CR1, MS4A6A
- `vascular_matrix`: COL4A1, NOTCH3, FOXF2, HTRA1
- `background`: housekeeping names used as decoys

The **label is simulated**. Do not treat the later gene ranking as a WMH discovery.
"""
)

code(
    """PATHWAYS = {
    "lipid_endocytosis": ("APOE", "ABCA7", "CLU", "SORL1"),
    "immune": ("TREM2", "CD33", "CR1", "MS4A6A"),
    "vascular_matrix": ("COL4A1", "NOTCH3", "FOXF2", "HTRA1"),
    "background": (
        "GAPDH", "ACTB", "RPLP0", "B2M", "ALB", "TUBB",
        "PPIA", "HPRT1", "YWHAZ", "UBC", "RPS18", "EEF1A1",
    ),
}
N_SNPS_PER_GENE = 8
N_SAMPLES = 1600
SEED = 7


def simulate_cohort(n_samples=N_SAMPLES, n_snps_per_gene=N_SNPS_PER_GENE, seed=SEED):
    rng = np.random.default_rng(seed)
    gene_names, gene_to_pathway = [], {}
    pathway_names = list(PATHWAYS)
    for pw, genes in PATHWAYS.items():
        for g in genes:
            gene_names.append(g)
            gene_to_pathway[g] = pw
    n_genes, n_pathways = len(gene_names), len(pathway_names)
    n_snps = n_genes * n_snps_per_gene

    maf = rng.uniform(0.08, 0.42, size=n_snps)
    p = maf
    X = np.empty((n_samples, n_snps), dtype=np.float32)
    for j in range(n_snps):
        pr = [(1 - p[j]) ** 2, 2 * p[j] * (1 - p[j]), p[j] ** 2]
        X[:, j] = rng.choice([0.0, 1.0, 2.0], size=n_samples, p=pr)

    snp_names, snp_to_gene = [], np.empty(n_snps, dtype=np.int32)
    gene_to_pw = np.array([pathway_names.index(gene_to_pathway[g]) for g in gene_names])
    rows = []
    for g_i, gene in enumerate(gene_names):
        pw_i = int(gene_to_pw[g_i])
        for k in range(n_snps_per_gene):
            s = g_i * n_snps_per_gene + k
            name = f"{gene}_s{k}"
            snp_names.append(name)
            snp_to_gene[s] = g_i
            rows.append({
                "chr": 19 if gene == "APOE" else (13 if gene == "COL4A1" else (g_i % 22) + 1),
                "layer0_node": s, "layer0_name": name,
                "layer1_node": g_i, "layer1_name": gene,
                "layer2_node": pw_i, "layer2_name": pathway_names[pw_i],
            })
    topology = pd.DataFrame(rows)

    apoe = gene_names.index("APOE")
    col4 = gene_names.index("COL4A1")
    a = apoe * n_snps_per_gene
    b = a + 1
    col4_idx = np.arange(col4 * n_snps_per_gene, (col4 + 1) * n_snps_per_gene)

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-6)

    logit = (
        1.55 * z(X[:, a])
        + 0.95 * z(X[:, b])
        + 1.15 * z(X[:, a]) * z(X[:, b])
        + 0.55 * z(X[:, col4_idx].mean(axis=1))
    )
    logit = logit - logit.mean()
    y = rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(np.float32)

    order = rng.permutation(n_samples)
    n_train, n_val = int(0.70 * n_samples), int(0.15 * n_samples)
    sets = np.empty(n_samples, dtype=np.int8)
    sets[order[:n_train]] = 1
    sets[order[n_train:n_train + n_val]] = 2
    sets[order[n_train + n_val:]] = 3

    subjects = pd.DataFrame({
        "patient_id": [f"s{i:04d}" for i in range(n_samples)],
        "labels": y.astype(int),
        "genotype_row": np.arange(n_samples),
        "set": sets,
    })
    snp_gene_mask = np.zeros((n_snps, n_genes), np.float32)
    snp_gene_mask[np.arange(n_snps), snp_to_gene] = 1
    gene_pathway_mask = np.zeros((n_genes, n_pathways), np.float32)
    gene_pathway_mask[np.arange(n_genes), gene_to_pw] = 1

    return {
        "X": X, "y": y, "sets": sets,
        "snp_names": snp_names, "gene_names": gene_names, "pathway_names": pathway_names,
        "snp_to_gene": snp_to_gene, "gene_to_pathway": gene_to_pw,
        "topology": topology, "subjects": subjects,
        "snp_gene_mask": snp_gene_mask, "gene_pathway_mask": gene_pathway_mask,
        "interaction": (snp_names[a], snp_names[b]),
    }


def split(bundle):
    s, X, y = bundle["sets"], bundle["X"], bundle["y"]
    return X[s == 1], y[s == 1], X[s == 2], y[s == 2], X[s == 3], y[s == 3]


bundle = simulate_cohort()
print(f\"people={bundle['X'].shape[0]}  SNPs={bundle['X'].shape[1]}  prevalence={bundle['y'].mean():.2f}\")
print("split counts", {k: int((bundle['sets'] == k).sum()) for k in (1, 2, 3)})
bundle["topology"].groupby(["layer2_name", "layer1_name"]).size().head(12)"""
)

md(
    """### Look at the topology the way GenNet would
"""
)

code(
    """topo = bundle["topology"]
print("APOE rows (SNP → gene → pathway):")
display(topo[topo.layer1_name == "APOE"][["layer0_name", "layer1_name", "layer2_name"]])
print("How many SNPs per pathway?")
display(topo.groupby("layer2_name")["layer0_name"].nunique().to_frame("n_snps"))
print("subjects.csv head — set 1=train, 2=val, 3=test")
display(bundle["subjects"].head())"""
)

md(
    """## 3. A teaching replica of the directed layer

Published GenNet uses `LocallyDirected1D`: a sparse mask times a weight matrix. That class is tied to TensorFlow 2.11 internals, which is why a full install is a bad idea in a 60-minute Colab.

The layer below is the same **scientific** object at this scale: `output = activation( X @ (W ⊙ M) + b )`. A 1 in `M` is an allowed biological edge. L1 on `W` pushes unused edges toward zero.
"""
)

code(
    """class DirectedLayer(tf.keras.layers.Layer):
    def __init__(self, mask, activation="tanh", l1=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.mask_np = np.asarray(mask, dtype=np.float32)
        self.activation = tf.keras.activations.get(activation)
        self.l1 = float(l1)

    def build(self, input_shape):
        n_in, n_out = self.mask_np.shape
        self.mask = tf.constant(self.mask_np, dtype=self.dtype)
        self.kernel = self.add_weight(
            "kernel", shape=(n_in, n_out),
            initializer="glorot_uniform",
            regularizer=tf.keras.regularizers.l1(self.l1),
        )
        self.bias = self.add_weight("bias", shape=(n_out,), initializer="zeros")

    def call(self, x):
        return self.activation(tf.matmul(x, self.kernel * self.mask) + self.bias)

    def directed_weights(self):
        return self.kernel.numpy() * self.mask_np


def build_gennet(bundle, l1=5e-4):
    n_snps = bundle["X"].shape[1]
    inp = tf.keras.Input((n_snps,), name="genotype")
    genes = DirectedLayer(bundle["snp_gene_mask"], l1=l1, name="gene_layer")(inp)
    pathways = DirectedLayer(bundle["gene_pathway_mask"], l1=l1, name="pathway_layer")(genes)
    out = tf.keras.layers.Dense(
        1, activation="sigmoid",
        kernel_regularizer=tf.keras.regularizers.l1(l1),
        name="output",
    )(pathways)
    model = tf.keras.Model(inp, out, name="mini_gennet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


model = build_gennet(bundle)
model.summary()"""
)

md(
    """## 4. Train (CPU, about 30–60 seconds)

Early stopping watches validation AUC. If this cell is slow, wait; do not switch to GPU.
"""
)

code(
    """X_tr, y_tr, X_va, y_va, X_te, y_te = split(bundle)
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=40,
    batch_size=64,
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=8, restore_best_weights=True
    )],
)

def auc(m, X, y):
    return float(roc_auc_score(y, m.predict(X, verbose=0).ravel()))

gennet_auc = {"train": auc(model, X_tr, y_tr), "val": auc(model, X_va, y_va), "test": auc(model, X_te, y_te)}
print("GenNet AUC", {k: round(v, 3) for k, v in gennet_auc.items()})

fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(history.history["auc"], label="train AUC")
ax.plot(history.history["val_auc"], label="val AUC")
ax.set_xlabel("epoch"); ax.set_ylabel("AUC"); ax.legend(); ax.set_ylim(0.45, 1.0)
ax.set_title("Learning curves")
plt.show()"""
)

md(
    """## 5. Prediction: GenNet vs L1 logistic regression

Lasso sees 192 SNPs with no gene names. GenNet is forced to pool SNPs inside genes. On this planted trait both should beat chance; they answer different questions.
"""
)

code(
    """lasso = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(penalty="l1", solver="saga", C=0.2, max_iter=4000, random_state=0)),
])
lasso.fit(X_tr, y_tr)
lasso_auc = float(roc_auc_score(y_te, lasso.predict_proba(X_te)[:, 1]))
print(f"test AUC   GenNet={gennet_auc['test']:.3f}   L1 logistic={lasso_auc:.3f}")

coef = pd.DataFrame({
    "snp": bundle["snp_names"],
    "abs_coef": np.abs(lasso.named_steps["clf"].coef_.ravel()),
    "gene": [bundle["gene_names"][i] for i in bundle["snp_to_gene"]],
}).sort_values("abs_coef", ascending=False)
print("Top lasso SNPs (no gene layer — the model has to pick variants):")
display(coef.head(10))"""
)

md(
    """## 6. Explainability: weight paths to genes and pathways

GenNet importance here is the product of:

`mean |SNP→gene| × |gene→pathway| × |pathway→output|`

That is the same idea as `python GenNet.py interpret -type get_weight_scores` (and the Manhattan / sunburst plots in the real tool).
"""
)

code(
    """w_sg = model.get_layer("gene_layer").directed_weights()
w_gp = model.get_layer("pathway_layer").directed_weights()
w_out = model.get_layer("output").get_weights()[0].reshape(-1)

rows = []
for g, gene in enumerate(bundle["gene_names"]):
    snps = np.where(bundle["snp_to_gene"] == g)[0]
    p = int(bundle["gene_to_pathway"][g])
    score = float(np.mean(np.abs(w_sg[snps, g])) * abs(w_gp[g, p]) * abs(w_out[p]))
    rows.append({"gene": gene, "pathway": bundle["pathway_names"][p], "importance": score})
gene_imp = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
gene_imp["rank"] = np.arange(1, len(gene_imp) + 1)
display(gene_imp.head(10))

fig, ax = plt.subplots(figsize=(7, 4))
top = gene_imp.head(12).iloc[::-1]
ax.barh(top["gene"], top["importance"])
ax.set_xlabel("weight-path importance")
ax.set_title("Which genes did the directed net use?")
plt.show()

pw_rows = []
for p, pw in enumerate(bundle["pathway_names"]):
    genes = np.where(bundle["gene_to_pathway"] == p)[0]
    pw_rows.append({
        "pathway": pw,
        "importance": float(np.mean(np.abs(w_gp[genes, p])) * abs(w_out[p])),
    })
display(pd.DataFrame(pw_rows).sort_values("importance", ascending=False))"""
)

md(
    """## 7. Interaction: a readable NID

[NID](https://arxiv.org/abs/1705.04977) (Tsang et al.) looks for features that share a hidden unit with large incoming weights. GenNet applies that inside each gene:

`strength(i, j) = min(|w_i|, |w_j|) × |w_gene→pathway| × |w_pathway→out|`

Full CLI: `python GenNet.py interpret -type NID`. DFIM and PathExplain are the heavier options; they will not finish in this hour on real data.
"""
)

code(
    """nid_rows = []
for g, gene in enumerate(bundle["gene_names"]):
    snps = np.where(bundle["snp_to_gene"] == g)[0]
    p = int(bundle["gene_to_pathway"][g])
    later = abs(w_gp[g, p]) * abs(w_out[p])
    mags = np.abs(w_sg[snps, g])
    order = np.argsort(-mags)[:8]
    for ii in range(len(order)):
        for jj in range(ii + 1, len(order)):
            i, j = int(snps[order[ii]]), int(snps[order[jj]])
            nid_rows.append({
                "gene": gene,
                "snp_i": bundle["snp_names"][i],
                "snp_j": bundle["snp_names"][j],
                "strength": float(min(mags[order[ii]], mags[order[jj]]) * later),
            })
nid = pd.DataFrame(nid_rows).sort_values("strength", ascending=False).reset_index(drop=True)
display(nid.head(12))
print("Planted pair (look after you have stared at the table):", bundle["interaction"])"""
)

md(
    """## 8. What you should have recovered

The simulator planted:

1. Strong additive **APOE** (`APOE_s0`, `APOE_s1`).
2. A **multiplicative interaction** of those two SNPs (NID target).
3. Weaker additive **COL4A1**.
4. Noise everywhere else.

If APOE is absent from the top 5, re-run the training cell. The seed is fixed, so a second run should look similar.

This is the point of the exercise: on a trait whose signal **really does** sit in annotated genes, a directed net can name the gene and surface the pair. That is not a claim about WMH in CHARGE.
"""
)

code(
    """top5 = set(gene_imp.head(5)["gene"])
pair = set(bundle["interaction"])
pair_rank = next(
    (i + 1 for i, (a, b) in enumerate(zip(nid.snp_i, nid.snp_j)) if set([a, b]) == pair),
    None,
)
print("APOE in top 5 genes:", "APOE" in top5, "→", list(gene_imp.head(5)["gene"]))
print("COL4A1 rank:", int(gene_imp.set_index("gene").loc["COL4A1", "rank"]))
print("Planted interaction rank in NID table:", pair_rank)"""
)

md(
    """## 9. How this maps to the real GenNet CLI

After today, on a cluster or the [A-to-Z Colab](https://colab.research.google.com/github/ArnovanHilten/GenNet/blob/master/examples/A_to_Z/GenNet_A_to_Z.ipynb):

```bash
python GenNet.py convert  -g ./plink/ -study_name mystudy -o ./processed_data/
python GenNet.py topology -type create_gene_network -path ./processed_data/ -study_name mystudy
python GenNet.py train    -path ./run/ -ID 17 -L1 0.01 -epochs 100
python GenNet.py plot     -ID 17 -type manhattan_relative_importance
python GenNet.py interpret -type get_weight_scores -resultpath results/GenNet_experiment_17_/
python GenNet.py interpret -type NID -resultpath results/GenNet_experiment_17_/
```

Useful knobs: `-L1`, `-L1_act`, `-problem_type regression`, `-filters`, `-onehot`, `-hidden_activation`.

**ALIEN** ([roshchupkin.org/alien](https://www.roshchupkin.org/alien/)) is the research programme around this code — other topologies, multi-omics, cohorts — not a second pip package.

**Limits to take home**

- Annotation-only SNPs miss much regulatory signal.
- A modern PRS will usually predict a highly polygenic trait better.
- Attribution is a hypothesis. It is not a diagnosis.
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "cells": CELLS,
}

# strip extra trailing newlines on last line of each cell (nbformat style: last line often has no extra)
for cell in nb["cells"]:
    if cell["source"] and cell["source"][-1].endswith("\n"):
        cell["source"][-1] = cell["source"][-1][:-1] if cell["source"][-1] != "\n" else cell["source"][-1]

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {NB_PATH} ({len(CELLS)} cells)")
