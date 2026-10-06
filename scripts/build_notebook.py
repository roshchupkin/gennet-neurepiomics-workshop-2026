#!/usr/bin/env python3
"""Write notebooks/01_gennet_in_one_hour.ipynb (self-contained Colab practical)."""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_gennet_in_one_hour.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/roshchupkin/"
    "gennet-neurepiomics-workshop-2026/blob/main/notebooks/01_gennet_in_one_hour.ipynb"
)
FIG = (
    "https://raw.githubusercontent.com/roshchupkin/"
    "gennet-neurepiomics-workshop-2026/main/docs/figures"
)

CELLS = []


def md(source: str, metadata: dict | None = None) -> None:
    CELLS.append(
        {
            "cell_type": "markdown",
            "metadata": metadata or {},
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
    f'<a href="{COLAB_URL}" target="_parent">'
    '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>',
    metadata={"id": "view-in-github", "colab_type": "text"},
)

md(
    f"""# GenNet: from genotype to named biology

**Neurepiomics 2026** · Genetic and Multiomic Analyses · teaching notebook

Neuroepidemiology already **ranks people** well (GWAS, PRS-CS, LDpred). Naming **which annotated genes and pathways** a predictor used — including **SNP–SNP interaction** a linear score will miss — is a different job. That is what [GenNet](https://github.com/ArnovanHilten/GenNet) is for.

![GenNet architecture]({FIG}/paper_fig1.png)

*Fig. 1 from van Hilten et al., Communications Biology 2021 (CC BY 4.0). SNPs connect only to their genes; genes connect only to pathways. You draw those wires; the net may only use them.*

**Live hour:** `File → Save a copy in Drive` → Runtime = **CPU** → `Runtime → Run all`. GPU is not needed. Training is about 40 seconds. You can skim the theory while it runs.

**Take home:** the theory, paper figures, and section 9 (real CLI) are the path to your own PLINK/VCF. This notebook is a teaching replica, not UK Biobank.

It is **not** a GWAS and **not** a WMH discovery. Gene names (APOE, COL4A1, …) are a story scaffold so the topology looks like a CHARGE-style SVD/AD endophenotype.

Links: [paper](https://www.nature.com/articles/s42003-021-02622-z) · [GitHub](https://github.com/ArnovanHilten/GenNet) · [ALIEN](https://www.roshchupkin.org/alien/) · [A-to-Z Colab](https://colab.research.google.com/github/ArnovanHilten/GenNet/blob/master/examples/A_to_Z/GenNet_A_to_Z.ipynb)
"""
)

md(
    """## 0. Imports

Stock Colab already has TensorFlow, scikit-learn, pandas, and matplotlib. No GenNet pip install (the real CLI pins TensorFlow 2.11 and often fails here).
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
print("tensorflow", tf.__version__)"""
)

md(
    f"""## 1. What GenNet is configuring

A fully connected net would let every SNP talk to every hidden unit. With a million variants that is tens of billions of weights — and none of them have a gene name. GenNet **forbids** unbiological edges. You supply a **topology**: each row is one allowed path (usually Annovar SNP→gene, then KEGG or GTEx gene→pathway).

The CLI wants three files. That is the whole input.

![Three files]({FIG}/wiki_overview.png)

| File | Role |
|------|------|
| `genotype.h5` | people × SNPs (0/1/2) |
| `subjects.csv` | id, label, genotype row, train/val/test (`set` = 1/2/3) |
| `topology.csv` | allowed connections |

Today we keep the matrices in memory and print the same tables you would write on disk.
"""
)

md(
    """## 2. Simulate a tiny SVD-flavored cohort

**Live hour.** 1,600 people, 24 genes × 8 SNPs = 192 variants. Independent genotypes (no LD), Hardy–Weinberg draws.

Pathways (teaching names, not a real annotation):

- `lipid_endocytosis`: APOE, ABCA7, CLU, SORL1
- `immune`: TREM2, CD33, CR1, MS4A6A
- `vascular_matrix`: COL4A1, NOTCH3, FOXF2, HTRA1
- `background`: housekeeping decoys

**The label is simulated.** Do not treat the later gene ranking as a WMH discovery. We plant known biology so you can see whether the net recovers it — the same logic as Fig. 2a in the paper.
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
print(f"people={bundle['X'].shape[0]}  SNPs={bundle['X'].shape[1]}  prevalence={bundle['y'].mean():.2f}")
print("split counts", {k: int((bundle['sets'] == k).sum()) for k in (1, 2, 3)})
bundle["topology"].groupby(["layer2_name", "layer1_name"]).size().head(12)"""
)

md(
    """### Look at the topology the way GenNet would

Each row is one allowed path. APOE SNPs may talk to APOE, then to `lipid_endocytosis`. They may not skip to COL4A1 unless you put that edge in the table — that is the scientific choice.
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
    """### Why the mask matters (parameter count)

A dense layer from 192 SNPs to 24 genes would have 192×24 = 4,608 weights. The biological mask keeps **one gene per SNP** → 192 weights. At UK Biobank scale the gap is millions vs billions. That is why GenNet can train on exomes on a single GPU in the paper.
"""
)

code(
    """n_snps, n_genes = bundle["snp_gene_mask"].shape
n_pw = bundle["gene_pathway_mask"].shape[1]
dense = n_snps * n_genes + n_genes * n_pw
sparse = int(bundle["snp_gene_mask"].sum() + bundle["gene_pathway_mask"].sum())
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.bar(["Dense SNP→gene→pw", "GenNet mask"], [dense, sparse], color=["#9eb0c3", "#1f6f8b"])
ax.set_ylabel("Learnable edges in the two hidden maps")
ax.set_title("Biology prunes the wires")
for i, v in enumerate([dense, sparse]):
    ax.text(i, v, f" {v:,}", va="bottom")
plt.show()
print(f"sparsity: {sparse}/{dense} = {sparse/dense:.1%} of a dense net")"""
)

md(
    f"""## 3. A teaching replica of the directed layer

Published GenNet uses `LocallyDirected1D`: a sparse mask times a weight matrix. That class is tied to TensorFlow 2.11 internals, which is why a full install is a bad idea in a 60-minute Colab.

The layer below is the same **scientific** object at this scale:

\\[
y = \\mathrm{{act}}\\bigl(X\\,(W \\odot M) + b\\bigr)
\\]

A 1 in \\(M\\) is an allowed biological edge. **L1 on \\(W\\)** is a polygenicity knob: a larger penalty and the net uses fewer genes, like a lasso.

![Planted causal SNPs get thick weights]({FIG}/paper_fig2a.png)

*Fig. 2a, same paper. Causal SNPs (red) get large weights; control SNPs stay grey. Today's toy is that experiment with an APOE-like gene plus a planted interaction.*
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
        # Keras 3: add_weight's first positional arg is shape, not name.
        self.mask = tf.constant(self.mask_np, dtype="float32")
        self.kernel = self.add_weight(
            name="kernel",
            shape=(n_in, n_out),
            initializer="glorot_uniform",
            regularizer=tf.keras.regularizers.l1(self.l1),
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(n_out,),
            initializer="zeros",
        )
        super().build(input_shape)

    def call(self, x):
        return self.activation(tf.matmul(x, self.kernel * self.mask) + self.bias)

    def directed_weights(self):
        w = self.kernel.numpy() if hasattr(self.kernel, "numpy") else np.array(self.kernel)
        return np.asarray(w) * self.mask_np


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

On a highly heritable, low-polygenicity trait the paper's simulations (Fig. 2b–c) show AUC rising with sample size. We are in that regime on purpose: 1,600 people and a planted gene. Real WMH in CHARGE will not look this clean.
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

Lasso sees 192 SNPs with **no gene names**. It can pick `APOE_s0` but it cannot say "APOE". GenNet is forced to pool SNPs inside genes, then genes inside pathways.

In van Hilten et al. 2021 (Sweden schizophrenia exome), GenNet test AUC was **0.74** vs lasso **0.65**. That is a real, modest gain — not "deep learning beat GWAS." The paper used **exome only**, so it was not a bake-off against a genome-wide PRS.

On today's planted trait both methods should beat chance. They answer different questions.
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

GenNet importance is the product of weights along each allowed path:

`mean |SNP→gene| × |gene→pathway| × |pathway→output|`

That is **effect size along a named path**, not a p-value and not a GWAS hit. The same idea is `python GenNet.py interpret -type get_weight_scores`. Eye and hair colour in the paper recovered *HERC2* / *OCA2* (sanity check). Schizophrenia looks polygenic — many genes light up.

![Schizophrenia gene Manhattan]({FIG}/paper_fig2d.png)

*Fig. 2d · gene-layer weights for schizophrenia, coloured by chromosome. Same picture CHARGE already reads; different quantity.*
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
    f"""### Manhattan of SNP relative importance

The A-to-Z Colab ends with `python GenNet.py plot -type manhattan_relative_importance`. CHARGE rooms already read Manhattan plots — this is that picture, from the **net** instead of from a GWAS p-value.

On real data the CLI can also draw a **sunburst** of KEGG pathways (Fig. 3 in the paper). Read it from the centre. For schizophrenia the large slice was viral infectious-disease pathways — a **hypothesis**, not a diagnosis.

![KEGG sunburst]({FIG}/paper_fig3.png)
"""
)

code(
    """chr_by_snp = bundle["topology"].drop_duplicates("layer0_node").set_index("layer0_node")["chr"]
snp_rows = []
for s, name in enumerate(bundle["snp_names"]):
    g = int(bundle["snp_to_gene"][s])
    p = int(bundle["gene_to_pathway"][g])
    raw = float(abs(w_sg[s, g]) * abs(w_gp[g, p]) * abs(w_out[p]))
    snp_rows.append({
        "snp": name, "gene": bundle["gene_names"][g], "chr": int(chr_by_snp.loc[s]),
        "pos": s, "raw_importance": raw,
    })
snp_imp = pd.DataFrame(snp_rows)
snp_imp["relative_importance"] = snp_imp["raw_importance"] / snp_imp["raw_importance"].max()
display(snp_imp.sort_values("raw_importance", ascending=False).head(8))

fig, ax = plt.subplots(figsize=(9, 3.6))
colors = ["#7dcfe2", "#4b78b5", "darkgrey", "dimgray"]
for i, chrom in enumerate(sorted(snp_imp["chr"].unique())):
    sub = snp_imp[snp_imp["chr"] == chrom]
    ax.scatter(sub["pos"], sub["relative_importance"], s=18, c=colors[i % 4], label=f"chr {chrom}" if chrom in (19, 13) else None)
ax.set_xlabel("SNP index (grouped by gene / chromosome)")
ax.set_ylabel("Relative importance")
ax.set_title("Relative importance of all SNPs (this toy)")
ax.set_ylim(0, 1.25)
top = snp_imp.sort_values("raw_importance", ascending=False).head(6)
for _, r in top.iterrows():
    ax.annotate(r["snp"], (r["pos"], r["relative_importance"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.legend(loc="upper right", frameon=False)
plt.show()"""
)

md(
    """## 7. Interaction: a readable NID

A linear PRS **adds** SNP effects. Biology often **multiplies** them: two modest SNPs in the same gene can matter together and barely matter apart. Lasso tends to keep the stronger variant and shrink the partner. A gene node that sees both can learn a non-additive pattern — that is the point of the directed hidden layer.

[NID](https://arxiv.org/abs/1705.04977) (Tsang et al.) looks for features that share a hidden unit with large incoming weights. GenNet applies that **inside each gene**:

`strength(i, j) = min(|w_i|, |w_j|) × |w_gene→pathway| × |w_pathway→out|`

The A-to-Z Colab never runs this. Full CLI: `python GenNet.py interpret -type NID`. **DFIM** (perturb SNP A, watch SNP B's importance) and **PathExplain** (Expected Hessian) are the slower cousins — cluster jobs, not this room.
"""
)

code(
    """fig, ax = plt.subplots(figsize=(8.2, 2.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")
boxes = [
    (0.3, 1.7, "APOE_s0"), (0.3, 0.5, "APOE_s1"),
    (3.3, 1.1, "APOE gene"), (5.8, 1.1, "lipid pw"), (8.2, 1.1, "y"),
]
for x, y, t in boxes:
    ax.add_patch(plt.Rectangle((x, y), 1.8, 0.7, fill=True, facecolor="#fff4f1", edgecolor="#c45c4a", lw=1.5))
    ax.text(x + 0.9, y + 0.35, t, ha="center", va="center", fontsize=10)
ax.annotate("", xy=(3.3, 1.45), xytext=(2.1, 2.05), arrowprops=dict(arrowstyle="->", color="#1f6f8b"))
ax.annotate("", xy=(3.3, 1.25), xytext=(2.1, 0.85), arrowprops=dict(arrowstyle="->", color="#1f6f8b"))
ax.annotate("", xy=(5.8, 1.45), xytext=(5.1, 1.45), arrowprops=dict(arrowstyle="->", color="#1f6f8b"))
ax.annotate("", xy=(8.2, 1.45), xytext=(7.6, 1.45), arrowprops=dict(arrowstyle="->", color="#1f6f8b"))
ax.set_title("NID: pairs that share a gene node with large incoming weights")
plt.show()

nid_rows = []
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

The simulator planted (do not tell the room until they have ranked genes):

1. Strong additive **APOE** (`APOE_s0`, `APOE_s1`).
2. A **multiplicative interaction** of those two SNPs (NID target).
3. Weaker additive **COL4A1**.
4. Noise everywhere else.

If APOE is absent from the top 5, re-run the training cell. The seed is fixed, so a second run should look similar.

This is Fig. 2a as an exercise: on a trait whose signal **really does** sit in annotated genes, a directed net can name the gene and surface the pair. That is not a claim about WMH in CHARGE. It is why you might try GenNet on *your* endophenotype at home.
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
    """## 9. Take home: run this on real data

The [A-to-Z notebook](https://colab.research.google.com/github/ArnovanHilten/GenNet/blob/master/examples/A_to_Z/GenNet_A_to_Z.ipynb) is the **software tutorial** (convert PLINK, Annovar gene topology, Manhattan). This hour skipped convert because current Colab will not cleanly install TensorFlow 2.11, and added a pathway layer, a lasso baseline, and NID.

### When GenNet is the right tool

- The scientific question is **which annotated genes/pathways** (and interactions) the predictor used.
- You have a reason to believe signal sits in exons/genes/pathways (endophenotypes, Mendelian-looking genes, candidate pathways).
- You can write or generate a topology (Annovar, KEGG, GTEx, your own CSV).

### When it is the wrong tool

- You only need a **risk ranking** → modern PRS (PRS-CS, LDpred, clumped + LDpred).
- The trait is highly polygenic with most signal **non-coding**. Annotation-only SNPs and hierarchical pooling will dilute that.
- You need a p-value. Weight-path importance is an effect along a path, not a significance test.

### Checklist on a cluster or laptop (conda `env_GenNet`, Python 3.10, TF 2.11)

1. Clone https://github.com/ArnovanHilten/GenNet and `pip install -r requirements_GenNet.txt`.
2. PLINK or VCF → `python GenNet.py convert -g ./plink/ -study_name mystudy -step all`.
3. Build `subjects.csv`: `patient_id`, `labels`, `genotype_row`, `set` (1/2/3). Put close relatives in train, as in the paper.
4. Topology: Annovar gene layer, optional KEGG. Covariates (age, sex, PCs) exist in the CLI (`example_regression_cov`).
5. Train, plot, interpret:

```bash
python GenNet.py train -path ./run/ -ID 17 -L1 0.01 -epochs 100
python GenNet.py plot  -ID 17 -type manhattan_relative_importance
python GenNet.py plot  -ID 17 -type sunburst
python GenNet.py interpret -type get_weight_scores -resultpath results/GenNet_experiment_17_/
python GenNet.py interpret -type NID -resultpath results/GenNet_experiment_17_/
```

Useful knobs: `-L1`, `-L1_act`, `-problem_type regression`, `-filters`, `-onehot`, `-hidden_activation`.

Bundled toys: `examples/example_classification/` (SNP→gene), `example_regression/` (adds a pathway), `examples/A_to_Z/` (PLINK + Annovar).

**ALIEN** ([roshchupkin.org/alien](https://www.roshchupkin.org/alien/)) is the research map around this code — other topologies, multi-omics, brain regulatory context — not a second pip package. Attribution is a hypothesis. Replication still exists. It is not a diagnosis.
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {
            "name": "GenNet in one hour — Neurepiomics 2026",
            "provenance": [],
            "toc_visible": True,
        },
    },
    "cells": CELLS,
}

for cell in nb["cells"]:
    if cell["source"] and cell["source"][-1].endswith("\n"):
        cell["source"][-1] = cell["source"][-1][:-1] if cell["source"][-1] != "\n" else cell["source"][-1]

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {NB_PATH} ({len(CELLS)} cells)")
