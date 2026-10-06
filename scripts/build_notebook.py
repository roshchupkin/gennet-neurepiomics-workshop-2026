#!/usr/bin/env python3
"""Write notebooks/01_gennet_in_one_hour.ipynb (self-contained Colab practical)."""

from __future__ import annotations

import json
import ast
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


def shared_code(filename: str, *names: str) -> str:
    """Embed checked-in helpers so the Colab stays self-contained and in sync."""
    source = (NB_PATH.parents[1] / "src" / "gennet_workshop" / filename).read_text()
    tree = ast.parse(source)
    snippets = []
    for name in names:
        node = next(node for node in tree.body if getattr(node, "name", None) == name)
        snippet = ast.get_source_segment(source, node)
        snippet = snippet.replace("cohort: SimulatedCohort", "bundle")
        snippet = snippet.replace("cohort.n_snps", 'bundle["X"].shape[1]')
        snippet = snippet.replace("cohort.split()", "split(bundle)")
        for field in ("snp_gene_mask", "gene_pathway_mask", "snp_to_gene", "gene_to_pathway",
                      "snp_names", "gene_names", "pathway_names", "topology"):
            snippet = snippet.replace("cohort." + field, 'bundle["' + field + '"]')
        snippets.append(snippet)
    return "\n\n".join(snippets)


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

from typing import Tuple, Dict
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

tf.get_logger().setLevel("ERROR")
tf.config.experimental.enable_op_determinism()
print("tensorflow", tf.__version__, "scikit-learn", sklearn.__version__)"""
)

md(
    f"""## 1. What GenNet is configuring

A fully connected net would let every SNP talk to every hidden unit. With a million variants that is tens of billions of weights — and none of them have a gene name. GenNet restricts edges to the chosen biological prior. You supply a **topology**: each row is one allowed path (usually Annovar SNP→gene, then KEGG or GTEx gene→pathway).

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
MODEL_SEED = 7


def simulate_cohort(n_samples=N_SAMPLES, n_snps_per_gene=N_SNPS_PER_GENE, seed=SEED, interaction_strength=1.15):
    if not isinstance(n_snps_per_gene, (int, np.integer)) or n_snps_per_gene < 2:
        raise ValueError("n_snps_per_gene must be an integer >= 2")
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 20:
        raise ValueError("n_samples must be an integer >= 20")
    if not np.isfinite(interaction_strength):
        raise ValueError("interaction_strength must be finite")
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
        + interaction_strength * z(X[:, a]) * z(X[:, b])
        + 0.55 * z(X[:, col4_idx].mean(axis=1))
    )
    logit = logit - logit.mean()
    y = rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(np.float32)

    order = np.random.default_rng(seed + 1).permutation(n_samples)
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
        "interaction_indices": (a, b),
        "interaction_strength": float(interaction_strength),
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
    """### Why the mask matters (allowed connections)

A dense SNP-to-gene map has 192×24 = 4,608 connections. The mask permits 192 of them. Including gene-to-pathway edges, there are **216 allowed hidden connections**.

This teaching layer still allocates dense weight matrices: **4,704 hidden weights**, plus biases and the output layer. The model summary therefore reports 4,737 trainable parameters. Masked-out weights cannot affect predictions and receive no L1 penalty. The production GenNet layer stores sparse edges; this dense replica is suitable for the small toy only.
"""
)

code(
    """n_snps, n_genes = bundle["snp_gene_mask"].shape
n_pw = bundle["gene_pathway_mask"].shape[1]
dense = n_snps * n_genes + n_genes * n_pw
sparse = int(bundle["snp_gene_mask"].sum() + bundle["gene_pathway_mask"].sum())
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.bar(["Dense SNP→gene→pw", "GenNet mask"], [dense, sparse], color=["#9eb0c3", "#1f6f8b"])
ax.set_ylabel("Allowed edges in the two hidden maps")
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

A 1 in \\(M\\) is an allowed biological edge. **L1 on \\(W\\)** encourages small or sparse allowed-edge weights. It does not directly estimate polygenicity. Inputs are standardized using **training participants only**.

![Planted causal SNPs get thick weights]({FIG}/paper_fig2a.png)

*Fig. 2a, same paper. Causal SNPs (red) get large weights; control SNPs stay grey. Today's toy is that experiment with an APOE-like gene plus a planted interaction.*
"""
)

code(
    shared_code("model.py", "DirectedLayer", "build_gennet")
    + '\n\nmodel = build_gennet(bundle, seed=MODEL_SEED)\nmodel.summary()'
)

md(
    """## 4. Train (CPU, about 30–60 seconds)

Early stopping watches validation AUC. This cell **rebuilds** the model using `MODEL_SEED` before every fit. The data seed and model seed are separate. Run all downstream cells after retraining so tables reflect the new model.

If this cell is slow, wait; GPU is not needed. Different TensorFlow versions may still produce slightly different results.

On a highly heritable, low-polygenicity trait the paper's simulations (Fig. 2b–c) show AUC rising with sample size. We are in that regime on purpose: 1,600 people and a planted gene. Real WMH in CHARGE will not look this clean.
"""
)

code(
    """def reset_and_train(bundle, seed=MODEL_SEED, epochs=40, l1=5e-4):
    tf.keras.backend.clear_session()
    model = build_gennet(bundle, l1=l1, seed=seed)
    X_tr, y_tr, X_va, y_va, _, _ = split(bundle)
    history = model.fit(
        X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=64,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=8, restore_best_weights=True
        )],
    )
    return model, history


X_tr, y_tr, X_va, y_va, X_te, y_te = split(bundle)
model, history = reset_and_train(bundle)
print(f"Trained {len(history.history['auc'])} epochs with model seed {MODEL_SEED}")
# Early stopping uses validation only; report test AUC after fitting.

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

L1 logistic regression learns additive SNP effects without using the gene/pathway mask during training. Its coefficients can still be annotated and summarized by gene afterwards. GenNet uses that biological structure **during learning**, with nonlinear hidden layers.

In van Hilten et al. 2021 (Sweden schizophrenia exome), GenNet test AUC was **0.74** vs lasso **0.65**. That is a real, modest gain — not "deep learning beat GWAS." The paper used **exome only**, so it was not a bake-off against a genome-wide PRS.

Both methods should beat chance. This is a teaching comparison with a fixed logistic penalty, not a tuned benchmark or evidence that one method generally wins.
"""
)

code(
    shared_code("interpret.py", "fit_lasso")
    + """\n\nlasso, lasso_auc, _ = fit_lasso(bundle)
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
    f"""## 6. Explainability: weight paths to genes and pathways

GenNet importance is the product of weights along each allowed path:

`mean |SNP→gene| × |gene→pathway| × |pathway→output|`

This is a **weight-based importance score**, not an effect size or a p-value. It omits activation derivatives and depends on the model parameterization. Gene importance averages SNP path scores; the pathway table below is a downstream-weight summary that omits SNP-to-gene weights. The same idea is `python GenNet.py interpret -type get_weight_scores`. Eye and hair colour in the paper recovered *HERC2* / *OCA2* (sanity check). Schizophrenia looks polygenic — many genes light up.

![Schizophrenia gene Manhattan]({FIG}/paper_fig2d.png)

*Fig. 2d · gene-layer weights for schizophrenia, coloured by chromosome. Same picture CHARGE already reads; different quantity.*
"""
)

code(
    """w_sg = model.get_layer("gene_layer").get_directed_weights()
w_gp = model.get_layer("pathway_layer").get_directed_weights()
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
snp_imp["relative_importance"] = snp_imp["raw_importance"] / (float(snp_imp["raw_importance"].max()) or 1.0)
snp_imp = snp_imp.sort_values(["chr", "pos"]).reset_index(drop=True)
snp_imp["pos"] = np.arange(len(snp_imp))
display(snp_imp.sort_values("raw_importance", ascending=False).head(8))

fig, ax = plt.subplots(figsize=(9, 3.6))
colors = ["#7dcfe2", "#4b78b5", "darkgrey", "dimgray"]
for i, chrom in enumerate(sorted(snp_imp["chr"].unique())):
    sub = snp_imp[snp_imp["chr"] == chrom]
    ax.scatter(sub["pos"], sub["relative_importance"], s=18, c=colors[i % 4], label=f"chr {chrom}" if chrom in (19, 13) else None)
ax.set_xlabel("SNP index ordered by toy chromosome (not genomic distance)")
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
    """## 7. Interaction candidates: a readable NID

An additive predictor cannot represent the explicit SNP-product term we plant on the **logit scale**. A nonlinear gene node can learn non-additivity. Strong main effects, however, can also give a SNP pair a high weight-based score.

[NID](https://arxiv.org/abs/1705.04977) (Tsang et al.) looks for features that share a hidden unit with large incoming weights. GenNet applies that **inside each gene**:

`strength(i, j) = min(|w_i|, |w_j|) × |w_gene→pathway| × |w_pathway→out|`

**This simplified score ranks candidates; it does not prove epistasis or supply a significance test.** It ignores activation behavior and can be positive even for an additive-logit network. Our toy searches within genes only; cross-gene interactions are outside this table.

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

"""
    + "\n\n" + shared_code("interpret.py", "_layer", "directed_weights", "nid_pairwise")
    + """\n\nnid = nid_pairwise(model, bundle)
display(nid.head(12))
print("Planted pair (look after you have stared at the table):", bundle["interaction"])"""
)

md(
    """### Check non-additivity on the logit scale

A high NID score is a reason to inspect a pair. Fit an **additive-only control** with the same genotypes, split and model seed, but with the simulated interaction coefficient set to zero.

For each fitted model, vary the candidate dosages over 0/1/2 while holding the other SNPs at 32 fixed test-participant backgrounds. Average the model **logit** over those backgrounds. Adjacent mixed differences quantify departure from additivity on this chosen scale:

`Δ = f(a+1,b+1) - f(a+1,b) - f(a,b+1) + f(a,b)`

The known additive mechanism has zero mixed differences. A fitted neural network may still invent non-additivity through finite-sample error or its architecture; the control makes that limitation visible. With only one unit per gene, this tiny model also cannot represent arbitrary interaction surfaces. The fitted control can show **more** non-additivity than the interaction-trained model: good prediction and a correct candidate ranking do not establish recovery of the mechanism.

Neither this probe nor NID is a biological effect estimate or a significance test. In real LD data, arbitrary dosage combinations may be unsupported and require a different evaluation design.
"""
)

code(
    shared_code("interpret.py", "mixed_difference", "pair_logit_surface")
    + """\n\nadditive_bundle = simulate_cohort(interaction_strength=0.0)
assert np.array_equal(bundle["X"], additive_bundle["X"])
assert np.array_equal(bundle["sets"], additive_bundle["sets"])
control_model, control_history = reset_and_train(additive_bundle)
control_nid = nid_pairwise(control_model, additive_bundle)
print("Additive-only control: top NID candidates (no interaction was planted)")
display(control_nid.head(3))

a, b = bundle["interaction_indices"]
reference = X_te[:32]
main_surface = pair_logit_surface(model, reference, (a, b))
control_surface = pair_logit_surface(control_model, reference, (a, b))
za = (np.arange(3) - bundle["X"][:, a].mean()) / (bundle["X"][:, a].std() + 1e-6)
zb = (np.arange(3) - bundle["X"][:, b].mean()) / (bundle["X"][:, b].std() + 1e-6)
oracle_additive = 1.55 * za[:, None] + 0.95 * zb[None, :]
oracle_interacting = oracle_additive + bundle["interaction_strength"] * za[:, None] * zb[None, :]
surfaces = [oracle_additive, oracle_interacting, control_surface, main_surface]
titles = ["Known additive mechanism", "Known interaction mechanism",
          "Fitted additive-only control", "Fitted interaction model"]
centered = [surface - surface.mean() for surface in surfaces]
limit = max(float(np.abs(surface).max()) for surface in centered)
fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
for ax, title, surface in zip(axes.ravel(), titles, centered):
    im = ax.imshow(surface, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xlabel(bundle["snp_names"][b] + " dosage")
    ax.set_ylabel(bundle["snp_names"][a] + " dosage")
    ax.set_title(title)
fig.colorbar(im, ax=list(axes.ravel()), label="Centered logit")
plt.show()
interaction_checks = pd.DataFrame({
    "mechanism / model": titles,
    "mean absolute mixed difference": [float(np.abs(mixed_difference(surface)).mean()) for surface in surfaces],
})
display(interaction_checks)
assert np.allclose(mixed_difference(oracle_additive), 0)
print("Inspect the fitted control before interpreting the main model's non-additivity.")
"""
)

md(
    """## 8. What you should have recovered

The simulator planted (do not tell the room until they have ranked genes):

1. Strong additive **APOE** (`APOE_s0`, `APOE_s1`).
2. A **multiplicative interaction** of those two SNPs (NID target).
3. Weaker additive **COL4A1**.
4. Noise everywhere else.

Expect APOE near the top for this fixed teaching configuration. COL4A1 is weaker and its rank may vary. If recovery fails, keep the result visible and discuss optimization or use the completed backup notebook. Re-running with the same seed restarts the same experiment; it is not a way to select a favorable answer.

This is Fig. 2a as an exercise: on a trait whose signal **really does** sit in annotated genes, a directed net can recover gene importance and propose the pair for further checks. That is not a claim about WMH in CHARGE. It is why you might try GenNet on *your* endophenotype at home.
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

### What requires further evaluation

- For risk ranking, compare against suitable PRS methods and other prediction baselines on held-out data.
- For highly polygenic or non-coding traits, evaluate variant coverage and the chosen regulatory mapping. This toy does not establish performance in that setting.
- For statistical inference, use a validated testing procedure. Weight-path importance and this NID score do not supply p-values.

### Checklist on a cluster or laptop (conda `env_GenNet`, Python 3.10, TF 2.11)

1. Clone https://github.com/ArnovanHilten/GenNet and `pip install -r requirements_GenNet.txt`.
2. PLINK or VCF → `python GenNet.py convert -g ./plink/ -study_name mystudy -step all`.
3. Build `subjects.csv`: `patient_id`, `labels`, `genotype_row`, `set` (1/2/3). Keep related participants together; prevent relatives from straddling train, validation and test sets. State the kinship and cohort split policy explicitly.
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

md(
    """## 10. Optional ALIEN experiments

The live exercise ends above. For a longer practical, change one factor at a time:

1. **Topology:** run the optional cell below. It shuffles SNP-to-gene assignments while preserving the number of allowed edges. Compare validation/test AUC; one realization does not establish that biology always helps.
2. **Initialization:** repeat `reset_and_train(bundle, seed=...)` with several model seeds and compare gene ranks. Preserve every run, including failures. A fixed seed makes a workshop repeatable; multiple seeds assess robustness.
3. **Sparsity:** vary `l1` using validation results and inspect prediction versus gene ranking. Keep test data out of configuration selection.

These are known-truth benchmark exercises for ALIEN, not new WMH findings. This example does not test LD, ancestry transfer, covariates, multiomics or federation.
"""
)

code(
    """RUN_TOPOLOGY_EXPERIMENT = False  # Optional: change to True and run this cell.
if RUN_TOPOLOGY_EXPERIMENT:
    shuffled_bundle = dict(bundle)
    permutation = np.random.default_rng(17).permutation(bundle["X"].shape[1])
    shuffled_bundle["snp_gene_mask"] = bundle["snp_gene_mask"][permutation]
    shuffled_model, _ = reset_and_train(shuffled_bundle)
    print("Correct topology AUC:", gennet_auc)
    print("Shuffled topology AUC:", {
        "val": auc(shuffled_model, X_va, y_va), "test": auc(shuffled_model, X_te, y_te)
    })
    print("Compare prediction here; the shuffled gene labels are no longer biological annotations.")
else:
    print("Optional topology experiment skipped; set RUN_TOPOLOGY_EXPERIMENT=True to explore.")
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

for i, cell in enumerate(nb["cells"]):
    cell["id"] = f"workshop-{i:02d}"
    if cell["source"] and cell["source"][-1].endswith("\n"):
        cell["source"][-1] = cell["source"][-1][:-1] if cell["source"][-1] != "\n" else cell["source"][-1]

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {NB_PATH} ({len(CELLS)} cells)")
