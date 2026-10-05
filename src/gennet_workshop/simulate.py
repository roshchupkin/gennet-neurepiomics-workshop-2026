"""Simulate a tiny neuroepidemiology-flavored genotype cohort.

This is teaching data, not a genetic analysis. Gene names (APOE, COL4A1, …)
are a story scaffold so the topology looks like a CHARGE-style SVD/AD
endophenotype. Allele frequencies, LD, and effect sizes are invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

N_SNPS_PER_GENE = 8

PATHWAYS: Dict[str, Sequence[str]] = {
    "lipid_endocytosis": ("APOE", "ABCA7", "CLU", "SORL1"),
    "immune": ("TREM2", "CD33", "CR1", "MS4A6A"),
    "vascular_matrix": ("COL4A1", "NOTCH3", "FOXF2", "HTRA1"),
    "background": (
        "GAPDH",
        "ACTB",
        "RPLP0",
        "B2M",
        "ALB",
        "TUBB",
        "PPIA",
        "HPRT1",
        "YWHAZ",
        "UBC",
        "RPS18",
        "EEF1A1",
    ),
}

# Planted biology (see simulate_cohort docstring).
CAUSAL_GENE = "APOE"
CAUSAL_SNP_A = 0  # first SNP in APOE
CAUSAL_SNP_B = 1  # second SNP in APOE (interacts with A)
VASCULAR_GENE = "COL4A1"


@dataclass
class SimulatedCohort:
    X: np.ndarray
    y: np.ndarray
    sets: np.ndarray
    snp_names: List[str]
    gene_names: List[str]
    pathway_names: List[str]
    snp_to_gene: np.ndarray
    gene_to_pathway: np.ndarray
    topology: pd.DataFrame
    subjects: pd.DataFrame
    snp_gene_mask: np.ndarray
    gene_pathway_mask: np.ndarray
    planted: dict

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_snps(self) -> int:
        return int(self.X.shape[1])

    def split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train = self.sets == 1
        val = self.sets == 2
        test = self.sets == 3
        return self.X[train], self.y[train], self.X[val], self.y[val], self.X[test], self.y[test]


def _catalog() -> Tuple[List[str], List[str], Dict[str, str]]:
    gene_to_pathway: Dict[str, str] = {}
    gene_names: List[str] = []
    pathway_names = list(PATHWAYS.keys())
    for pathway, genes in PATHWAYS.items():
        for gene in genes:
            gene_names.append(gene)
            gene_to_pathway[gene] = pathway
    return gene_names, pathway_names, gene_to_pathway


def simulate_cohort(
    n_samples: int = 1600,
    n_snps_per_gene: int = N_SNPS_PER_GENE,
    seed: int = 7,
) -> SimulatedCohort:
    """Draw genotypes and a binary 'high WMH burden' label.

    Planted signal (known to the instructor, recovered in the practical):

    1. Additive APOE (two SNPs) — strong main effect, analogous to ε4.
    2. Within-gene interaction APOE_snp0 × APOE_snp1 — NID target.
    3. Weaker additive COL4A1 — vascular-matrix pathway.
    4. Everything else is noise.

    Splits: 70% train (set=1), 15% validation (set=2), 15% test (set=3).
    Genotypes are independent Bernoulli draws coded 0/1/2 (no LD).
    """
    rng = np.random.default_rng(seed)
    gene_names, pathway_names, gene_to_pathway = _catalog()
    n_genes = len(gene_names)
    n_pathways = len(pathway_names)
    n_snps = n_genes * n_snps_per_gene

    maf = rng.uniform(0.08, 0.42, size=n_snps)
    # Hardy–Weinberg genotype probabilities from allele frequency.
    p = maf
    probs = np.stack([(1 - p) ** 2, 2 * p * (1 - p), p ** 2], axis=1)
    X = np.empty((n_samples, n_snps), dtype=np.float32)
    for j in range(n_snps):
        X[:, j] = rng.choice([0.0, 1.0, 2.0], size=n_samples, p=probs[j])

    snp_names = []
    snp_to_gene = np.empty(n_snps, dtype=np.int32)
    gene_to_pw = np.array(
        [pathway_names.index(gene_to_pathway[g]) for g in gene_names], dtype=np.int32
    )
    rows = []
    for g_i, gene in enumerate(gene_names):
        pw_i = int(gene_to_pw[g_i])
        for k in range(n_snps_per_gene):
            s = g_i * n_snps_per_gene + k
            name = f"{gene}_s{k}"
            snp_names.append(name)
            snp_to_gene[s] = g_i
            rows.append(
                {
                    "chr": 19 if gene == "APOE" else (13 if gene == "COL4A1" else (g_i % 22) + 1),
                    "layer0_node": s,
                    "layer0_name": name,
                    "layer1_node": g_i,
                    "layer1_name": gene,
                    "layer2_node": pw_i,
                    "layer2_name": pathway_names[pw_i],
                }
            )
    topology = pd.DataFrame(rows)

    apoe = gene_names.index(CAUSAL_GENE)
    col4 = gene_names.index(VASCULAR_GENE)
    a = apoe * n_snps_per_gene + CAUSAL_SNP_A
    b = apoe * n_snps_per_gene + CAUSAL_SNP_B
    col4_idx = np.arange(col4 * n_snps_per_gene, (col4 + 1) * n_snps_per_gene)

    # Standardized dosages so coefficients are on a similar scale.
    def z(col: np.ndarray) -> np.ndarray:
        return (col - col.mean()) / (col.std() + 1e-6)

    logit = (
        1.55 * z(X[:, a])
        + 0.95 * z(X[:, b])
        + 1.15 * z(X[:, a]) * z(X[:, b])
        + 0.55 * z(X[:, col4_idx].mean(axis=1))
    )
    logit = logit - logit.mean()
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, prob).astype(np.float32)

    order = rng.permutation(n_samples)
    n_train = int(0.70 * n_samples)
    n_val = int(0.15 * n_samples)
    sets = np.empty(n_samples, dtype=np.int8)
    sets[order[:n_train]] = 1
    sets[order[n_train : n_train + n_val]] = 2
    sets[order[n_train + n_val :]] = 3

    subjects = pd.DataFrame(
        {
            "patient_id": [f"s{i:04d}" for i in range(n_samples)],
            "labels": y.astype(int),
            "genotype_row": np.arange(n_samples),
            "set": sets,
        }
    )

    snp_gene_mask = np.zeros((n_snps, n_genes), dtype=np.float32)
    snp_gene_mask[np.arange(n_snps), snp_to_gene] = 1.0
    gene_pathway_mask = np.zeros((n_genes, n_pathways), dtype=np.float32)
    gene_pathway_mask[np.arange(n_genes), gene_to_pw] = 1.0

    planted = {
        "phenotype": "simulated high WMH burden (binary)",
        "causal_gene": CAUSAL_GENE,
        "vascular_gene": VASCULAR_GENE,
        "interaction_snps": (snp_names[a], snp_names[b]),
        "interaction_indices": (int(a), int(b)),
        "prevalence": float(y.mean()),
    }
    return SimulatedCohort(
        X=X,
        y=y,
        sets=sets,
        snp_names=snp_names,
        gene_names=gene_names,
        pathway_names=pathway_names,
        snp_to_gene=snp_to_gene,
        gene_to_pathway=gene_to_pw,
        topology=topology,
        subjects=subjects,
        snp_gene_mask=snp_gene_mask,
        gene_pathway_mask=gene_pathway_mask,
        planted=planted,
    )
