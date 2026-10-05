"""Weight-path importance and a simplified NID pairwise interaction score."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model import DirectedLayer
from .simulate import SimulatedCohort, CAUSAL_GENE, VASCULAR_GENE


def _layer(model, name: str) -> DirectedLayer:
    layer = model.get_layer(name)
    if not isinstance(layer, DirectedLayer):
        raise TypeError(f"{name} is {type(layer)}, expected DirectedLayer")
    return layer


def directed_weights(model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w_sg = _layer(model, "gene_layer").get_directed_weights()
    w_gp = _layer(model, "pathway_layer").get_directed_weights()
    w_out = model.get_layer("output").get_weights()[0].reshape(-1)
    return w_sg, w_gp, w_out


def gene_importance(model, cohort: SimulatedCohort) -> pd.DataFrame:
    """Product of mean |SNP→gene| weight, |gene→pathway|, and |pathway→out|."""
    w_sg, w_gp, w_out = directed_weights(model)
    rows = []
    for g, gene in enumerate(cohort.gene_names):
        snps = np.where(cohort.snp_to_gene == g)[0]
        p = int(cohort.gene_to_pathway[g])
        snp_mag = float(np.mean(np.abs(w_sg[snps, g])))
        gene_pw = float(np.abs(w_gp[g, p]))
        pw_out = float(np.abs(w_out[p]))
        score = snp_mag * gene_pw * pw_out
        rows.append(
            {
                "gene": gene,
                "pathway": cohort.pathway_names[p],
                "snp_to_gene": snp_mag,
                "gene_to_pathway": gene_pw,
                "pathway_to_out": pw_out,
                "importance": score,
            }
        )
    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    return df.reset_index(drop=True)


def pathway_importance(model, cohort: SimulatedCohort) -> pd.DataFrame:
    w_sg, w_gp, w_out = directed_weights(model)
    rows = []
    for p, pathway in enumerate(cohort.pathway_names):
        genes = np.where(cohort.gene_to_pathway == p)[0]
        gene_mag = float(np.mean(np.abs(w_gp[genes, p]))) if len(genes) else 0.0
        score = gene_mag * float(np.abs(w_out[p]))
        rows.append(
            {
                "pathway": pathway,
                "n_genes": int(len(genes)),
                "gene_to_pathway": gene_mag,
                "pathway_to_out": float(np.abs(w_out[p])),
                "importance": score,
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def nid_pairwise(model, cohort: SimulatedCohort, top_n: int = 8) -> pd.DataFrame:
    """Simplified NID: min(|w_i|, |w_j|) × |w_gene→pw| × |w_pw→out| within each gene.

    Full GenNet NID (`python GenNet.py interpret -type NID`) uses the same
    Tsang-style idea on the trained LocallyDirected weights. This version is
    small enough to read in a workshop cell.
    """
    w_sg, w_gp, w_out = directed_weights(model)
    rows = []
    for g, gene in enumerate(cohort.gene_names):
        snps = np.where(cohort.snp_to_gene == g)[0]
        p = int(cohort.gene_to_pathway[g])
        later = float(np.abs(w_gp[g, p]) * np.abs(w_out[p]))
        mags = np.abs(w_sg[snps, g])
        order = np.argsort(-mags)[: min(top_n, len(snps))]
        for ii in range(len(order)):
            for jj in range(ii + 1, len(order)):
                i = int(snps[order[ii]])
                j = int(snps[order[jj]])
                strength = float(min(mags[order[ii]], mags[order[jj]]) * later)
                rows.append(
                    {
                        "gene": gene,
                        "snp_i": cohort.snp_names[i],
                        "snp_j": cohort.snp_names[j],
                        "strength": strength,
                    }
                )
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)


def fit_lasso(cohort: SimulatedCohort, C: float = 0.2) -> Tuple[Pipeline, float, pd.DataFrame]:
    X_train, y_train, _, _, X_test, y_test = cohort.split()
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=C,
                    max_iter=4000,
                    random_state=0,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    auc = float(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]))
    coef = np.abs(pipe.named_steps["clf"].coef_.ravel())
    df = pd.DataFrame({"snp": cohort.snp_names, "abs_coef": coef})
    df["gene"] = [cohort.gene_names[i] for i in cohort.snp_to_gene]
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return pipe, auc, df


def evaluate_auc(model, cohort: SimulatedCohort) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    X_train, y_train, X_val, y_val, X_test, y_test = cohort.split()
    out = {}
    for name, X, y in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        pred = model.predict(X, verbose=0).ravel()
        out[name] = float(roc_auc_score(y, pred))
    return out


def planted_recovery(
    gene_imp: pd.DataFrame, nid: pd.DataFrame, cohort: SimulatedCohort
) -> Dict[str, object]:
    """Sanity checks for the instructor / automated test."""
    top_genes: List[str] = gene_imp["gene"].head(5).tolist()
    pair = set(cohort.planted["interaction_snps"])
    nid_pairs = [set([a, b]) for a, b in zip(nid["snp_i"], nid["snp_j"])]
    pair_rank = next((i + 1 for i, p in enumerate(nid_pairs) if p == pair), None)
    return {
        "apoe_in_top5": CAUSAL_GENE in top_genes,
        "col4a1_in_top8": VASCULAR_GENE in gene_imp["gene"].head(8).tolist(),
        "interaction_rank": pair_rank,
        "top_genes": top_genes,
    }
