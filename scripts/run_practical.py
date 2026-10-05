#!/usr/bin/env python3
"""Train the workshop model locally and print recovery (instructor check)."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from gennet_workshop.interpret import evaluate_auc, gene_importance, nid_pairwise, planted_recovery, snp_importance
from gennet_workshop.model import build_gennet, train_gennet
from gennet_workshop.simulate import simulate_cohort


def main() -> int:
    cohort = simulate_cohort(n_samples=1600, seed=7)
    print(
        f"n={cohort.n_samples} snps={cohort.n_snps} "
        f"prevalence={cohort.planted['prevalence']:.2f} "
        f"pair={cohort.planted['interaction_snps']}"
    )
    model = build_gennet(cohort, l1=5e-4)
    train_gennet(model, cohort, epochs=35, verbose=1)
    aucs = evaluate_auc(model, cohort)
    genes = gene_importance(model, cohort)
    nid = nid_pairwise(model, cohort)
    snps = snp_importance(model, cohort)
    rec = planted_recovery(genes, nid, cohort, snps)
    print("AUC", {k: round(v, 3) for k, v in aucs.items()})
    print(genes.head(8).to_string(index=False))
    print(snps.head(8).to_string(index=False))
    print(nid.head(8).to_string(index=False))
    print("recovery", rec)
    ok = (
        rec["apoe_in_top5"]
        and rec["interaction_rank"] is not None
        and rec["interaction_rank"] <= 15
        and rec.get("planted_snp_in_top5", True)
    )
    return 0 if ok and aucs["test"] >= 0.70 else 1


if __name__ == "__main__":
    raise SystemExit(main())
