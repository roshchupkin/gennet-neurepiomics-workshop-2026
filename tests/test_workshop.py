from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from gennet_workshop.simulate import simulate_cohort, CAUSAL_GENE  # noqa: E402


def test_simulate_shapes_and_splits():
    cohort = simulate_cohort(n_samples=200, seed=1)
    assert cohort.X.shape == (200, 24 * 8)
    assert set(np.unique(cohort.sets)) == {1, 2, 3}
    assert cohort.topology["layer1_name"].nunique() == 24
    assert cohort.topology["layer2_name"].nunique() == 4
    assert 0.25 < cohort.planted["prevalence"] < 0.75
    a, b = cohort.planted["interaction_indices"]
    assert cohort.snp_names[a].startswith("APOE")
    assert cohort.snp_names[b].startswith("APOE")


def test_masks_are_exclusive():
    cohort = simulate_cohort(n_samples=50, seed=2)
    assert np.all(cohort.snp_gene_mask.sum(axis=1) == 1)
    assert np.all(cohort.gene_pathway_mask.sum(axis=1) == 1)


@pytest.mark.slow
def test_model_recovers_planted_signal():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf  # noqa: WPS433

    tf.config.set_visible_devices([], "GPU")
    from gennet_workshop.interpret import (  # noqa: WPS433
        evaluate_auc,
        gene_importance,
        nid_pairwise,
        planted_recovery,
        snp_importance,
    )
    from gennet_workshop.model import build_gennet, train_gennet  # noqa: WPS433

    cohort = simulate_cohort(n_samples=1600, seed=7)
    model = build_gennet(cohort, l1=5e-4)
    train_gennet(model, cohort, epochs=35, verbose=0)
    aucs = evaluate_auc(model, cohort)
    assert aucs["test"] >= 0.70
    snp_imp = snp_importance(model, cohort)
    recovery = planted_recovery(
        gene_importance(model, cohort), nid_pairwise(model, cohort), cohort, snp_imp
    )
    assert recovery["apoe_in_top5"], recovery
    assert recovery["interaction_rank"] is not None
    assert recovery["interaction_rank"] <= 15, recovery
    assert recovery["planted_snp_in_top5"], recovery
    assert CAUSAL_GENE in recovery["top_genes"]
