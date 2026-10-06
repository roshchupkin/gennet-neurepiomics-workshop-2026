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
    train_gennet(model, cohort, epochs=40, verbose=0)
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


def test_interaction_control_preserves_genotypes_and_split():
    main = simulate_cohort(seed=7)
    control = simulate_cohort(seed=7, interaction_strength=0)
    np.testing.assert_array_equal(main.X, control.X)
    np.testing.assert_array_equal(main.sets, control.sets)
    assert not np.array_equal(main.y, control.y)


@pytest.mark.parametrize("n_snps_per_gene", [0, 1, 2.5])
def test_invalid_snp_count_is_rejected(n_snps_per_gene):
    with pytest.raises(ValueError, match="n_snps_per_gene"):
        simulate_cohort(n_snps_per_gene=n_snps_per_gene)


def test_masked_edges_affect_neither_prediction_nor_regularization():
    import tensorflow as tf
    from gennet_workshop.model import DirectedLayer

    layer = DirectedLayer(np.array([[1, 0], [0, 1]]), activation="linear", l1=0.1)
    X = tf.constant([[2.0, 3.0]])
    layer(X)
    layer.kernel.assign([[1, 0], [0, 2]])
    before = layer(X).numpy()
    loss_before = float(sum(layer.losses))
    layer.kernel.assign([[1, 1000], [-1000, 2]])
    np.testing.assert_allclose(layer(X).numpy(), before)
    assert float(sum(layer.losses)) == pytest.approx(loss_before)
    assert loss_before == pytest.approx(0.3)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(X)) + sum(layer.losses)
    gradient = tape.gradient(loss, layer.kernel).numpy()
    assert gradient[0, 1] == gradient[1, 0] == 0


def test_train_only_normalization_and_seeded_initialization():
    from gennet_workshop.model import build_gennet

    cohort = simulate_cohort(n_samples=200)
    first = build_gennet(cohort, seed=7)
    np.testing.assert_allclose(
        first.get_layer("train_normalization").mean.numpy().ravel(),
        cohort.split()[0].mean(axis=0),
    )
    second = build_gennet(cohort, seed=7)
    for left, right in zip(first.get_weights(), second.get_weights()):
        np.testing.assert_array_equal(left, right)


def test_nid_can_score_an_additive_logit_model():
    from gennet_workshop.model import build_gennet
    from gennet_workshop.interpret import nid_pairwise, mixed_difference, pair_logit_surface

    cohort = simulate_cohort(n_samples=200)
    model = build_gennet(cohort, hidden_activation="linear")
    gene_layer = model.get_layer("gene_layer")
    weights = np.zeros_like(cohort.snp_gene_mask)
    weights[0, 0], weights[1, 0] = 2, 1
    gene_layer.kernel.assign(weights)
    model.get_layer("pathway_layer").kernel.assign(cohort.gene_pathway_mask)
    model.get_layer("output").set_weights([np.ones((4, 1), dtype=np.float32), np.zeros(1)])
    candidates = nid_pairwise(model, cohort)
    assert candidates.iloc[0]["strength"] == pytest.approx(1)
    surface = pair_logit_surface(model, cohort.split()[4][:8], (0, 1))
    np.testing.assert_allclose(mixed_difference(surface), 0, atol=1e-5)
    with pytest.raises(ValueError, match="top_n"):
        nid_pairwise(model, cohort, top_n=1)
