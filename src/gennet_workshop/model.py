"""A Colab-safe teaching replica of GenNet's directed SNP→gene→pathway net.

The published GenNet layer is LocallyDirected1D (sparse COO mask). That class
imports TensorFlow 2.11 keras internals and is brittle on current Colab. This
module keeps the same scientific idea — only biologically allowed edges exist —
with a dense mask multiply that is trivial at workshop scale (~200 SNPs).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .simulate import SimulatedCohort


class DirectedLayer(tf.keras.layers.Layer):
    """y = activation(x @ (W ⊙ M) + b), with L1 only on allowed edges.

    M is a 0/1 mask of shape (n_in, n_out). A 1 means that input node is
    allowed to connect to that output node (SNP→gene, or gene→pathway).
    """

    def __init__(
        self,
        mask: np.ndarray,
        activation: str = "tanh",
        l1: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        mask = np.asarray(mask, dtype=np.float32)
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D, got {mask.shape}")
        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError("mask must contain only 0 and 1")
        if l1 < 0:
            raise ValueError("l1 must be non-negative")
        self.mask_np = mask
        self.activation_name = activation
        self.l1 = float(l1)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        n_in, n_out = self.mask_np.shape
        if int(input_shape[-1]) != n_in:
            raise ValueError(
                f"Last dim {input_shape[-1]} does not match mask rows {n_in}"
            )
        self.mask = tf.constant(self.mask_np, dtype=tf.float32)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(n_in, n_out),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(n_out,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        directed_kernel = self.kernel * self.mask
        self.add_loss(self.l1 * tf.reduce_sum(tf.abs(directed_kernel)))
        return self.activation(tf.matmul(inputs, directed_kernel) + self.bias)

    def get_config(self):
        return {
            **super().get_config(),
            "mask": self.mask_np.tolist(),
            "activation": self.activation_name,
            "l1": self.l1,
        }

    def get_directed_weights(self) -> np.ndarray:
        w = self.kernel.numpy() if hasattr(self.kernel, "numpy") else np.array(self.kernel)
        return (np.asarray(w) * self.mask_np).astype(np.float32)


def build_gennet(
    cohort: SimulatedCohort,
    l1: float = 5e-4,
    hidden_activation: str = "tanh",
    seed: int = 7,
) -> tf.keras.Model:
    # Seed initialization and training shuffle, independently of simulation.
    tf.keras.utils.set_random_seed(seed)
    n_snps = cohort.n_snps
    inputs = tf.keras.Input(shape=(n_snps,), name="genotype")
    X_train = cohort.split()[0]
    # Fit preprocessing on training participants only. Keep raw dosages at the API.
    normalized = tf.keras.layers.Normalization(
        mean=X_train.mean(axis=0),
        variance=X_train.var(axis=0),
        name="train_normalization",
    )(inputs)
    genes = DirectedLayer(
        cohort.snp_gene_mask,
        activation=hidden_activation,
        l1=l1,
        name="gene_layer",
    )(normalized)
    pathways = DirectedLayer(
        cohort.gene_pathway_mask,
        activation=hidden_activation,
        l1=l1,
        name="pathway_layer",
    )(genes)
    logit = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=tf.keras.regularizers.l1(l1),
        name="output",
    )(pathways)
    model = tf.keras.Model(inputs, logit, name="mini_gennet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="acc"),
        ],
    )
    return model


def train_gennet(
    model: tf.keras.Model,
    cohort: SimulatedCohort,
    epochs: int = 40,
    batch_size: int = 64,
    patience: int = 8,
    verbose: int = 0,
) -> tf.keras.callbacks.History:
    X_train, y_train, X_val, y_val, _, _ = cohort.split()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
