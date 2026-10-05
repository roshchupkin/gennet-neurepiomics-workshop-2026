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
    """y = activation(x @ (W ⊙ M) + b), with L1 on W.

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
        self.mask = tf.constant(self.mask_np, dtype=self.dtype)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(n_in, n_out),
            initializer="glorot_uniform",
            regularizer=tf.keras.regularizers.l1(self.l1),
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
        return self.activation(tf.matmul(inputs, self.kernel * self.mask) + self.bias)

    def get_directed_weights(self) -> np.ndarray:
        return (self.kernel.numpy() * self.mask_np).astype(np.float32)


def build_gennet(
    cohort: SimulatedCohort,
    l1: float = 1e-3,
    hidden_activation: str = "tanh",
) -> tf.keras.Model:
    n_snps = cohort.n_snps
    inputs = tf.keras.Input(shape=(n_snps,), name="genotype")
    genes = DirectedLayer(
        cohort.snp_gene_mask,
        activation=hidden_activation,
        l1=l1,
        name="gene_layer",
    )(inputs)
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
