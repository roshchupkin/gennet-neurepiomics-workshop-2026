# Expected outputs

Recorded 6 October 2026 with Python 3.12, TensorFlow 2.20.0 and scikit-learn 1.8.0, CPU. Both the simulation and model use seed 7; training runs up to 40 epochs. The split is fixed independently of the phenotype mechanism.

| Check | Recorded output |
|-------|-----------------|
| People / variants | 1,600 / 192 |
| Train / validation / test | 1,120 / 240 / 240 |
| Prevalence | 0.42 |
| GenNet test AUC | 0.790 |
| L1 logistic test AUC | 0.788 |
| Gene rank 1 / 2 | APOE / COL4A1 |
| NID candidate rank 1 | APOE_s0 × APOE_s1 |
| Allowed hidden connections | 216 |
| Allocated hidden weights / total trainable parameters | 4,704 / 4,737 |

Numbers and weaker ranks may change across TensorFlow versions. These are teaching outputs, not a performance benchmark or a guarantee for arbitrary seeds.

## Interaction control

No interaction is planted in the additive-only control, yet its top NID candidate is also APOE_s0 × APOE_s1. Mean absolute adjacent mixed differences on the logit scale:

| Surface | Recorded value |
|---------|----------------|
| Known additive mechanism | approximately 0 |
| Known interaction mechanism | 2.590 |
| Fitted additive-only control | 0.421 |
| Fitted interaction model | 0.349 |

The fitted control is more non-additive than the main model in this configuration. Do not describe the NID ranking as validated interaction recovery. The one-unit-per-gene bottleneck, finite-sample error and optimization affect mechanism recovery.

If recovery fails, record it and use the completed backup. The reset-and-train cell repeats a seeded experiment; it does not select a successful restart.
