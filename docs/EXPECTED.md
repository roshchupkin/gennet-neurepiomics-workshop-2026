# Expected outputs (seed=7, 1600 samples, 35–40 epochs, CPU)

Recorded on this workstation with `env_GenNet` (TensorFlow 2.11), 2026-10-05.
Colab’s TensorFlow is newer; numbers will move a little, ranks should not.

| Check | Expected |
|-------|----------|
| Prevalence | ~0.42 |
| Test AUC | ~0.75–0.85 (this run 0.77) |
| Train wall time | ~6 seconds for 35 epochs |
| Gene rank 1 | **APOE** |
| Gene rank 2 | **COL4A1** (often) |
| NID rank 1 | **APOE_s0 × APOE_s1** |

If APOE is not top 5, re-run the train cell once. Do not debug Keras in the room.
