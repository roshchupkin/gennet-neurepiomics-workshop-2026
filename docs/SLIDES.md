# Slide script (≈18 minutes)

Open [`slides.html`](slides.html) full screen. Arrow keys or click. Figures are stored under [`figures/`](figures/) so the deck works offline.

Figures from van Hilten et al., *Communications Biology* 2021, **CC BY 4.0**.

---

**1. Title — From genotype to named biology**  
*Say:* Not a foundation-model hour. Architecture, configuration, then what you can read: genes, pathways, and SNP–SNP interaction. Paper is van Hilten et al. 2021.

---

**2. Two jobs**  
CHARGE already ranks people (PRS, GWAS). This hour is naming annotated biology, including interactions a linear score misses.

---

**3. Fig. 1 — You draw the wires**  
Linger here. SNPs may only talk to their gene. That mask is Annovar, KEGG, GTEx, or a CSV you wrote. Scientific choice, not a YAML hyperparameter.

---

**4. Fig. 2a — Planted causes go red**  
This is the whole teaching trick. Causal SNPs get thick weights. Today’s Colab is the same experiment with an APOE-like gene.

---

**5. What the paper showed**  
Eye colour: *HERC2* / *OCA2* (sanity check). Schizophrenia exome: AUC 0.74 vs lasso 0.65. Modest, real, not “deep learning beat GWAS.” Exome only.

---

**6. Fig. 2d — Manhattan of weights**  
Same picture they already know. Quantity is path-weight, not a p-value. Schizophrenia is polygenic; many genes light up.

---

**7. Fig. 3 — Sunburst**  
Read from the centre. Viral infectious-disease pathways were the large slice. Hypothesis, not a diagnosis.

---

**8. Interpretation stack**  
Weights → NID → DFIM/PathExplain. A-to-Z Colab stops at Manhattan. We added the interaction row.

---

**9. Why interaction**  
Linear PRS adds. Biology often multiplies. Two SNPs in one gene can matter together.

---

**10. NID formula**  
`min(|wi|, |wj|) × |w_later|`. DFIM: knock out A, watch B. Cluster, not this room.

---

**11. ALIEN**  
GenNet is the released code. ALIEN is the map: data/knowledge → train → investigate, plus regulatory/brain context. Not a second install.

---

**12. Three files**  
`genotype.h5`, `subjects.csv`, `topology.csv`. Then train / plot / interpret.

---

**13. Honest limits**  
Loses to modern PRS on highly polygenic, non-coding traits. Wins when signal is in annotated genes, or when the question is interaction.

---

**14. Planted toy**  
APOE, COL4A1, APOE_s0×APOE_s1. Do not quote as WMH biology.

---

**15. Colab**  
Save a copy. CPU. Run all. URL on the button.

---

**16. After today**  
Paper, GitHub, A-to-Z for convert, this hour for interaction, alien site.
