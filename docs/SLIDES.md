# Slide script (≈15 minutes)

Use with [`slides.html`](slides.html). One idea per slide. Timing assumes ~90 seconds of speech after slide 1.

---

**1. Title**  
GenNet: biologically structured neural nets for genetic prediction  
Neurepiomics 2026 · 1 hour, including Colab  
Gennady Roshchupkin · Erasmus MC

*Say:* This hour is not “train a foundation model.” It is: what is the architecture, how do you configure it, and what can you read off a tiny trained net.

---

**2. Two jobs in neurogenetics**  
- **Rank people** — polygenic scores, PRS-CS, LDpred.  
- **Name biology** — which genes/pathways the predictor actually used.

*Say:* CHARGE already does the first job well. GenNet is aimed at the second. If you only need a risk score, you probably want a PRS.

---

**3. The idea**  
You draw the wires. The net may only use those wires.

SNP → gene → pathway → phenotype  
Example: APOE SNPs may talk to the APOE gene node, not to COL4A1.

*Say:* A dense net could connect every SNP to everything. Here the mask is Annovar, KEGG, GTEx, or a table you wrote. That is a scientific choice, not a hyperparameter hiding in a YAML file.

---

**4. Three files (the whole configuration)**

| File | Role |
|------|------|
| `genotype.h5` | people × SNPs (0/1/2) |
| `subjects.csv` | id, label, row, train/val/test |
| `topology.csv` | every allowed path, SNP to output |

*Say:* If you can write `topology.csv`, you can run GenNet. The CLI is `convert`, `topology`, `train`, `plot`, `interpret`.

---

**5. What training does**  
L1 sparsity: most edges shrink toward zero. The remaining weights are readable.  
We will not wait for UK Biobank. Simulated 1,600 people, 192 SNPs, a planted “high WMH” label.

---

**6. What you read afterwards**  
- **Weights along a path** → gene / pathway importance (Manhattan, sunburst in the real tool).  
- **NID** → SNP pairs with strong joint weights (epistasis candidates).  
- **DFIM / PathExplain** → perturbation / Hessian interactions (too slow for this hour).

*Say:* These are hypotheses. Replication and wet lab still exist.

---

**7. Honest limits**  
- Annotation-only SNPs miss a lot of regulatory signal.  
- Hierarchical pooling can wash out infinitesimal genome-wide effects.  
- No LD clumping inside the net.  
- On a typical polygenic trait, a modern PRS will usually predict better.  
GenNet should **win** when the signal really does sit in annotated genes/pathways — which is the story we planted in the practical.

---

**8. Practical (Colab, CPU)**  
You will: simulate → look at topology → train ~40 seconds → AUC vs lasso → rank genes → rank SNP pairs.  
Open the notebook. Runtime = CPU. Run all.

---

**9. After today**  
Paper, GitHub, A-to-Z Colab, ALIEN map.  
`python GenNet.py interpret --help`

---

**10. Backup / Q&A**  
“Can I use imaging + genetics?” Covariates exist in the CLI.  
“Multi-omics?” That’s the ALIEN direction, not this notebook.  
“Is APOE going to come out on top?” Yes — because we planted it. On real WMH data, nobody promises that in 40 seconds.
