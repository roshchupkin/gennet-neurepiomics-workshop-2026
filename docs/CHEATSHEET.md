# GenNet cheat sheet

## This hour (Colab notebook)

Stock Colab CPU. No `pip` of GenNet.

| Step | What to look at |
|------|-----------------|
| Simulate | 1600 × 192, prevalence ~0.4–0.6 |
| Topology | `layer0` SNP, `layer1` gene, `layer2` pathway |
| Train | val AUC ≫ 0.5 |
| Lasso | top SNPs should include `APOE_s0` / `APOE_s1` |
| Gene importance | APOE high; COL4A1 often present |
| NID | `APOE_s0` with `APOE_s1` |

Planted (instructor): APOE additive + APOE_s0×APOE_s1 + weaker COL4A1. Simulated WMH, not real data.

## Full GenNet CLI

Canonical repository: https://github.com/ArnovanHilten/GenNet

```bash
python GenNet.py convert  -g /path/to/plink -study_name mystudy -o ./processed_data/
python GenNet.py topology -type create_annovar_input -path ./processed_data/ -study_name mystudy
python GenNet.py topology -type create_gene_network  -path ./processed_data/ -study_name mystudy
python GenNet.py topology -type create_pathway_KEGG  -path ./processed_data/ -study_name mystudy

python GenNet.py train -path ./run/ -ID 17 -epochs 100 -L1 0.01 -problem_type classification
python GenNet.py train -path ./run/ -ID 18 -problem_type regression

python GenNet.py plot -ID 17 -type manhattan_relative_importance
python GenNet.py plot -ID 17 -type sunburst

python GenNet.py interpret -type get_weight_scores -resultpath results/GenNet_experiment_17_/
python GenNet.py interpret -type NID               -resultpath results/GenNet_experiment_17_/
python GenNet.py interpret -type RLIPP             -resultpath results/GenNet_experiment_17_/
python GenNet.py interpret -type DFIM              -resultpath results/GenNet_experiment_17_/
python GenNet.py interpret -type PathExplain       -resultpath results/GenNet_experiment_17_/
```

`subjects.csv` columns: `patient_id`, `labels`, `genotype_row`, `set`  
`set`: 1 = train, 2 = validation, 3 = test. Indices are 0-based.

Useful train flags: `-L1`, `-L1_act`, `-lr`, `-bs`, `-filters`, `-patience`, `-onehot`, `-hidden_activation`.

## Topology snippet

```text
layer0_node,layer0_name,layer1_node,layer1_name,layer2_node,layer2_name
0,APOE_s0,0,APOE,0,lipid_endocytosis
1,APOE_s1,0,APOE,0,lipid_endocytosis
```

Each row is one allowed path. You do not include the final phenotype node.

## Links

- Paper: https://www.nature.com/articles/s42003-021-02622-z
- A-to-Z Colab: https://colab.research.google.com/github/ArnovanHilten/GenNet/blob/master/examples/A_to_Z/GenNet_A_to_Z.ipynb
- ALIEN: https://www.roshchupkin.org/alien/
- Demo of the basic idea: https://tinyurl.com/y8hh8rul
