# GenNet workshop — Neurepiomics 2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshchupkin/gennet-neurepiomics-workshop-2026/blob/main/notebooks/01_gennet_in_one_hour.ipynb)

One-hour practical for **Neurepiomics 2026** (San Antonio, 5–7 October), in Wednesday’s session:

> **Genetic and Multiomic Analyses** · 11:45–15:00 · Center for Brain Health  
> Xueqiu Jian, Feiyang Zhao, Aniket Mishra, and Gennady Roshchupkin

GenNet is a neural network whose connections follow biology: variants connect to their genes, and genes connect to pathways. In this hour you train a small example, see how well it predicts, and read which genes and variant pairs it used.

## How to start

1. Open the slides: [GenNet introduction (PDF)](docs/GenNet_intro.pdf).
2. Open the practical with the button above. It loads the notebook in Google Colab. You do not upload a file.
3. In Colab, choose **File → Save a copy in Drive**.
4. Choose **Runtime → Change runtime type → CPU**. A GPU is not needed.
5. Choose **Runtime → Run all**. Training takes about a minute. Read the text in the notebook while it runs.

If the button does not open, use this link: [Launch the practical in Google Colab](https://colab.research.google.com/github/roshchupkin/gennet-neurepiomics-workshop-2026/blob/main/notebooks/01_gennet_in_one_hour.ipynb).

## During the hour

| When | What you do |
|------|-------------|
| First 5 minutes | Open Colab, set the runtime to CPU, and start the notebook. |
| Next 15 minutes | Follow the introductory slides. |
| Next 30 minutes | Train the model and read the plots: prediction, gene ranking, and variant-pair candidates. |
| Last 10 minutes | Questions, and how to try the same steps on your own data. |

The notebook uses a simulated cohort of 1,600 people and 192 variants. A few signals are built into that simulation so you can check whether the model finds them. Look at your own rankings in the notebook before you read that answer. The gene names are there to make the example easy to follow. The data are simulated for this session.

## After the workshop

- Notebook section 9: the same steps on your own genotype files.
- Cheat sheet: [`docs/CHEATSHEET.md`](docs/CHEATSHEET.md)
- Paper: [van Hilten et al., Communications Biology 2021](https://www.nature.com/articles/s42003-021-02622-z)
- GenNet code and the longer tutorial: [ArnovanHilten/GenNet](https://github.com/ArnovanHilten/GenNet) · [A-to-Z Colab](https://colab.research.google.com/github/ArnovanHilten/GenNet/blob/master/examples/A_to_Z/GenNet_A_to_Z.ipynb)

## For instructors

Session plan, answer key, and backup notebook: [`docs/INSTRUCTOR.md`](docs/INSTRUCTOR.md)
