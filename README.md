# XAI Anti-Forensics

MPhil thesis implementation — Enhancing Robustness and Transparency in
Pixel-Level Anti-Forensics: An Explainable AI Approach to Adversarial
Forgery Evasion in Digital Image Forensics.

## Structure
- `data/` — raw and processed datasets (raw/ excluded from Git, lives on Drive)
- `models/` — saved model checkpoints (excluded from Git, lives on Drive)
- `figures/` — all thesis figures (PNG, 300 DPI)
- `results/` — JSON results files from each notebook
- `notebooks/` — numbered Colab notebooks
- `src/` — reusable Python modules (concealer, attribution module,
  spatial-concentration detector, suppression module)

## Pipeline
00 — GitHub/Drive setup (this notebook)
01 — Environment, datasets (Columbia, COVERAGE, CASIA v1/v2, IMD2020)
02 — Concealer (SEAR reproduction)
03 — Attribution module (perturbation map + supervisor Grad-CAM)
04 — Spatial-concentration detector
05 — Region-aware suppression
06 — Full pipeline integration
07 — Baselines (FGSM, BIM, MIM, AdvGAN, SEAR vs SATFL/Mantra-Net/SPAN/ANSM/ForensicsSAM)
08 — Benchmark evaluation (white-box, black-box, retrained-defense, foundation-model-defense)
09 — Ablation study
10 — Final results and figures
