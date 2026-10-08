# Membership Inference Attacks on Text-Only and Vision-Language Models

# Overview

This repository contains all code, data, reports, and submissions for the Membership Inference Attack (MIA) group project. It was completed for the COLX 531 and COLX 585 classes as part of the Master of Data Science - Computational Linguistics program at UBC. All credit for project origins to Jian Zhu at UBC. This is Rachelle's copy of the project, completed by Jennifer Flake, Rachelle De Jager, Yucai Zhong, and Christina McCallum.

Membership inference attacks query a model to determine whether a given data point was in its training data, making them a key tool for auditing privacy risks. We applied MIA methods to fine-tuned Large Language Models (LLMs) and Vision-Language Models (VLMs) in a grey-box setting, extracting features from the models and training classifiers to predict membership.

The first phase (COLX 531) attacked language models fine-tuned on clinical discharge notes. The second phase (COLX 585) attacked a VLM fine-tuned on image-caption pairs. Our main contributions were extending neighbourhood comparison from LLMs to VLMs, combining recently published feature families in new ways, and comparing feature sets across the two settings.

---
# Datasets and Models

**Text-only (Milestones 1–3).** We attacked [`UBC-SLIME/colx_531_smollm2-135m`](https://huggingface.co/UBC-SLIME/colx_531_smollm2-135m) and [`UBC-SLIME/colx_531_smollm2-360m`](https://huggingface.co/UBC-SLIME/colx_531_smollm2-360m), two SmolLM2 models fine-tuned on patient discharge notes from the [`UBC-SLIME/colx_531_group_project`](https://huggingface.co/datasets/UBC-SLIME/colx_531_group_project) dataset. The data contains 50,000 training, 10,000 validation, and 15,000 unlabelled test examples, with balanced member and non-member classes.

**Multimodal (Milestones 4–6).** We attacked [`UBC-SLIME/colx_585_vlm`](https://huggingface.co/UBC-SLIME/colx_585_vlm), a SmolVLM model fine-tuned on image-caption pairs, using [`HuggingFaceTB/SmolVLM-Base`](https://huggingface.co/HuggingFaceTB/SmolVLM-Base) as the reference model. The [`UBC-SLIME/colx585_group_project_data`](https://huggingface.co/datasets/UBC-SLIME/colx585_group_project_data) dataset contains 6,000 training, 1,200 validation, and 6,000 test examples. A sample counts as a member only if the model saw both the image and the caption, so the non-member class mixes image-only, text-only, and unseen pairs, at a 1:3 member-to-non-member ratio.

In both settings, text lengths were consistent across members and non-members, so length bias did not drive the classifiers.

---

# Results

Our best text-only attack, a **weighted soft-voting ensemble** built on HT-MIA and log-likelihood ratio features, achieved a **Kaggle test AUC of 0.951** (validation AUC 0.949, TPR@FPR=0.1 of 0.831). This improved substantially on our 18-feature logistic regression baseline (test AUC 0.893).

Our best multimodal attack, a **global XGBoost model** on a filtered combination of temperature, Rényi, and image-perturbation features, achieved a **Kaggle test AUC of 0.836** (validation AUC 0.851). This improved on our VLM baseline (test AUC 0.769).

Across both settings, **comparative features outperformed absolute ones**. Signals comparing the fine-tuned model to a reference model (or the 360M model to the 135M model) carried far more membership information than raw loss or perplexity alone.

We also found that **feature selection beat feature accumulation**. Through Milestones 2–5, each new feature family added only small gains, but systematically removing weak and redundant features in Milestone 6 produced the largest single-milestone jump in the VLM setting.

Multimodal MIA proved considerably harder than text-only MIA. Many token-probability features that worked well on clinical text added little for the VLM, and image-based features such as corruption and perturbation contributed limited signal, partly because compute constraints (roughly 12 hours of training for image features) limited how fully they could be implemented.

For full methodology, experimentation details, and limitations, see the [final report](milestone7/TCOB_Milestone_7.pdf) and [presentation slides](milestone7/Membership_Inference_Attack_pres_slides.pdf).

---
## Quickstart

Clone the repo and set up the environment:

```bash
git clone https://github.com/rdjg34/membership_inference_attack.git
cd membership_inference_attack

# Create and activate the conda environment
conda env create -f environment.yml
conda activate tcob
```

The target models and datasets are hosted on Hugging Face and are downloaded when the code first runs. Feature extraction requires a GPU for reasonable runtimes, and the image-based VLM features in particular are compute-intensive.

---

## Repository Structure

- **`src/`** — Source code for the text-only attacks, including EDA, the baseline notebook, and feature extraction and classifier code for each experiment (COLX 531)
- **`milestone1/` – `milestone3/`** — Milestone reports, data inspection, and Kaggle submission files for the text-only phase (COLX 531)
- **`milestone4/` – `milestone6/`** — Source code, extracted feature data, outputs, and reports for the VLM phase (COLX 585)
- **`milestone7/`** — Final report and presentation slides (COLX 585)
- **`environment.yml`** — Conda environment specification

---

## Milestone Navigation

### Text-Only MIA (COLX 531)

| Milestone | Focus | Explore |
|-----------|-------|---------|
| Milestone 1 | Data inspection, teamwork contract, and baseline with 18 confidence features (LogReg, Random Forest) | [`src/COLX_531_Lab3.ipynb`](src/COLX_531_Lab3.ipynb), [`src/Lab3_EDA.ipynb`](src/Lab3_EDA.ipynb), [`milestone1/`](milestone1/) |
| Milestone 2 | Min-K% features, cross-model delta and ratio features, and partial SPV-MIA | [`src/min_k_xgboost/`](src/min_k_xgboost/), [`src/cross_model_delta_attack.py`](src/cross_model_delta_attack.py), [`src/partial_spv_min_k_xgboost/`](src/partial_spv_min_k_xgboost/) |
| Milestone 3 | Min-K%++, HT-MIA, log-likelihood ratio, and zlib features with a weighted soft-voting ensemble | [`src/model5/`](src/model5/), [`src/model6/`](src/model6/) |

### Vision-Language MIA (COLX 585)

| Milestone | Focus | Explore |
|-----------|-------|---------|
| Milestone 4 | VLM data inspection and baseline with caption-level and target-reference comparative features | [`milestone4/`](milestone4/) |
| Milestone 5 | Temperature sensitivity, Rényi entropy, and image corruption features | [`milestone5/`](milestone5/) |
| Milestone 6 | Feature merging and selection, neighbourhood comparison, PC-MMIA, source routing, and neural stacking | [`milestone6/`](milestone6/) |
| Milestone 7 | Final report and presentation | [`milestone7/`](milestone7/) |

For full methodology and results, see the [final report](milestone7/TCOB_Milestone_7.pdf).

---
