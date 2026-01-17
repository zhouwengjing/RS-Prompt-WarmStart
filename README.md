# Initialization Matters: Deterministic Warm-Start Prompt Learning for Remote Sensing Recognition

## 📖 Introduction

This repository contains the official implementation of our paper.

In this work, we investigate the **Prompt Learning** paradigm for remote sensing image recognition. While Context Optimization (CoOp) has shown success in natural images, we identify a severe **"Cold Start" instability** when applying it to remote sensing domains under standard random initialization.

To address this, we **systematically validate** a **Semantically Guided "Warm Start" Strategy**. By initializing context vectors with semantic priors (e.g., "a photo of a"), we position the model within a valid semantic manifold. Our experiments demonstrate that this strategy is **critical** for remote sensing tasks, ensuring **deterministic convergence** and **robust performance** comparable to fully supervised baselines, without the risk of optimization collapse.

**Key Features:**

- 🚀 **High Performance:** Boosts EuroSAT accuracy from **43.25% (Zero-shot)** to **91.85%**, achieving performance competitive with well-tuned baselines.

  🛡️ **Stability & Robustness:** Identifies and eliminates the **"Cold Start" failure** observed in high-variance random initialization.

  💾 **Parameter Efficient:** Requires only **~32 KB** of storage for task-specific parameters, making it ideal for bandwidth-constrained edge deployment.

------

## 🛠️ Environment Setup

We recommend using **Anaconda** to manage the environment.

### 1. Create Environment

```bash
# Create a virtual environment with Python 3.10
conda create -n multimodal-env python=3.10
conda activate multimodal-env
```

### 2. Install Dependencies

Ensure you have a GPU available (check with `nvidia-smi`).

```bash
# Install other requirements
pip install -r requirements.txt
```

------

## 📂 Data Preparation

Please download the datasets and organize them into the `data/` directory as follows.

**Directory Structure:**

Plaintext

```
RS-Prompt-WarmStart/
├── data/
│   ├── EuroSAT/
│   │   └── 2750/              <-- Contains 27,000 images (.jpg/.png)
│   ├── UCMerced_LandUse/
│   │   └── Images/            <-- Contains 21 subfolders (.tif)
│   └── NWPU-RESISC45/         <-- Contains 45 subfolders (.jpg)
├── weights/
├── tools/
└── experiments/
```

### Download Links

- **EuroSAT (RGB):** [GitHub](https://github.com/phelber/eurosat) | [Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
- **UC Merced (UCM):** [Official Site](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
- **NWPU-RESISC45:** [TensorFlow Catalog](https://tensorflow.google.cn/datasets/catalog/resisc45)

**Data Splits (Global Seed = 42):**

- **EuroSAT:** 80% Training / 20% Testing
- **UCM:** 50% Training / 50% Testing
- **RESISC45:** 80% Training / 20% Testing

------

## 🤖 Model Preparation

We use the pre-trained `CLIP-ViT-B/32` as our backbone.

Download Weights:

Run the following script to automatically download and save the CLIP model to weights/models/.

```bash
python weights/download_clip-vit-base-patch32.py
```

------

## 🚀 Running Experiments

We provide scripts to reproduce all experiments in the paper. All results are deterministic with `seed=42`.

### A. Baseline (Zero-Shot)

Evaluate the direct zero-shot performance of CLIP on the three datasets.

```bash
# EuroSAT Zero-shot
python experiments/exp00_baseline/run.py --dataset eurosat

# UCM Zero-shot
python experiments/exp00_baseline/run.py --dataset ucm

# RESISC45 Zero-shot
python experiments/exp00_baseline/run.py --dataset resisc45
```

### B. Warm-Start Training (Main Results)

Train the learnable context vectors using our Warm Start strategy.

- **Note:** The backbone is frozen; only prompts are updated.
- **Output:** Best model weights will be saved to `weights/outputs/`.

```bash
# 1. EuroSAT (15 Epochs) -> Acc: ~91.85%
python experiments/exp01_eurosat_rgb/run.py

# 2. Train on UCM (20 Epochs) -> Acc: ~86.67%
python experiments/exp02_ucm/run.py

# 3. Train on RESISC45 (25 Epochs) -> Acc: ~85.71%
python experiments/exp03_resisc45/run.py
```

### C. Qualitative Analysis (Heatmap)

Generate the confusion matrix to visualize the model's behavior on EuroSAT.

- **Prerequisite:** Ensure `exp01` is finished and weights are saved.

```bash
python experiments/analysis_eurosat_heatmap/run.py
# The heatmap will be saved to pictures/eurosat_heatmap.png
```

### D. Ablation Study (Initialization Dynamics)

Reproduce the 4-line comparison plot (Figure 2 in the paper) to demonstrate the robustness of Warm Start vs. Random Initialization.

```bash
python experiments/analysis_eurosat_ablation/run.py
# The plot will be saved to pictures/ablation_initialization.png
```

### E. Cross-Dataset Transferability

Evaluate the generalization of prompts learned on EuroSAT when directly applied to UCM and RESISC45 (Limitations section).

```bash
python experiments/analysis_transferability/run.py
```

------

## 📊 Main Results

| **Dataset**  | **Method**            | **Accuracy (%)** | **Param Storage** |
| ------------ | --------------------- | ---------------- | ----------------- |
| **EuroSAT**  | Zero-Shot CLIP        | 43.25            | -                 |
|              | **Ours (Warm Start)** | **91.85**        | **32 KB**         |
| **UCM**      | Zero-Shot CLIP        | 59.38            | -                 |
|              | **Ours (Warm Start)** | **86.67**        | **32 KB**         |
| **RESISC45** | Zero-Shot CLIP        | 52.83            | -                 |
|              | **Ours (Warm Start)** | **85.71**        | **32 KB**         |

------

## 🖊️ Citation

If you find this code useful for your research, please consider citing our paper:

Code snippet

```
@article{Zhou2026Initialization,
  title={Initialization Matters: Deterministic Warm-Start Prompt Learning for Remote Sensing Recognition},
  author={Zhou, Wenjing},
  journal={Manuscript in preparation},
  year={2026}
}
```
