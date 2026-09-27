# UNLV PhD Technical Assessment
### Demand Forecasting & Aerial Building Footprint Segmentation

**Author:** Md Rahinur Rahman  
**Degree:** B.Sc. in Electrical and Electronic Engineering (EEE)  
**Department:** Department of Electrical and Electronic Engineering  
**Institution:** Bangladesh University of Engineering and Technology (BUET)  
**Specialization:** Communication and Signal Processing (CSP)  

---

## Research Overview

This repository contains the complete implementation, analytical codebase, and technical documentation for the **UNLV PhD Technical Assessment** prepared for Dr. Sohn's Research Lab in the Department of Civil and Environmental Engineering and Construction at the University of Nevada, Las Vegas (UNLV).

The assessment evaluates advanced empirical methods across two core data science and machine learning domains:
1. **Part 1: Capital Bikeshare Demand Analysis & Time-Series Forecasting** — Econometric modeling, micro-weather analysis, multicollinearity pruning, count-data overdispersion diagnostics, and zero-leakage hourly demand forecasting.
2. **Part 2: Inria Aerial Building Footprint Semantic Segmentation** — Deep convolutional U-Net architectures with ResNet-34 encoders, spatial city-level split design, composite loss formulation, and zero-leakage out-of-domain held-out test evaluation on Vienna, Austria.

Both tracks strictly implement task-appropriate temporal and geographic holdout protocols to guarantee empirical integrity.

---

## Assessment Components

The submission is organized into two primary executable Jupyter notebooks and a formal 6-page research technical memorandum:

- **[Part 1 Notebook: Bikeshare Demand Forecasting](notebooks/01_bikeshare_demand_forecasting.ipynb)** — Executable data ingestion, descriptive statistics, econometric OLS / Log-OLS / Negative Binomial regressions, VIF diagnostics, and time-series demand forecasting.
- **[Part 2 Notebook: Building Footprint Segmentation](notebooks/02_building_footprint_segmentation.ipynb)** — Executable aerial patch verification, PyTorch DataLoader pipeline, U-Net ResNet-34 model definition, local RTX 4050 checkpoint loading/evaluation, Vienna test set inference, qualitative error overlay mapping, and empirical failure analysis.
- **[Supplementary Colab Notebook](notebooks/colab_exploratory_run.ipynb)** — Preserved Google Colab Tesla T4 exploratory reference run.
- **[Full Research Summary PDF](reports/UNLV_PhD_Technical_Assessment_Research_Summary.pdf)** — Formal 6-page technical memorandum presenting statistical regression tables, forecasting performance, spatial split schema, quantitative test results, and error visual analysis.

---

## Part 1 — Capital Bikeshare Demand Analysis & Forecasting

### 1. Exploratory Data Analysis & Micro-Weather Dynamics
The dataset contains 17,379 hourly rental observations from Capital Bikeshare in Washington, D.C. (2011–2012). Disaggregating demand into user categories (`casual` vs. `registered`) reveals strong behavioral divergence:
- **Registered Commuters:** Display sharp bimodal commuting peaks at 08:00 and 17:00–18:00 on working days.
- **Casual Riders:** Display unimodal leisure curves peaking on weekend afternoons (12:00–16:00).
- **Environmental Suppression:** Temperature scales demand positively up to ~32°C before heat-penalty suppression occurs. Adverse weather conditions (rain, snow, thunderstorms) induce substantial reductions in ridership.

### 2. Econometric Regression & Variance Inflation Diagnostics
- **Multicollinearity Pruning:** Severe collinearity was identified between `temp` (VIF = 43.60) and `atemp` (VIF = 43.73). Pruning `atemp` resolved collinearity, reducing all remaining predictor VIFs to `< 1.50`.
- **Residual Diagnostics:** Breusch-Pagan test (`p = 1.28e-182`) confirmed persistent heteroskedasticity, and Jarque-Bera test (`p < 0.001`) confirmed non-normality, justifying robust standard errors.
- **Count-Data Overdispersion:** Overdispersion ratio `Var(Y) / E[Y] = 173.66 >> 1.0` confirmed severe overdispersion, motivating count-data models (Negative Binomial family).
- **Log-Likelihood Adjustments:** Jacobian-adjusted Log-OLS AIC reached `214,474.59`, incorporating the `sum log(y+1) = 79,504.38` Jacobian transformation term.

### 3. Time-Series Forecasting Setup
Evaluating time-series demand models requires strict leakage prevention:
- **Training Set:** Days 1 through 20 of each calendar month.
- **Test Set:** Days 21 through month-end.

---

## Part 2 — Inria Building Footprint Semantic Segmentation

### 1. Dataset & City-Level Spatial Split
The Inria Aerial Image Labeling dataset contains 180 orthorectified 30cm RGB tiles (5000×5000 pixels) across 5 urban regions. To prevent spatial autocorrelation leakage, tiles were partitioned exclusively at the city level before patch extraction (512×512 pixels):
- **Training Set (108 tiles / 8,748 patches):** Austin (36 tiles), Chicago (36 tiles), Kitsap (36 tiles).
- **Validation Set (36 tiles / 2,916 patches):** Tyrol-w (used exclusively for checkpoint selection).
- **Held-Out Test Set (36 tiles / 2,916 patches):** Vienna, Austria (strictly isolated until final evaluation).
- **Total Dataset Audit:** 180 tiles = **14,580 patches (512×512 px)**.

### 2. Model Architecture & Loss Formulation
- **Architecture:** U-Net semantic segmentation network with an ImageNet-pretrained ResNet-34 encoder backbone. Skip connections preserve spatial perimeter detail while feature extraction leverages deep pretrained representations.
- **Composite Loss Function:** Combines Binary Cross-Entropy with Logits and Soft Dice Loss to balance pixel classification supervision and region overlap under class imbalance:
  $$\mathcal{L}_{\text{total}} = 1.0 \times \mathcal{L}_{\text{BCE}} + 1.0 \times \mathcal{L}_{\text{Dice}}$$
- **Boundary Quality Consideration:** Boundary quality was assessed through qualitative error overlay visualization and empirical failure analysis. A dedicated boundary-aware loss was not used in the reported training run; boundary-aware losses are identified as a potential future improvement.

---

## GPU Computational Workflow

```
Data Ingestion & Verification
  │
  ├──► Part 1: EDA ──► VIF Pruning ──► Log-OLS / NegBin ──► Days 1-20/21-End Split ──► Forecasting
  │
  └──► Part 2: Spatial Split ──► 512x512 Patch Extraction ──► DataLoader Setup
         │
         ├──► Google Colab Tesla T4 Run (Exploratory Reference, Epoch 22, Peak Val IoU 0.7174 @ Ep 15)
         │
         └──► Local NVIDIA RTX 4050 Run (Reproducible Final Checkpoint, Stopped @ Ep 8, Best Ep 5)
                │
                └──► Checkpoint Selection (Val IoU: 0.706835, Dice: 0.8282)
                       │
                       └──► Held-Out Vienna Test Evaluation (2,916 patches)
                              │
                              └──► Quantitative Metrics + Qualitative Error Maps + Failure Case Analysis
```

### Two-GPU Execution Provenance
1. **Google Colab Tesla T4 Exploratory Run (Reference Baseline):** Executed through Epoch 22 on a Tesla T4 GPU, reaching a peak Validation IoU of `0.7174` at Epoch 15. This run served as an exploratory reference baseline and was **not** used for the final Vienna test evaluation.
2. **Local NVIDIA RTX 4050 Run (Final Evaluation Model):** Trained locally on an **NVIDIA GeForce RTX 4050 Laptop GPU** (PyTorch 2.11.0+cu128, CUDA 12.8, Albumentations 2.0.8, segmentation-models-pytorch 0.5.0, Python 3.12.14). Training reached **Epoch 8** before manual termination due to execution time constraints. The optimal checkpoint was saved at **Epoch 5** with **Validation IoU: 0.706835** and **Validation Dice: 0.8282**. All reported final test evaluations were performed using this local Epoch-5 checkpoint.

---

## Final Quantitative Results

### Part 1: Bikeshare Demand Forecasting (Days 21–End Test Set)

| Forecasting Model Specification | RMSLE | MAE | R² Score |
|:---|:---:|:---:|:---:|
| **Pure Exogenous / Zero Target-Lag Model** | **0.4050** | **32.332** | **0.9093** |
| **Day-Ahead Rolling Model (24-Hour Target Lag)** | **0.4004** | **34.367** | **0.9018** |

### Part 2: Vienna Held-Out Test Set Segmentation Performance (N = 2,916 Patches)

| Metric Name | Held-Out Test Value |
|:---|:---:|
| **Overall Test IoU** | **0.6644** |
| **Dice / F1 Score** | **0.7734** |
| **Precision** | **0.7713** |
| **Recall** | **0.8222** |
| **Pixel Accuracy** | **0.8966** |
| **Evaluated Test Patches** | **2,916** |

### Part 2: Building Coverage Stratification on Vienna

| Building Coverage Stratum | IoU | Dice (F1) | Precision | Recall | Pixel Accuracy | Patch Count (n) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Low Coverage (< 5%)** | 0.4682 | 0.5275 | 0.5720 | 0.7043 | 0.9840 | 430 |
| **Moderate Coverage (5–25%)** | 0.6579 | 0.7823 | 0.7823 | 0.8077 | 0.9327 | 829 |
| **High Coverage (> 25%)** | **0.7186** | **0.8327** | **0.8175** | **0.8601** | 0.8559 | 1,657 |

---

## Qualitative Analysis & Empirical Failure Breakdown

### Qualitative Inferences
Representative 512×512 held-out test predictions ([Figure 4](outputs/figures/fig9_segmentation_predictions.png)) illustrate successful building extraction alongside error map overlays:
- **True Positives (TP):** Green
- **False Positives (FP):** Red
- **False Negatives (FN):** Blue

### Failure Modes Analysis
Categorization of representative lowest-IoU failure cases on Vienna identified three primary failure modes:
1. **False Positives (FP):** Over-segmentation on spectrally similar paved courtyards, bright parking structures, solar panels, and shadow boundaries.
2. **Missed Small Buildings (FN):** Under-segmentation of low-contrast residential structures or annexes matching surrounding terrain.
3. **Boundary Alignment Errors:** Spatial misalignment along complex angled roof perimeters.

---

## Reproducibility & Environment Setup

### Verified Software Stack
- **OS:** Windows 11 (Conda environment `unlv_gpu`)
- **Python:** 3.12.14
- **PyTorch:** 2.11.0+cu128
- **CUDA Runtime:** 12.8
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- **Computer Vision Libraries:** Albumentations 2.0.8, segmentation-models-pytorch 0.5.0
- **Random Seed:** Global seed `42` set across Python `random`, `numpy`, `torch`, and `torch.cuda`.

### Local Checkpoint Note
The final local Epoch-5 checkpoint (`best_unet_resnet34.pth`, ~293.5 MB) is preserved locally as a reproducibility artifact but is excluded from Git tracking to maintain repository cleanliness and avoid exceeding GitHub file size limits.

---

## Repository Structure

```
github_repo/
├── README.md                                # Professor-level technical documentation
├── LICENSE                                  # MIT License & dataset terms notice
├── .gitignore                               # Git exclusion rules
├── requirements.txt                         # Python environment dependencies
│
├── notebooks/
│   ├── 01_bikeshare_demand_forecasting.ipynb# Part 1: Bikeshare analysis & forecasting
│   ├── 02_building_footprint_segmentation.ipynb# Part 2: Building footprint segmentation
│   └── colab_exploratory_run.ipynb          # Supplementary Colab T4 exploratory notebook
│
├── src/
│   ├── data/                                # Dataset & patch extraction loaders
│   ├── models/                              # U-Net architecture, losses & forecasting
│   └── utils/                               # Plotting, metrics & split utilities
│
├── reports/
│   └── UNLV_PhD_Technical_Assessment_Research_Summary.pdf # 6-page Research Summary PDF
│
├── outputs/
│   ├── figures/                             # Generated plots & visualization maps
│   └── tables/                              # Result CSVs & evaluation metrics
│
├── audits/                                  # Audit documents detailing split logic & design
└── docs/                                    # Plan & implementation status documentation
```

---

## Limitations & Future Directions

1. **Graph Neural Networks for Bikeshare:** Incorporating Spatio-Temporal Graph Neural Networks (ST-GNNs) could model spatial interactions across bike station networks.
2. **Boundary-Aware Loss Functions:** Integrating active contour or Hausdorff distance loss terms may refine building boundary alignment on sparse rural structures.
3. **Multi-Scale Architectures:** High-Resolution Nets (HRNet) or Feature Pyramid Networks (FPN) could improve detection of small residential buildings.

---

## Author & Contact

**Md Rahinur Rahman**  
B.Sc. in Electrical and Electronic Engineering (EEE)  
Department of Electrical and Electronic Engineering  
Bangladesh University of Engineering and Technology (BUET)  
Specialization: Communication and Signal Processing (CSP)  
GitHub: [https://github.com/rahin-rahman](https://github.com/rahin-rahman)
