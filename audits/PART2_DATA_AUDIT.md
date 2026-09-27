# Part 2 Data Audit Report: Inria Aerial Image Labeling Dataset

**Assessment:** UNLV PhD Technical Assessment — Dr. Sohn's Research Lab  
**Domain:** Part 2: Building Footprint Semantic Segmentation  
**Audit Date:** September 26, 2026  
**Auditor:** PhD Technical Candidate

---

## 1. Dataset Source & Metadata

* **Dataset Name:** Inria Aerial Image Labeling Dataset
* **Official Primary Source:** [https://project.inria.fr/aerialimagelabeling/](https://project.inria.fr/aerialimagelabeling/)
* **Access Protocol & Mirror Used:**
  - Official Direct Web Directory (`files.inria.fr`): Requires web form submission / return HTTP 403 on direct scripts.
  - Verified Open Academic Mirror: Hugging Face (`blanchon/INRIA-Aerial-Image-Labeling`)
* **Download Date:** September 26, 2026
* **Dataset Version:** Official Release (0.3m spatial resolution RGB aerial orthorectified imagery).
* **Ground Truth Availability:** Ground truth masks are available for the 180 training tiles. (Test set masks are withheld by INRIA for official competition benchmarking).

---

## 2. Dataset Inventory Table

| Region / City | Train Image Tiles | Ground-Truth Masks | Image Dimensions | Mask Dimensions | Image Format | Mask Format |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Austin** | 36 | 36 | 5000 × 5000 px (RGB) | 5000 × 5000 px (1-ch) | `.tif` (GeoTIFF) | `.tif` (TIFF) |
| **Chicago** | 36 | 36 | 5000 × 5000 px (RGB) | 5000 × 5000 px (1-ch) | `.tif` (GeoTIFF) | `.tif` (TIFF) |
| **Kitsap** | 36 | 36 | 5000 × 5000 px (RGB) | 5000 × 5000 px (1-ch) | `.tif` (GeoTIFF) | `.tif` (TIFF) |
| **Tyrol-w** | 36 | 36 | 5000 × 5000 px (RGB) | 5000 × 5000 px (1-ch) | `.tif` (GeoTIFF) | `.tif` (TIFF) |
| **Vienna** | 36 | 36 | 5000 × 5000 px (RGB) | 5000 × 5000 px (1-ch) | `.tif` (GeoTIFF) | `.tif` (TIFF) |
| **Total (Training Set)** | **180** | **180** | **5000 × 5000 × 3** | **5000 × 5000 × 1** | **.tif** | **.tif** |

*Note: In addition, 180 evaluation test tiles (36 per city: Bellingham, Bloomington, Inverness, San Francisco, Tyrol-e) are provided without ground-truth masks for official benchmark evaluation.*

---

## 3. Geographic Regions & Urban Morphology

The dataset spans 5 diverse geographic regions exhibiting distinct urban and natural characteristics:
1. **Austin (Texas, USA):** Low-to-medium density suburban residential layouts with large roofs, swimming pools, and highway grids.
2. **Chicago (Illinois, USA):** High-density urban grid architecture with complex commercial rooflines, alleyways, and street shadows.
3. **Kitsap County (Washington, USA):** Rural and heavily forested terrain with highly sparse, isolated single-family structures.
4. **Tyrol-w (Austrian Alps):** Alpine mountainous landscape featuring steep topography, agricultural buildings, and irregular village footprints.
5. **Vienna (Austria):** Dense historical European city center with connected urban blocks, complex roof geometries, and inner courtyards.

---

## 4. Image & Mask Specifications Verification

* **Spatial Resolution:** $0.3\text{ m / pixel}$ (Ground Sampling Distance / GSD).
* **Tile Coverage Area:** $1.5\text{ km} \times 1.5\text{ km} = 2.25\text{ km}^2$ per $5000 \times 5000$ tile.
* **Total Training Area Coverage:** $180 \times 2.25\text{ km}^2 = 405\text{ km}^2$.
* **Color Channels:** 3 Channels (Red, Green, Blue uint8 values $[0, 255]$).
* **Mask Specifications:** 1 Channel binary mask uint8 values.

---

## 5. Mask-Value Verification & Building Pixel Statistics

Empirical audit performed on representative sample tiles (1 tile per city):

| City / Region | Sample Tile ID | Image Shape | Mask Shape | Unique Mask Values | Building Coverage (%) | Background Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Austin** | `austin1` | `(5000, 5000, 3)` | `(5000, 5000)` | `[0, 255]` | **14.76%** | **85.24%** |
| **Chicago** | `chicago1` | `(5000, 5000, 3)` | `(5000, 5000)` | `[0, 255]` | **14.36%** | **85.64%** |
| **Kitsap** | `kitsap1` | `(5000, 5000, 3)` | `(5000, 5000)` | `[0, 255]` | **0.20%** | **99.80%** |
| **Tyrol-w** | `tyrol-w1` | `(5000, 5000, 3)` | `(5000, 5000)` | `[0, 255]` | **6.98%** | **93.02%** |
| **Vienna** | `vienna1` | `(5000, 5000, 3)` | `(5000, 5000)` | `[0, 255]` | **18.04%** | **81.96%** |

### Verified Properties
1. **Binary Encoding:** Masks contain **strictly 0 (Background)** and **255 (Building Footprint)** values. No intermediate grayscale artifact values exist.
2. **Class Imbalance:** Background pixels dominate aerial tiles ($81.96\%$ to $99.80\%$ background), demonstrating significant spatial class imbalance across regions (especially in rural Kitsap County).

---

## 6. Environment & GPU Hardware Verification

```text
================================================================================
ENVIRONMENT SPECIFICATIONS
================================================================================
Python Version      : 3.14.6
PyTorch Version     : 2.14.0+cpu
Torchvision Version : 0.29.0+cpu
CUDA Available      : False (CPU Execution Mode Active)
Segmentation Models : segmentation-models-pytorch 0.5.0 installed
Augmentations Lib   : albumentations 2.0.8 installed
Image Processing Lib: OpenCV 5.0.0.93 / PIL Pillow 12.3.0 installed
================================================================================
```

---

## 7. Visual Sanity Check Results

Qualitative inspection figure generated and saved:
[`outputs/figures/fig5_aerial_mask_sanity_check.png`](file:///d:/UNLV_PhD_Technical_Assessment/outputs/figures/fig5_aerial_mask_sanity_check.png)

### Visual Inspection Findings
1. **Exact Boundary Alignment:** Overlaying ground-truth masks on aerial RGB imagery confirms precise spatial alignment of building footprints.
2. **Morphological Variation:** Confirms high intra-class variance between small rural cabins (Kitsap), suburban housing blocks (Austin), dense city grids (Chicago/Vienna), and alpine farms (Tyrol).

---

## 8. Dataset Access & Computational Limitations

1. **Tile Memory Footprint:** Full $5000 \times 5000$ tiles require $\sim 75\text{ MB}$ uncompressed RAM per RGB image and $\sim 25\text{ MB}$ per mask. Full tiles cannot be fed directly into deep learning networks due to memory constraints.
2. **Patch Extraction Requirement:** Training requires uniform sub-patch extraction (e.g., $512 \times 512$ pixel patches).
3. **Spatial Data Leakage Risk:** Standard random patch splitting across neighboring crops would result in severe spatial leakage (overlapping buildings in both train and validation splits).

---

## 9. Recommended Next Step

**Phase 2 Step 2:** Design and implement a **Spatially Leakage-Resistant Data Partitioning Strategy**:
1. Partition dataset at the **Full Tile Level** (5000×5000) or **City Level** prior to patch creation.
2. Formulate a dual validation scheme:
   - **City-Level Holdout (Pure Geographic Generalization):** Hold out 1 full city (e.g. Vienna or Chicago) exclusively for evaluation.
   - **Intra-City Tile Grid Buffer Split:** Hold out 20% of full tiles per city with a spatial buffer zone.
3. Generate uniform $512 \times 512$ patches strictly respecting spatial split boundaries.

---

### End of Part 2 Data Audit Report
