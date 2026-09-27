# Part 2: 512x512 Patch Extraction & Processing Pipeline Report

**Project:** UNLV PhD Technical Assessment — Dr. Sohn's Research Lab  
**Author:** UNLV PhD Technical Candidate  
**Pipeline Status:** Complete, Audited, and Verified (`STATUS: [PASS]`)  
**Patch Manifest:** `outputs/tables/part2_patch_manifest.csv`  
**Quality Figure:** `outputs/figures/fig7_patch_samples.png`

---

## 1. Executive Summary & Audit Correction

Following a rigorous methodological audit, the 512x512 aerial patch extraction pipeline was expanded from the preliminary 10-tile representative subset (810 patches) to encompass **all 180 full $5000 \times 5000$ aerial tiles** in the Inria Aerial Image Labeling Dataset.

- **Total Tiles Processed:** **180 source tiles** ($100\%$ of dataset).
- **Patch Extraction Geometry:** $512 \times 512 \text{ px}$ patch size, stride $= 512 \text{ px}$ (Non-overlapping grid).
- **Patches per Tile:** $\lfloor 5000 / 512 \rfloor = 9 \text{ rows} \times 9 \text{ columns} = \mathbf{81 \text{ patches per tile}}$.
- **Total Patches Extracted:** $180 \text{ tiles} \times 81 \text{ patches} = \mathbf{14,580 \text{ patches}}$.
- **Spatial Autocorrelation Guard:** City-level partitioning strictly isolates cities at the source tile level **before** patch extraction.

---

## 2. Quantitative Split & Geographic Distribution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PATCH DISTRIBUTION BY CITY & SPLIT                    │
├────────────┬─────────┬──────────────┬──────────────┬────────────────────────┤
│ Split      │ City    │ Source Tiles │ Patch Count  │ Total Pixels           │
├────────────┼─────────┼──────────────┼──────────────┼────────────────────────┤
│ Train      │ Austin  │ 36 tiles     │ 2,916 patches│ 764,411,904 px         │
│ Train      │ Chicago │ 36 tiles     │ 2,916 patches│ 764,411,904 px         │
│ Train      │ Kitsap  │ 36 tiles     │ 2,916 patches│ 764,411,904 px         │
├────────────┼─────────┼──────────────┼──────────────┼────────────────────────┤
│ SUB-TOTAL  │ TRAIN   │ 108 tiles    │ 8,748 patches│ 2,293,235,712 px       │
├────────────┼─────────┼──────────────┼──────────────┼────────────────────────┤
│ Validation │ Tyrol-w │ 36 tiles     │ 2,916 patches│ 764,411,904 px         │
├────────────┼─────────┼──────────────┼──────────────┼────────────────────────┤
│ Test       │ Vienna  │ 36 tiles     │ 2,916 patches│ 764,411,904 px         │
├────────────┼─────────┼──────────────┼──────────────┼────────────────────────┤
│ TOTAL      │ ALL     │ 180 tiles    │14,580 patches│ 3,822,059,520 px       │
└────────────┴─────────┴──────────────┴──────────────┴────────────────────────┘
```

---

## 3. Patch Extraction Geometry & Border Handling

1. **Grid Calculation:** For each $5000 \times 5000$ tile:
   $$\text{Patches per axis} = \lfloor 5000 / 512 \rfloor = 9$$
   $$\text{Total Grid Patches} = 9 \times 9 = 81$$
2. **Excluded Edge Strip:**
   $$\text{Excluded Pixels} = 5000 - (9 \times 512) = 392\text{ px}$$
   - A $392\text{ pixel}$ strip along the right and bottom edges of each $5000 \times 5000$ tile is excluded to prevent partial/padded patch distortions.
3. **No Zero-Padding:** Every extracted patch is a native $512 \times 512 \times 3$ RGB tile and $512 \times 512$ binary mask without synthetic padding.

---

## 4. Dataset Class Imbalance Statistics (All 14,580 Patches)

- **Total Dataset Pixels:** $14,580 \times 512 \times 512 = 3,822,059,520 \text{ pixels}$.
- **Building Pixels:** $605,375,617 \text{ pixels}$ (**$15.84\%$**).
- **Background Pixels:** $3,216,683,903 \text{ pixels}$ (**$84.16\%$**).

### Building Coverage Stratification:
- **Low Coverage ($<5\%$ buildings):** $5,379 \text{ patches}$ (**$36.89\%$**)
- **Moderate Coverage ($5-25\%$ buildings):** $5,527 \text{ patches}$ (**$37.91\%$**)
- **High Coverage ($>25\%$ buildings):** $3,674 \text{ patches}$ (**$25.20\%$**)

---

## 5. Automated Patch Integrity Audit (`check_patch_integrity`)

An automated verification script scanned all 14,580 image patches and 14,580 ground-truth mask patches on disk:

```text
================================================================================
AUTOMATED PATCH INTEGRITY AUDIT — STATUS: [PASS]
================================================================================
  - Check 1_patch_count_14580: [PASSED] (14,580 total patches in manifest)
  - Check 2_files_exist:        [PASSED] (29,160 files exist, 0 missing/corrupt)
  - Check 3_shapes_512x512:     [PASSED] (Images: 512x512x3, Masks: 512x512)
  - Check 4_binary_mask_values: [PASSED] (Mask values strictly in {0, 255})
  - Check 5_every_tile_has_81:  [PASSED] (All 180 tiles yield exactly 81 patches)
================================================================================
```

---

## 6. Representative Patch Inspection Figure

The quality inspection figure [`outputs/figures/fig7_patch_samples.png`](file:///d:/UNLV_PhD_Technical_Assessment/outputs/figures/fig7_patch_samples.png) illustrates 5 representative patches (RGB Image, Ground Truth Mask, Footprint Boundary Overlay) extracted across Austin, Chicago, Kitsap, Tyrol-w, and Vienna.
