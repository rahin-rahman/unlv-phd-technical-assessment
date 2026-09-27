# Step 6 Patch Extraction Audit & Reconciliation Report (`PART2_PATCH_AUDIT.md`)

**Project:** UNLV PhD Technical Assessment — Dr. Sohn's Research Lab  
**Author:** UNLV PhD Technical Candidate  
**Audit Date:** September 26, 2026  
**Audit Target:** Step 6 Patch Extraction Pipeline & Data Artifacts  
**Final Audit Status:** **`STATUS: [PASS]`**

---

## 1. Original Inconsistency

During a rigorous audit of the preliminary Step 6 report (`PART2_PATCH_PIPELINE.md`), a mathematical inconsistency was identified:

- **Source Dataset:** $180$ labeled $5000 \times 5000$ aerial tiles across 5 cities.
- **Extraction Geometry:** Patch size $= 512 \times 512 \text{ px}$, Stride $= 512 \text{ px}$ (Non-overlapping grid).
- **Theoretical Calculation:** 
  $$\text{Patches per axis} = \lfloor 5000 / 512 \rfloor = 9$$
  $$\text{Patches per tile} = 9 \times 9 = 81 \text{ patches}$$
  $$\text{Total Expected Patches} = 180 \text{ tiles} \times 81 \text{ patches} = \mathbf{14,580 \text{ patches}}$$
- **Reported Preliminary Count:** Only $810 \text{ total patches}$ ($486$ train, $162$ validation, $162$ test).

This represented a 18x discrepancy between full theoretical dataset coverage and the preliminary extracted patch count.

---

## 2. Root Cause Analysis

Inspection of `src/data/extract_aerial_patches.py` identified the exact root cause:

1. **Hard-Coded Tile Filtering:** Lines 28–34 in `src/data/extract_aerial_patches.py` defined a hard-coded dictionary `TARGET_TILES_PER_CITY`:
   ```python
   TARGET_TILES_PER_CITY = {
       'Austin': ['austin1', 'austin2'],
       'Chicago': ['chicago1', 'chicago2'],
       'Kitsap': ['kitsap1', 'kitsap2'],
       'Tyrol-w': ['tyrol-w1', 'tyrol-w2'],
       'Vienna': ['vienna1', 'vienna2']
   }
   ```
2. **Sub-Selection Logic:** Lines 75–76 filtered the full spatial manifest (`df_manifest`) to ONLY these 2 tiles per city ($10 \text{ tiles total}$):
   ```python
   df_targets = df_manifest[df_manifest['tile_id'].isin(target_tile_ids)].copy()
   ```
3. **Execution Rationale:** The preliminary script restricted extraction to 2 tiles per city to allow fast local prototyping without un-accelerated CPU disk/memory overload.
4. **Audit Conclusion:** Retaining a sampled subset of 10 tiles out of 180 is not methodologically defensible for the PhD assessment assessment when the full dataset of 180 tiles can be extracted.

---

## 3. Actual Extraction Logic & Geometry

All 180 source tiles in the Inria Aerial Image Labeling Dataset were verified to possess identical $5000 \times 5000$ pixel dimensions.

- **Tile Dimensions:** $5000 \text{ px} \times 5000 \text{ px} \times 3 \text{ channels}$ (RGB).
- **Patch Dimensions:** $512 \text{ px} \times 512 \text{ px} \times 3 \text{ channels}$.
- **Grid Layout:** 
  - Rows: $r \in \{0, 1, 2, 3, 4, 5, 6, 7, 8\}$ (9 rows, starting at $y = 0, 512, 1024, \dots, 4096$).
  - Columns: $c \in \{0, 1, 2, 3, 4, 5, 6, 7, 8\}$ (9 columns, starting at $x = 0, 512, 1024, \dots, 4096$).
- **Excluded Boundary Strip:**
  $$\text{Right Edge Excluded} = 5000 - (9 \times 512) = 392 \text{ px}$$
  $$\text{Bottom Edge Excluded} = 5000 - (9 \times 512) = 392 \text{ px}$$
  - Zero-padding is omitted. No synthetic pixels are created.

---

## 4. Corrected Patch Counts & Reconciliation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RECONCILED DATASET PATCH COUNTS                        │
├────────────┬─────────┬──────────────┬──────────────┬────────────────────────┤
│ Split      │ City    │ Tile Count   │ Patch Formula│ Total Patches          │
├────────────┼─────────┼──────────────┼──────────────┼────────────────────────┤
│ Train      │ Austin  │ 36 tiles     │ 36 × 81      │ 2,916 patches          │
│ Train      │ Chicago │ 36 tiles     │ 36 × 81      │ 2,916 patches          │
│ Train      │ Kitsap  │ 36 tiles     │ 36 × 81      │ 2,916 patches          │
├────────────┼─────────┼──────────────┼──────────────┼────────────────────────┤
│ SUB-TOTAL  │ TRAIN   │ 108 tiles    │ 108 × 81     │ 8,748 patches          │
├────────────┼─────────┼──────────────┼──────────────┼────────────────────────┤
│ Validation │ Tyrol-w │ 36 tiles     │ 36 × 81      │ 2,916 patches          │
├────────────┼─────────┼──────────────┼──────────────┼────────────────────────┤
│ Test       │ Vienna  │ 36 tiles     │ 36 × 81      │ 2,916 patches          │
├────────────┼─────────┼──────────────┼──────────────┼────────────────────────┤
│ TOTAL      │ ALL     │ 180 tiles    │ 180 × 81     │ 14,580 patches         │
└────────────┴─────────┴──────────────┴──────────────┴────────────────────────┘
```

---

## 5. Per-City and Per-Tile Verification

- **Per-City Patch Count:** Each of the 5 cities (Austin, Chicago, Kitsap, Tyrol-w, Vienna) contains exactly **36 tiles $\times$ 81 patches = 2,916 patches**.
- **Per-Tile Patch Count:** Every single tile from `austin1` through `vienna36` yields **exactly 81 patches** ($\min = 81, \max = 81, \text{std} = 0.0$).

---

## 6. File Integrity Reconciliation

A three-way reconciliation was performed comparing manifest records against disk storage:

- **Manifest Record Count (`part2_patch_manifest.csv`):** **14,580 rows**
- **Image Files on Disk (`data/patches/*/images/*.png`):** **14,580 PNG files**
- **Mask Files on Disk (`data/patches/*/gt/*.png`):** **14,580 PNG files**
- **File Matching Rate:** **$100.0\%$** ($0$ missing files, $0$ corrupt files, $0$ unreferenced files).

---

## 7. Corrected Class Statistics Across All 14,580 Patches

Class statistics were re-calculated across all $14,580$ patches ($3,822,059,520 \text{ total pixels}$):

| Statistical Metric | Re-Calculated Full Dataset Value | Methodological Significance |
| :--- | :--- | :--- |
| **Total Building Pixels** | $605,375,617 \text{ px}$ | Ground truth positive building pixels |
| **Total Background Pixels** | $3,216,683,903 \text{ px}$ | Non-building background land cover |
| **Overall Building Pixel %** | **$15.8390\%$** | Real building foreground density |
| **Overall Background Pixel %**| **$84.1610\%$** | Dominant background proportion |
| **Low Coverage Patches ($<5\%$)** | $5,379 \text{ patches}$ ($36.89\%$) | Sparse urban/rural land cover |
| **Moderate Coverage ($5-25\%$)** | $5,527 \text{ patches}$ ($37.91\%$) | Suburban residential structures |
| **High Coverage ($>25\%$)** | $3,674 \text{ patches}$ ($25.20\%$) | Dense urban commercial centers |

---

## 8. Final Audit Status & Checklist

```text
================================================================================
FINAL STEP 6 PATCH EXTRACTION AUDIT — STATUS: [PASS]
================================================================================
  [✓] 1. Inconsistency Identified & Documented (810 -> 14,580 patches)
  [✓] 2. Root Cause Traced (Removed hard-coded 10-tile TARGET_TILES_PER_CITY filter)
  [✓] 3. Extraction Geometry Verified (9x9 = 81 patches per 5000x5000 tile)
  [✓] 4. Corrected Counts Reconciled (Train: 8,748 | Val: 2,916 | Test: 2,916)
  [✓] 5. Per-City & Per-Tile Consistency (36 tiles/city, 81 patches/tile)
  [✓] 6. 3-Way File Reconciliation Passed (14,580 manifest = 14,580 imgs = 14,580 masks)
  [✓] 7. Automated Integrity Check Passed (512x512x3 RGB, 512x512 binary {0, 255})
  [✓] 8. Class Statistics Re-Calculated (Building: 15.84%, Background: 84.16%)
  [✓] 9. Documentation & Manifest Updated (PART2_PATCH_PIPELINE.md & manifest CSV)
================================================================================
```
