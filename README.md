# Raman Spectroscopy Analysis of Macrophage Polarization States

Machine learning pipeline for classifying macrophage polarization states (M0, M1, M2) from Raman spectroscopy data, covering raw data pre-processing through dimensionality reduction and classification.

---

## Dataset

- **Input:** Raman spectra of macrophages with 1300 wavenumber features per spectrum
- **Classes:** M0 (resting), M1 (pro-inflammatory), M2 (anti-inflammatory)
- **Source file:** `macrophage.csv` (labels in column 0, spectra in columns 1–1300)

---

## Workflow

### 1. Pre-processing (`Raman data pre-processing.ipynb`)

Raw Raman data is processed in four sequential steps:

**Step 1 — Column Separation**  
Reshapes a stacked multi-spectrum CSV into individual spectrum columns; removes duplicate columns. Output: `M1 column separation.xlsx`

**Step 2 — Cosmic Ray Removal**  
Detects spike artifacts using a modified Z-score on the first derivative of intensity. Spikes exceeding a threshold of 5 are replaced by the local mean of non-spike neighbors within a window of ±7 points. Output: `M1 cosmic ray removal.xlsx`

**Step 3 — Baseline Correction**  
Removes fluorescence background using the ZhangFit algorithm (`BaselineRemoval` library), which iteratively fits and subtracts a polynomial baseline. Output: `M1 baseline correction.xlsx`

**Step 4 — Normalization**  
Applies min-max scaling (via `sklearn.preprocessing.MinMaxScaler`) to bring all spectra to the [0, 1] range. Output: `M1 out.xlsx`

---

### 2. Dimensionality Reduction (`Macrophage_PCA vs tSNE vs UMAP.ipynb`)

Three unsupervised methods are compared for 2D visualization of the spectral data:

| Method | Notes |
|--------|-------|
| **PCA** | Linear; fast; preserves global variance |
| **t-SNE** | Non-linear; preserves local cluster structure |
| **UMAP** | Non-linear; `n_neighbors=5`, `min_dist=0.3`, `metric='correlation'` |

Scatter plots are generated and saved as PNG files for each method.

---

### 3. Classification (`Macrophage_PCA+classifier.ipynb`, `PCA-LDA_Macrophage.ipynb`)

All classifiers are evaluated with **10-fold repeated stratified cross-validation (10 repeats)**.

#### Baseline
| Model | Accuracy |
|-------|----------|
| Decision Tree (raw features) | 56.6% |

#### PCA + Classifier Pipelines

PCA is used for feature reduction before classification. The number of principal components (1–59) is swept to find the optimal setting.

| Pipeline | Best Accuracy |
|----------|--------------|
| PCA + Logistic Regression | **69.4% ± 5.1%** (10 PCs) |
| PCA + LDA | Comparable to LR; visualized via canonical variables |
| PCA + Decision Tree | **~71.9% ± 4.6%** (5 PCs) |

#### PCA-LDA Score Plot (`PCA-LDA_Macrophage.ipynb`)
PCA (2 or 8 components) followed by Linear Discriminant Analysis for class-separating 2D visualization. LDA canonical variables are plotted with M0/M1/M2 color-coded.

---

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
umap-learn
BaselineRemoval
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn umap-learn BaselineRemoval
```

---

## File Structure

```
├── Raman data pre-processing.ipynb       # Steps 1–4: column sep → cosmic ray → baseline → normalize
├── Macrophage_PCA vs tSNE vs UMAP.ipynb  # PCA / t-SNE / UMAP visualization comparison
├── Macrophage_PCA+classifier.ipynb       # PCA + LR / LDA / Decision Tree with component sweep
├── PCA-LDA_Macrophage.ipynb              # PCA-LDA score plot (2 PC and 8 PC variants)
├── Macrophage.ipynb                      # PCA scatter plot (basic)
└── README.md
```
