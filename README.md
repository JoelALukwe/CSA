# Cytokine Signature Analysis Pipeline

A reproducible statistical and machine-learning workflow for multiplex cytokine data (Luminex-style).

## Quick summary
- Preprocessing: missingness, imputation, log transform, scaling
- Stats: univariate testing, multiple-testing correction
- Multivariate: PCA, clustering
- ML: Random Forest baseline with cross-validation + feature importance
- Reproducibility: conda environment, runnable scripts and notebook

## Repo structure
env/environment.yml
data/synthetic_cytokine.csv
scripts/generate_synthetic_data.py,preprocess.py,analysis.py
notebooks/cytokine_analysis_notebook.py
results/pcaplot.png,scaled_matrix.png,feature_summary.csv

## Usage (quick)
1. Create conda env: `conda env create -f env/environment.yml`
2. Activate: `conda activate cytokine-env`
3. Generate synthetic data (optional): `python scripts/generate_synthetic_data.py --output data/synthetic_cytokine.csv`
4. Preprocess: `python scripts/preprocess.py --input data/synthetic_cytokine.csv --output data/synthetic_cytokine_preprocessed.csv`
5. Run analysis: `python scripts/analysis.py --input data/synthetic_cytokine_preprocessed.csv --outdir results/`
6. Open `notebooks/Cytokine_analysis_notebook.py` (or convert to .ipynb) to see narrative and saved figures.

## Notes
- All analysis is reproducible with the environment file.
- Please note Data used is synthetic and is provided to illustrate pipeline behavior.
- Replace `data/synthetic_cytokine.csv` with your real dataset (same shape: samples x cytokines plus `group` column).
