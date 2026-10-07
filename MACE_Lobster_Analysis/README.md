# MACE + LOBSTER Site Feature Analysis

This directory contains a feature-analysis pipeline that combines MACE node embeddings with site-level LOBSTER descriptors to study relationships between learned atomic representations and bonding/charge quantities.

The workflow includes:

- reading a material metadata CSV,
- constructing paths to structure and LOBSTER files,
- extracting site-level LOBSTER features (charge, ICOHP, ICOBI, etc.),
- computing MACE descriptors for each atomic site,
- reducing MACE descriptors with PCA,
- comparing MACE and LOBSTER feature spaces with bootstrap distance correlation,
- fitting linear models for charge and ICOXX predictions,
- generating scatter plots and correlation heatmaps.

## Repository contents

- `run_mace_lobster_analysis.py`
  Workflow entry point for the analysis.

- `mace_lobster_node_feature_analysis.py`
  Main analysis class with feature extraction, PCA, plotting, and regression routines.

- `lobster_site_feature_extractor.py`
  Helper used to extract LOBSTER-derived site features.

- `distance_correlation.py`
  Bootstrap-based distance-correlation implementation.

- `nonmetals_mp_info.csv`
  Material metadata table used by the pipeline.

- `mace-mpa-0-medium.model`
  MACE model used for descriptor generation.

## Expected input format

The workflow expects a CSV file with a column named `mat_path`. Each entry should point to a material directory containing the required structure and LOBSTER files.

Example:

```csv
mat_path
/path/to/material_1
/path/to/material_2
```

The script automatically constructs paths such as:

- `CONTCAR.gz`
- `ICOHPLIST.lobster.gz`
- `ICOBILIST.lobster.gz`
- `CHARGE.lobster.gz`

## Run the analysis

From this directory, execute:

```bash
python run_mace_lobster_analysis.py --num-dcor-samples 50 --dcor-sample-size 300 --dpi 800
```

The default arguments are defined in the script, so this also works:

```bash
python run_mace_lobster_analysis.py
```

## What the script does

The workflow currently performs the following sequence:

1. Initializes `MACE_LOBSTER_NodeFeatureAnalysis`.
2. Extracts MACE descriptors and LOBSTER site features.
3. Builds the combined dataframe with PCA coordinates.
4. Generates Pearson correlation heatmaps for all features and PCA terms.
5. Fits linear regression models for:
   - `chg_loew`
   - `chg_mull`
   - `sum_icohp`
   - `sum_icobi`
   - `mean_icohp`
   - `mean_icobi`
6. Computes bootstrap distance-correlation summaries from sampled data using `compute_distance_correlation(...)`.
7. Evaluates the relationship between MACE PCA axes and charge vectors using `plot_pc_space_and_charge_dcor(...)`.
8. Produces charge-vs-PC plots and sum-ICOXX-vs-PC plots to visualize the learned feature structure.

In the current run script, the dCor step is explicitly called after the linear-probing fits and before the PCA charge and ICOXX scatter plots are generated.

## Generated outputs

The script writes output files to the current working directory and to subdirectories created automatically:

- `MACE_Lobster_corr/`
  Correlation and dCor heatmap outputs.

- `MACE_chrg_corr/`
  Charge-related PCA plots and charge dCor summaries.

- `LinearProbe/`
  Linear-regression models and prediction CSVs.

Representative outputs include:

- `PCAMACE_lobster_Pearson.jpg`
- `MACE_lobster_mean_dcor.jpg`
- `MACE_lobster_std_dcor.jpg`
- `2DPCAMACE_LoewdinChg.jpg`
- `2DPCAMACE_MullChg.jpg`
- `chg_loew_PC1_MACE.jpg`
- `chg_mull_PC2_MACE.jpg`
- `sum_icohp_PC2_MACE.jpg`
- `sum_icobi_PC2_MACE.jpg`
- `chg_loew_linearProbe.jpg`

The script also saves JSON summaries such as:

- `MACE_lobster_dcor_mean.json`
- `MACE_lobster_dcor_std.json`
- `PCAMACE_charges_dcor_mean.json`
- `PCAMACE_charges_dcor_std.json`

These dCor summaries are generated from the bootstrap sample size and bootstrap count configured in the run script (`--dcor-sample-size` and `--num-dcor-samples`).

## Notes

- This pipeline assumes a working MACE installation and access to the corresponding model file.
- The metadata file must contain valid paths to each material directory and the expected LOBSTER/structure files.
- Output directories are created automatically if they do not exist.
- The full analysis is intended for comparing MACE node embeddings with site-level charge and bonding information derived from LOBSTER outputs.

## Installation

First install DGL with the correct PyTorch/CUDA wheel for this environment:

```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

Then install the remaining Python dependencies from the requirements file:

```bash
pip install -r requirements.txt
```
