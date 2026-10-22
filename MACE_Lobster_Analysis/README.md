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

- `mp_data_paths.csv`
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
python run_mace_lobster_analysis.py \
  --num-dcor-samples 50 \
  --dcor-sample-size 300 \
  --extension .pdf \
  --dpi 800
```

The default arguments are defined in the script, so this also works:

```bash
python run_mace_lobster_analysis.py
```

Available command-line arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--num-dcor-samples` | `50` | Number of bootstrap samples used for distance-correlation estimates. |
| `--dcor-sample-size` | `300` | Number of observations sampled in each bootstrap iteration. |
| `--extension` | `.png` | File extension for saved figures, including the leading dot (for example, `.png`, `.pdf`, or `.jpg`). |
| `--dpi` | `800` | Resolution in dots per inch for saved figures. |

The selected extension applies to figure outputs throughout the workflow. It does not change the extensions of JSON summaries, CSV files, or serialized model files.

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

- `mace_lobster_pearson_heatmap.png`
- `mace_lobster_pearson_correlation.json`
- `mace_lobster_dcor_mean_heatmap.png`
- `mace_lobster_dcor_mean.json`
- `mace_lobster_dcor_std.json`
- `mace_pca_loewdin_charge_scatter.png`
- `mace_pca_mulliken_charge_scatter.png`
- `mace_pc1_vs_loewdin_charge.png`
- `mace_pc2_vs_mulliken_charge.png`
- `mace_pc2_vs_sum_icohp.png`
- `mace_pc2_vs_sum_icobi.png`
- `mace_linear_probe_chg_loew.png`
- `mace_linear_probe_chg_loew_model.pkl`
- `mace_linear_probe_sum_icohp_results.csv`

The script also saves JSON summaries such as:

- `mace_lobster_dcor_mean.json`
- `mace_lobster_dcor_std.json`
- `mace_pca_charge_dcor_mean.json`
- `mace_pca_charge_dcor_std.json`

These dCor summaries are generated from the bootstrap sample size and bootstrap count configured in the run script (`--dcor-sample-size` and `--num-dcor-samples`).

## Notes

- This pipeline assumes a working MACE installation and access to the corresponding model file.
- The metadata file must contain valid paths to each material directory and the expected LOBSTER/structure files.
- Output directories are created automatically if they do not exist.
- The full analysis is intended for comparing MACE node embeddings with site-level charge and bonding information derived from LOBSTER outputs.

## Installation

First create an environment and activate it

```bash
conda create -n mace_lobster_env python=3.12.13
```
```bash
conda activate mace_lobster_env
```

Then, install DGL with the correct PyTorch/CUDA wheel for this environment:

```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

Then install the remaining Python dependencies from the requirements file:

```bash
pip install -r requirements.txt
```
