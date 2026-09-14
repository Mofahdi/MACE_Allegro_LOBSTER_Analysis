# MACE LOBSTER Analyzer

This directory contains a small analysis pipeline that combines MACE-based node descriptors with LOBSTER-derived site features and quantifies their relationship using distance correlation (dCor).

The workflow is designed to:

- read a material metadata table,
- load structure and LOBSTER output files for each material,
- extract site-level features from ICOHP/ICOBI and charge data,
- compute MACE descriptors for each atomic site,
- standardize and compare the feature spaces,
- project MACE embeddings to 2D PCA and color them by charge values,
- compute bootstrap-based distance correlation between PCA/MACE features and charge vectors.

## Contents

- `run_mace_lobster_analysis.py`
  Entry point for the full workflow.

- `mace_lobster_node_feature_analysis.py`
  Main analysis class and feature extraction logic.

- `lobster_site_feature_extractor.py`
  Extracts site-level LOBSTER descriptors and charge features.

- `distance_correlation.py`
  Bootstrap-based distance-correlation estimator.

- `nonmetals_mp_info.csv`
  Material metadata table used by the pipeline.

- `mace-mpa-0-medium.model`
  Pretrained MACE model used to generate descriptors.


## Dependencies

This analysis used these packages:

- Python 3.12.13
- PyTorch 2.4.0
- DGL 2.4.0+cu124
- ASE 3.29.0
- scikit-learn 1.9.0
- pandas 2.3.3
- NumPy 2.5.2
- matplotlib 3.11.1
- pymatgen 2026.5.4
- lobsterpy 0.6.1
- MACE 0.3.16


## Inputs
The analysis expects a CSV file containing a column named `mat_path`, where each row points to a structure directory. The script automatically constructs paths such as:

- `CONTCAR.gz`
- `ICOHPLIST.lobster.gz`
- `ICOBILIST.lobster.gz`

Example metadata file:

```csv
mat_path
/path/to/material_1
/path/to/material_2
.. and so on
```


## Running the analysis

From this directory, run:

```bash
python run_mace_lobster_analysis.py
```

This script will:

1. initialize the analyzer with the material metadata file,
2. extract MACE and LOBSTER features for all sites,
3. compute the MACE–LOBSTER dCor matrix,
4. run PCA on the MACE node embeddings,
5. save charge-colored PCA plots,
6. compute dCor between PCA features and the charge values.

## Output files

The script writes the following artifacts:

- `MACE_lobster_dcor_mean.json`
  Mean distance-correlation matrix between MACE and LOBSTER features.

- `MACE_lobster_dcor_std.json`
  Standard deviation of the bootstrap distance-correlation matrix.

- `MACE_charges_dcor_mean.json`
  Mean distance-correlation values between PCA/MACE features and charge vectors.

- `MACE_charges_dcor_std.json`
  Standard deviation of the charge-related dCor estimates.

- `PCAMACE_2D_LoewdinChg.jpg`
  PCA embedding colored by Loewdin charge.

- `PCAMACE_2D_MullChg.jpg`
  PCA embedding colored by Mulliken charge.

## Notes

- The metadata table must contain a `mat_path` column pointing to directories with the required LOBSTER and structure files.
- The script expects the corresponding structure files to be present in each material directory, including files such as `CONTCAR.gz`, `ICOHPLIST.lobster.gz`, `ICOBILIST.lobster.gz`, and `CHARGE.lobster.gz`.
- The pipeline is most useful for comparing how MACE node embeddings align with site charge information and LOBSTER-derived bonding descriptors.
