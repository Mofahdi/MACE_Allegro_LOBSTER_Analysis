# Allegro and MACE + LOBSTER Analysis

This repository brings together two complementary pipelines for comparing machine learning interatomic potential (MLIP) structure descriptors with LOBSTER-derived bonding and charge information:

- `Allegro_Eij_ICOXX_Analysis/` studies Allegro-predicted edge energies and edge-level bonding descriptors.
- `MACE_Lobster_Analysis/` studies MACE-derived node-level descriptors and their relationship to LOBSTER charge and bond summaries.

Together, they provide a consistent framework for testing whether different learned representations capture chemically meaningful structure-bonding relationships. In both workflows, the analysis combines descriptor extraction, PCA, linear probing, and distance-correlation evaluation to quantify alignment between model embeddings and physically interpretable chemical quantities.

## Overview

The two workflows share the same overall objective:

1. read material metadata and locate the relevant structural files,
2. compute model-derived descriptors from either Allegro or MACE,
3. extract chemically relevant quantities from LOBSTER output files,
4. compare the descriptor spaces using Pearson and distance-correlation analyses,
5. visualize the results and save summary metrics for downstream interpretation.

## Project structure

```text
Allegro_Eij_ICOXX/
├── Allegro_Eij_ICOXX_Analysis/
│   └── Allegro-based edge-energy and ICOXX workflow
├── MACE_Lobster_Analysis/
│   └── MACE + LOBSTER feature-analysis workflow
├── README.md
└── other analysis outputs / experiments
```

## 1) Allegro_Eij_ICOXX_Analysis

The Allegro workflow evaluates how edge-level quantities relate to each other, including:

- Allegro edge energies $E_{ij}$
- bond distances
- ICOHP values
- ICOBI values
- PCA-reduced edge-feature embeddings $x_{ij}$

This pipeline loads a CSV of material entries, builds the Allegro graph for each structure, intersects it with the corresponding LOBSTER ICOXX graph, and then compares the resulting edge-level features using correlation analysis and visualization.

### Main files

- `Allegro_Eij_ICOXX_Analysis/allegro_eij_icoxx_analysis.py`
- `Allegro_Eij_ICOXX_Analysis/allegro_graph_pipeline.py`
- `Allegro_Eij_ICOXX_Analysis/icoxx_graph.py`
- `Allegro_Eij_ICOXX_Analysis/run_allegro_eij_icoxx_analysis.py`

### Typical inputs

The workflow expects a metadata CSV with at least a `mat_path` and `id` columns. Each structure directory should include:

- `CONTCAR.gz`
- `ICOHPLIST.lobster.gz`
- `ICOBILIST.lobster.gz`

A trained Allegro model package is also required, for example:

```text
Allegro-OAM-L-0.1.nequip.zip
```

### Current outputs

The Allegro workflow creates:

- `EijAnalysis/` with plots and summary files such as:
  - `Eij_d.jpg`
  - `Eij_ICOHP.jpg`
  - `Eij_ICOBI.jpg`
  - `PCAXijFeatureSpace_dist.jpg`
  - `PCAXijFeatureSpace_ICOHP.jpg`
  - `PCAXijFeatureSpace_ICOBI.jpg`
  - `Dist_Eij_all.csv`
  - `ICOHP_Eij_all.csv`
  - `ICOBI_Eij_all.csv`
  - `XijD_dcor_mean_mat.json`
  - `XijICOHP_dcor_mean.json`
  - `XijICOBI_dcor_mean.json`

- `LinearProbe/` with fitted models and predictions such as:
  - `icohp_linear_regression_model.pkl`
  - `icobi_linear_regression_model.pkl`
  - `icohp_LinProb_results.csv`
  - `icobi_LinProb_results.csv`
  - `ICOHP_linearProbe.jpg`
  - `ICOBI_linearProbe.jpg`

### Example execution

```bash
cd Allegro_Eij_ICOXX_Analysis
python run_allegro_eij_icoxx_analysis.py \
  --num-dcor-samples 40 \
  --dcor-sample-size 50000 \
  --dpi 800
```

---

## 2) MACE_Lobster_Analysis

The MACE workflow evaluates how learned site descriptors relate to LOBSTER-derived charge and bond quantities. It compares MACE node embeddings with:

- Loewdin charges
- Mulliken charges
- summed ICOHP / ICOBI
- mean ICOHP / ICOBI
- PCA-compressed MACE node representations

This analysis reads structure metadata, extracts LOBSTER site features, computes MACE descriptors, and then studies their statistical association through correlation matrices and distance-correlation estimation.

### Main files

- `MACE_Lobster_Analysis/run_mace_lobster_analysis.py`
- `MACE_Lobster_Analysis/mace_lobster_node_feature_analysis.py`
- `MACE_Lobster_Analysis/lobster_site_feature_extractor.py`
- `MACE_Lobster_Analysis/distance_correlation.py`

### Typical inputs

The workflow expects structure metadata and structure directories containing:

- `CONTCAR.gz`
- `ICOHPLIST.lobster.gz`
- `ICOBILIST.lobster.gz`
- `CHARGE.lobster.gz`

The current default MACE model is:

```text
mace-mpa-0-medium.model
```

### Current outputs

The MACE workflow writes to:

- `MACE_Lobster_corr/` for correlation summaries and PCA visuals
- `MACE_chrg_corr/` for charge-focused summaries
- `LinearProbe/` for linear-probe regression outputs across targeted charge and ICOXX quantities

Representative outputs include:

- `PCAMACE_lobster_Pearson.jpg`
- `Pearson_r_AllFeats.json`
- `MACE_lobster_dcor_mean.json`
- `MACE_lobster_dcor_std.json`
- `MACE_charges_dcor_mean.json`
- `MACE_charges_dcor_std.json`
- linear-probe plots and CSV outputs for:
  - Loewdin charge
  - Mulliken charge
  - sum ICOHP
  - sum ICOBI
  - mean ICOHP
  - mean ICOBI

### Example execution

```bash
cd MACE_Lobster_Analysis
python run_mace_lobster_analysis.py \
  --num-dcor-samples 50 \
  --dcor-sample-size 300 \
  --dpi 800
```

---

## Shared scientific principles

Both analysis workflows are built around the same core ideas:

- learned descriptors can be compared directly against physically meaningful chemical quantities,
- PCA helps visualize high-dimensional descriptor structure,
- distance correlation offers a nonlinear measure of association beyond linear correlation,
- linear probing provides a simple quantitative test of how much information is encoded in the learned representations.

## Notes

- These workflows are intended for reproducible analysis of large structural datasets rather than one-off plotting only.
- Distance-correlation calculations can become expensive as the sample size and bootstrap count increase.
- Some file paths and model names may need to be adjusted depending on your current dataset or environment.
