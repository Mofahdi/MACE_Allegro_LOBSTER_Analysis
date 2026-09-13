# Allegro_Eij_ICOXX_Analysis and MACE_Lobster_Analysis

This project folder brings together two complementary materials-analysis workflows:

- `Allegro_Eij_ICOXX_Analysis` studies Allegro-predicted edge energies together with LOBSTER ICOXX bond descriptors.
- `MACE_Lobster_Analysis` studies MACE-derived site descriptors together with LOBSTER charge and bonding features.

The shared goal is to compare different machine-learning representations of the same material systems and quantify how they relate to bonding, charge, and structura descriptors using distance correlation and visualization. In both workflows, we test whether edge-based from Allegro and site-based descriptors from MACE implicitly capture chemical rules.


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

The `Allegro_Eij_ICOXX_Analysis` workflow is built around the Allegro model and LOBSTER ICOXX outputs. It evaluates how edge-level properties such as

- Allegro edge energies $E_{ij}$,
- bond distances,
- ICOHP values,
- ICOBI values,
- PCA-reduced edge-feature embeddings

relate to one another.

This workflow is implemented mainly in:

- `Allegro_Eij_ICOXX_Analysis/allegro_eij_icoxx_analysis.py`
- `Allegro_Eij_ICOXX_Analysis/allegro_graph_pipeline.py`
- `Allegro_Eij_ICOXX_Analysis/icoxx_graph.py`
- `Allegro_Eij_ICOXX_Analysis/run_allegro_eij_icoxx_analysis.py`

Typical outputs include distance-correlation matrices and plots such as:

- `Eij_d.jpg`
- `Eij_ICOHP.jpg`
- `Eij_ICOBI.jpg`
- `XijD_dcor_mean_mat.json`
- `XijICOHP_dcor_mean.json`
- `XijICOBI_dcor_mean.json`

## 2) MACE_Lobster_Analysis

The `MACE_Lobster_Analysis` workflow focuses on MACE-derived site descriptors and LOBSTER-derived site features. It compares descriptor spaces and tests their alignment with charge information using distance correlation.

This workflow is implemented mainly in:

- `MACE_Lobster_Analysis/run_mace_lobster_analysis.py`
- `MACE_Lobster_Analysis/mace_lobster_node_feature_analysis.py`
- `MACE_Lobster_Analysis/lobster_site_feature_extractor.py`
- `MACE_Lobster_Analysis/distance_correlation.py`

Typical outputs include:

- `MACE_lobster_dcor_mean.json`
- `MACE_lobster_dcor_std.json`
- `MACE_charges_dcor_mean.json`
- `MACE_charges_dcor_std.json`
- `PCAMACE_2D_LoewdinChg.jpg`
- `PCAMACE_2D_MullChg.jpg`

## Shared scientific workflow

Both analysis folders follow a similar overall pipeline:

1. read a metadata table containing material paths,
2. load structure and bonding data,
3. generate descriptors from ML models,
4. align those descriptors with LOBSTER-derived chemical features,
5. compare the feature spaces with distance correlation,
6. visualize the results and save summary statistics.


## Notes

- Both workflows depend on consistent material metadata and valid path conventions.
- Distance-correlation calculations can become expensive for large datasets, especially when the number of bootstrap samples or sample sizes is high.
