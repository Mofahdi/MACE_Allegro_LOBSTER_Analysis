# Allegro Eij / ICOXX Analysis

This workflow loads crystal-structure metadata, runs a trained Allegro model, intersects the Allegro edge graph with ICOXX bond data from Lobster, and then analyzes how the Allegro edge energy $E_{ij}$ relates to bond distance, ICOHP, ICOBI, and learned edge-feature embeddings.

The current pipeline includes:

- edge-level extraction of paired Allegro and ICOXX quantities
- PCA on the Allegro edge-feature matrix
- scatter plots of $E_{ij}$ versus bond distance and ICOXX values
- linear-probe regression for predicting ICOHP or ICOBI from the Allegro edge features
- distance-correlation summaries for the PCA space and the target scalar quantities

## Project files

- `allegro_eij_icoxx_analysis.py` — main analysis class, plotting functions, and regression workflow
- `allegro_graph_pipeline.py` — model loading, graph construction, and graph transforms
- `icoxx_graph.py` — ICOXX graph conversion and bond parsing
- `distance_correlation.py` — bootstrap-based distance-correlation utilities
- `run_allegro_eij_icoxx_analysis.py` — command-line entry point for the full analysis
- `README.md` — project usage and output description

## Dependencies

This project is built around the following stack:

- Python 3.12.13
- PyTorch 2.4.0
- DGL 2.4.0+cu124
- ASE 3.29.0
- scikit-learn 1.9.0
- pandas 3.0.6
- NumPy 2.5.3
- matplotlib 3.11.1
- pymatgen 2026.5.4
- nequip-allegro 0.8.3
- dcor 0.7


## Installation

First create an environment and activate it

```bash
conda create -n allegro_icoxx python=3.12.13
```
```bash
conda activate allegro_icoxx
```

Then, install DGL with the correct PyTorch/CUDA wheel for this environment:

```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

Then install the remaining Python dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

## Input data

The analysis expects a CSV file with at least a `mat_path` column. Each row should point to a structure directory whose contents include:

- `CONTCAR.gz`
- `ICOHPLIST.lobster.gz`
- `ICOBILIST.lobster.gz`

The code also reads an `id` column when present and uses it in the output tables for tracking individual structures or edges.

Example metadata file:

```csv
id,mat_path
1,/path/to/material_1
2,/path/to/material_2
3,/path/to/material_3
.......
```

The script also expects a trained Allegro model package, for example:

```text
Allegro-OAM-L-0.1.nequip.zip
```

The default script uses:

- `df_path="mp_data_paths.csv"`
- `model_path="Allegro-OAM-L-0.1.nequip.zip"`

## How the analysis works

The main execution flow in `run_allegro_eij_icoxx_analysis.py` is:

1. Load the metadata CSV and model.
2. Build the Allegro graph and ICOXX graph for each structure.
3. Intersect the two graphs on matching bond edges.
4. Extract:
   - `ids`
   - `Eijs`
   - `EijFeats`
   - `icohps`
   - `icobis`
   - `dists`
5. Standardize and reduce the edge-feature matrix with PCA.
6. Save scatter plots and summary statistics for:
   - $E_{ij}$ vs. bond distance
   - $E_{ij}$ vs. ICOHP
   - $E_{ij}$ vs. ICOBI
   - PCA feature space colored by distance and ICOXX values
7. Train and save linear-probe models to predict ICOHP or ICOBI from the Allegro edge embeddings.
8. Compute bootstrap distance-correlation summaries for the PCA and scalar features.

## Running the analysis

From the project directory, run:

```bash
python run_allegro_eij_icoxx_analysis.py \
  --num-dcor-samples 20 \
  --dcor-sample-size 50000 \
  --dpi 800
```

The default arguments are defined in the script, so this also works:

```bash
python run_allegro_eij_icoxx_analysis.py 
```


Command-line arguments:

- `--num-dcor-samples`: number of bootstrap samples used in the distance-correlation analysis
- `--dcor-sample-size`: sample size used in each bootstrap iteration
- `--dpi`: saved figure resolution in dots per inch

The script automatically uses GPU if available and falls back to CPU otherwise.

## Generated outputs

The pipeline creates two output folders in the current working directory if they do not already exist:

- `EijAnalysis/`
- `LinearProbe/`

### `EijAnalysis/` outputs

This folder contains scatter plots and tabular summaries such as:

- `Eij_vs_distance.png`
- `Eij_vs_ICOHP.png`
- `Eij_vs_ICOBI.png`
- `PCA_feature_space_distance.png`
- `PCA_feature_space_ICOHP.png`
- `PCA_feature_space_ICOBI.png`
- `Eij_vs_distance_data.csv`
- `Eij_vs_ICOHP_data.csv`
- `Eij_vs_ICOBI_data.csv`
- `PCA_feature_space_distance_data.csv`
- `PCA_feature_space_ICOHP_data.csv`
- `PCA_feature_space_ICOBI_data.csv`
- `PCA_distance_dcor_mean.json`
- `PCA_distance_dcor_std.json`
- `PCA_ICOHP_dcor_mean.json`
- `PCA_ICOHP_dcor_std.json`
- `PCA_ICOBI_dcor_mean.json`
- `PCA_ICOBI_dcor_std.json`

### `LinearProbe/` outputs

This folder stores the trained linear-regression models and evaluation tables:

- `linear_probe_ICOHP_model.pkl`
- `linear_probe_ICOBI_model.pkl`
- `linear_probe_ICOHP_results.csv`
- `linear_probe_ICOBI_results.csv`
- `linear_probe_ICOHP.png`
- `linear_probe_ICOBI.png`

## Notes and caveats

- The graph intersection step expects the Allegro graph and ICOXX graph to be aligned on comparable edge-level structure after filtering.
- Large datasets can be expensive because the script performs per-structure graph construction, regression, and bootstrap distance-correlation estimation.
- Increasing `--dcor-sample-size` or `--num-dcor-samples` will increase runtime and memory use.
- The code is intended for reproducible analysis and result generation rather than only quick visual inspection.

## Typical workflow

1. Prepare a metadata CSV with `mat_path` entries (and an optional `id` column).
2. Ensure the trained Allegro model file is available in the working directory or edit the model path in the script.
3. Run the analysis command.
4. Review the generated plots and summary tables in `EijAnalysis/` and `LinearProbe/`.
