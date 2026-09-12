# Allegro Eij / ICOXX Analysis

This project combines Allegro graph predictions with Lobster ICOXX bond data to analyze the relationship between:

- Allegro edge energies $E_{ij}$
- bond distance
- ICOHP values
- ICOBI values
- reduced PCA features derived from edge features

The workflow reads structure metadata, loads the trained Allegro model, intersects the model graph with the ICOXX graph, and then produces plots and distance-correlation summaries.

## Project files

- `allegro_eij_icoxx_analysis.py` — main analysis class and plotting logic
- `allegro_graph_pipeline.py` — Allegro model loading and DGL graph construction
- `icoxx_graph.py` — ICOXX graph conversion utilities
- `distance_correlation.py` — distance-correlation helper functions
- `run_allegro_eij_icoxx_analysis.py` — command-line runner for the full analysis

## Requirements

This project expects the following dependencies:

- Python 3.9+
- PyTorch 2.4.0
- DGL 2.4.0+cu124
- ASE 3.29.0
- scikit-learn 1.9.0
- pandas 3.0.5
- NumPy 2.5.2
- matplotlib 3.11.1
- pymatgen 2026.5.4
- nequip-allegro 0.8.3
- dcor 0.7

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

You also need a trained Allegro model package, for example:

```text
Allegro-OAM-L-0.1.nequip.zip
```

## Running the analysis

From this directory, run:

```bash
python run_allegro_eij_icoxx_analysis.py \
  --num-dcor-samples 40 \
  --dcor-sample-size 50000 \
  --dpi 800
```

Optional arguments:

- `--num-dcor-samples`: number of bootstrap samples for the distance-correlation analysis
- `--dcor-sample-size`: sample size used in each bootstrap draw
- `--dpi`: image resolution for saved plots

## Outputs

The script produces:

- scatter plots for:
  - `Eij` vs. bond distance
  - `Eij` vs. ICOHP
  - `Eij` vs. ICOBI
  - PCA feature space colored by distance
  - PCA feature space colored by ICOHP/ICOBI
- summary JSON files such as:
  - `XijD_dcor_mean_mat.json`
  - `XijD_dcor_std_mat.json`
  - `XijICOHP_dcor_mean.json`
  - `XijICOHP_dcor_std.json`
  - `XijICOBI_dcor_mean.json`
  - `XijICOBI_dcor_std.json`

## Typical workflow

1. Prepare a metadata CSV with `mat_path` entries.
2. Place the trained Allegro model in the same folder or update the model path.
3. Run the CLI script.
4. Inspect saved plots and JSON summaries.

## Notes

- The analysis pipeline expects the structure graph and the ICOXX graph to share comparable edge keys after intersection.
- If your dataset is large, increasing `dcor_sample_size` or `num_dcor_samples` may increase runtime and memory usage.
- The code is designed for a reproducible analysis workflow, not only a quick one-off plot.
