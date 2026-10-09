import os
import argparse

import torch

from allegro_eij_icoxx_analysis import AllegroEijIcoxxAnalyzer

parser = argparse.ArgumentParser(description="Run the Allegro Eij/ICOXX analysis workflow.")
parser.add_argument("--num-dcor-samples", type=int, default=20, help="Number of bootstrap samples for distance-correlation estimation.")
parser.add_argument("--dcor-sample-size", type=int, default=50000, help="Sample size used within each distance-correlation bootstrap iteration.")
parser.add_argument("--dpi", type=int, default=800, help="Figure resolution in dots per inch for saved plots.")
args = parser.parse_args()

# good numbers are: 20 then 50,000, then 800
num_dcor_samples = args.num_dcor_samples
dcor_sample_size = args.dcor_sample_size
dpi = args.dpi

# Use GPU when available; otherwise fall back to CPU for model inference.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device ='cpu'
# Initialize the analysis pipeline with the structure metadata CSV and trained model.
alleg_icoxx = AllegroEijIcoxxAnalyzer(
    df_path="mp_data_paths.csv",
    model_path="Allegro-OAM-L-0.1.nequip.zip",
    device=device,
)


# Extract paired Eij values, edge features, ICOHP/ICOBI terms, and bond distances
# across all structures after intersecting the Allegro and ICOXX graphs.
# We also output them here in case someone needs to process them differently
ids, Eijs, xijFeats, icohps, icobis, dists = alleg_icoxx.get_Eijs_EijFeats_icoxxs()

# Reduce the Xij feature matrix to a 2D PCA space for visualization and downstream analysis.
PCAXijFeats_var = alleg_icoxx.PCA_Xij()
print("PCA EijFeats variation:", PCAXijFeats_var)
print("number of edges", Eijs.shape[0])


# Plot Eij versus bond distance and report Pearson and distance-correlation values.
Pcorr_d, dcor_d = alleg_icoxx.plot_Eij_d(
    save_plot=True,
    Eij_icoxx_plot_name="Eij_vs_distance.png",
    dpi=dpi,
)
print("Eij and bond length Pearson and dcor Correlations:", Pcorr_d, float(dcor_d))

# Plot Eij versus ICOHP and report the corresponding correlation metrics.
PCorr_ICOHP, dcor_ICOHP = alleg_icoxx.plot_Eij_icoxx(
    icohp=True,
    save_plot=True,
    Eij_icoxx_plot_name="Eij_vs_ICOHP.png",
    dpi=dpi,
)
print("Eij and ICOHP Pearson and dcor Correlations:", PCorr_ICOHP, float(dcor_ICOHP))

# Plot Eij versus ICOBI as a complementary comparison to the ICOHP analysis.
PCorr_ICOBI, dcor_ICOBI = alleg_icoxx.plot_Eij_icoxx(
    icohp=False,
    save_plot=True,
    Eij_icoxx_plot_name="Eij_vs_ICOBI.png",
    dpi=dpi,
)
print("Eij and ICOBI Pearson and dcor Correlations:", PCorr_ICOBI, float(dcor_ICOBI))


print('\n\nICOHP Linear Probe:')
alleg_icoxx.fit_linProb(icohp = True, dpi = dpi)

print('\nICOBI Linear Probe:')
alleg_icoxx.fit_linProb(icohp = False, dpi = dpi)

# Visualize the PCA-reduced xij feature space colored by bond distance and
# compute the distance-correlation matrix between the PCA components and distance.
XijD_dcor_mean_mat, XijD_dcor_std_mat = alleg_icoxx.plot_PCAXijFeats_d_and_dcor(
    num_dcor_samples=num_dcor_samples,
    dcor_sample_size=dcor_sample_size,
    save_plot=True,
    PCAXijFeats_dist_plot_name="PCA_feature_space_distance.png",
    dpi=dpi,
)
print("\n\nd, Xij dcor (mean, std):")
print(XijD_dcor_mean_mat)
print(XijD_dcor_std_mat)
XijD_dcor_mean_mat.to_json(os.path.join(alleg_icoxx.EijAnalysis, "PCA_distance_dcor_mean.json"), orient="records", indent=4)
XijD_dcor_std_mat.to_json(os.path.join(alleg_icoxx.EijAnalysis, "PCA_distance_dcor_std.json"), orient="records", indent=4)

# Repeat the PCA-coloring workflow using ICOHP as the color value to compare it with distance.
XijICOHP_dcor_mean, XijICOHP_dcor_std = alleg_icoxx.plot_PCAXij_icoxx_and_dcor(
    num_dcor_samples=num_dcor_samples,
    dcor_sample_size=dcor_sample_size,
    icohp=True,
    save_plot=True,
    PCAXijFeats_icoxx_plot_name="PCA_feature_space_ICOHP.png",
    dpi=dpi,
)
print("\n\nICOHP, Xij dcor (mean, std):")
print(XijICOHP_dcor_mean)
print(XijICOHP_dcor_std)
XijICOHP_dcor_mean.to_json(os.path.join(alleg_icoxx.EijAnalysis, "PCA_ICOHP_dcor_mean.json"), orient="records", indent=4)
XijICOHP_dcor_std.to_json(os.path.join(alleg_icoxx.EijAnalysis, "PCA_ICOHP_dcor_std.json"), orient="records", indent=4)

# Repeat the PCA-coloring workflow using ICOBI as the color value for comparison.
XijICOBI_dcor_mean, XijICOBI_dcor_std = alleg_icoxx.plot_PCAXij_icoxx_and_dcor(
    num_dcor_samples=num_dcor_samples,
    dcor_sample_size=dcor_sample_size,
    icohp=False,
    save_plot=True,
    PCAXijFeats_icoxx_plot_name="PCA_feature_space_ICOBI.png",
    dpi=dpi,
)
print("\n\nICOBI, Xij dcor (mean, std):")
print(XijICOBI_dcor_mean)
print(XijICOBI_dcor_std)
XijICOBI_dcor_mean.to_json(os.path.join(alleg_icoxx.EijAnalysis, "PCA_ICOBI_dcor_mean.json"), orient="records", indent=4)
XijICOBI_dcor_std.to_json(os.path.join(alleg_icoxx.EijAnalysis, "PCA_ICOBI_dcor_std.json"), orient="records", indent=4)



