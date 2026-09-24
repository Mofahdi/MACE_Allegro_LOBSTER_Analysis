import argparse
import torch
import joblib

from allegro_eij_icoxx_analysis import AllegroEijIcoxxAnalyzer

parser = argparse.ArgumentParser(description="test LR of Allegro feats.")
parser.add_argument("--num-dcor-samples", type=int, default=40, help="Number of bootstrap samples for distance-correlation estimation.")
parser.add_argument("--dcor-sample-size", type=int, default=50000, help="Sample size used within each distance-correlation bootstrap iteration.")
parser.add_argument("--dpi", type=int, default=800, help="Figure resolution in dots per inch for saved plots.")
args = parser.parse_args()

dpi = args.dpi


# Use GPU when available; otherwise fall back to CPU for model inference.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device ='cpu'
# Initialize the analysis pipeline with the structure metadata CSV and trained model.
alleg_icoxx = AllegroEijIcoxxAnalyzer(
    df_path="nonmetals_mp_info_test3.csv",
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
PCA_12=alleg_icoxx.EijFeats_pca




print('\n\nICOHP Predictions')
alleg_icoxx.linProb(icohp = True, dpi = dpi)



print('\n\nICOBI Predictions')
alleg_icoxx.linProb(icohp = False, dpi = dpi)


# Plot Eij versus bond distance and report Pearson and distance-correlation values.
Pcorr_d, dcor_d = alleg_icoxx.plot_Eij_d(
    save_plot=True,
    Eii_icoxx_plot_name="Eij_d.jpg",
    dpi=dpi,
)
print("Eij and bond length Pearson and dcor Correlations:", Pcorr_d, dcor_d)

# Plot Eij versus ICOHP and report the corresponding correlation metrics.
PCorr_ICOHP, dcor_ICOHP = alleg_icoxx.plot_Eij_icoxx(
    icohp=True,
    save_plot=True,
    Eii_icoxx_plot_name="Eij_ICOHP.jpg",
    dpi=dpi,
)
print("Eij and ICOHP Pearson and dcor Correlations:", PCorr_ICOHP, dcor_ICOHP)

# Plot Eij versus ICOBI as a complementary comparison to the ICOHP analysis.
PCorr_ICOBI, dcor_ICOBI = alleg_icoxx.plot_Eij_icoxx(
    icohp=False,
    save_plot=True,
    Eii_icoxx_plot_name="Eij_ICOBI.jpg",
    dpi=dpi,
)
print("Eij and ICOBI Pearson and dcor Correlations:", PCorr_ICOBI, dcor_ICOBI)




