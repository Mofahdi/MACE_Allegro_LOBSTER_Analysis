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
    df_path="nonmetals_mp_info.csv",
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


# Plot Eij versus ICOBI as a complementary comparison to the ICOHP analysis.
PCorr_ICOBI, dcor_ICOBI = alleg_icoxx.plot_Eij_icoxx(
    icohp=False,
    save_plot=True,
    Eii_icoxx_plot_name="Eij_ICOBI.jpg",
    dpi=dpi,
)
print("Eij and ICOBI Pearson and dcor Correlations:", PCorr_ICOBI, dcor_ICOBI)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('\n\nICOHP Predictions')
# PCA_12: shape [number_of_edges, 2]
# icohps: torch tensor with shape [number_of_edges, 1]
#X = np.asarray(PCA_12, dtype=np.float32)
X = np.asarray(xijFeats, dtype=np.float32)
y = icohps.detach().cpu().numpy().reshape(-1)

# Remove invalid or non-finite samples
valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[valid]
y = y[valid]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "icohp_linear_regression_model.pkl")
y_pred = model.predict(X_test)

#print("Regression coefficients:", model.coef_)
#print("Intercept:", model.intercept_)
print("MAE:", mean_absolute_error(y_test, y_pred))
#print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R2:", r2_score(y_test, y_pred))

# Predicted versus actual ICOHP
plt.figure(figsize=(5, 4))
plt.scatter(y_test, y_pred, c='b', alpha=0.2)
limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(limits, limits, "k--", label="ideal prediction")
plt.xlabel("Actual ICOHP (eV)")
plt.ylabel("Predicted ICOHP (eV)")
plt.legend()
plt.tight_layout()
plt.savefig("ICOHP_linear_regression.png", dpi=dpi)
#plt.show()
plt.close()





print('\n\nICOBI Predictions')
y = icobis.detach().cpu().numpy().reshape(-1)

# Remove invalid or non-finite samples
valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[valid]
y = y[valid]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "icobi_linear_regression_model.pkl")

y_pred = model.predict(X_test)

#print("Regression coefficients:", model.coef_)
#print("Intercept:", model.intercept_)
print("MAE:", mean_absolute_error(y_test, y_pred))
#print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R2:", r2_score(y_test, y_pred))

# Predicted versus actual ICOHP
plt.figure(figsize=(5, 4))
plt.scatter(y_test, y_pred, c='b', alpha=0.2)
limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(limits, limits, "k--", label="ideal prediction")
plt.xlabel("Actual ICOBI")
plt.ylabel("Predicted ICOBI")
plt.legend()
plt.tight_layout()
plt.savefig("ICOBI_linear_regression.png", dpi=dpi)
#plt.show()

# Plot Eij versus ICOBI as a complementary comparison to the ICOHP analysis.
PCorr_ICOBI, dcor_ICOBI = alleg_icoxx.plot_Eij_icoxx(
    icohp=False,
    save_plot=True,
    Eii_icoxx_plot_name="Eij_ICOBI.jpg",
    dpi=dpi,
)
print("Eij and ICOBI Pearson and dcor Correlations:", PCorr_ICOBI, dcor_ICOBI)




