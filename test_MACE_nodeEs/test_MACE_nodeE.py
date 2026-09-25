import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from ase.io import read
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import torch
from icoxx_graph import Icoxx2Graph
from mace.calculators import mace_mp
from mace.calculators.mace import MACECalculator
import dcor

#from pymatgen.core import Structure
from pymatgen.io.lobster import Charge


default_dtype = torch.float32
torch.set_default_dtype(default_dtype)
device = "cuda" if torch.cuda.is_available() else "cpu"
#mace_calc = mace_mp(model="medium-mpa-0", device=device,) # default_dtype='float32')
mace_calculator = MACECalculator(model_paths='mace-mpa-0-medium.model', model_dtype='float32', device=device)
df=pd.read_csv('nonmetals_mp_info.csv')
df['struc_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR.gz'))
df['ICOHP_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster.gz'))
df['ICOBI_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster.gz'))
df['CHRG_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CHARGE.lobster.gz'))

mace_node_chunks=[]; mace_node_feats_chunks=[]; sum_ICOHP_chunks=[]; sum_ICOBI_chunks=[]; 
sum_masked_ICOHP_chunks=[]; mean_maskedICOHPs_chunks=[]; chg_mull_chunks=[]; chg_loew_chunks=[]
for i, (cont_path, icohp_path, icobi_path, chrg_path) in enumerate(zip(df['struc_dir'].values, 
                                                                      df['ICOHP_dir'].values, 
                                                                      df['ICOBI_dir'].values, 
                                                                      df['CHRG_dir'].values)):
    ase_atoms=read(cont_path)
    ase_atoms.calc = mace_calculator
    #ase_atoms.get_potential_energy()
    # mace_node_chunks.append(np.expand_dims(ase_atoms.calc.results["node_energy"], axis=0))
    mace_node_chunks.append(ase_atoms.get_potential_energies())

    mace_descs = mace_calculator.get_descriptors(atoms=ase_atoms).astype(np.float32) # [# nodes, feats]
    mace_node_feats_chunks.append(mace_descs)
    #print(type(ase_atoms.calc.results["node_energy"]), ase_atoms.calc.results["node_energy"].shape)

    graph_icoxx = Icoxx2Graph(bond_length_cutoff=5.0).get_graph(
        path_to_structure=os.path.join(cont_path),
        path_to_icohplist=os.path.join(icohp_path),
        path_to_icobilist=os.path.join(icobi_path),
    )
    sum_ICOHP_chunks.append(graph_icoxx.ndata['sum_ICOHPs'].numpy())
    sum_ICOBI_chunks.append(graph_icoxx.ndata['sum_ICOBIs'].numpy())

    charge_obj = Charge(filename=chrg_path)
    chg_mull = charge_obj.mulliken
    chg_loew = charge_obj.loewdin
    chg_mull_chunks.append(chg_mull); chg_loew_chunks.append(chg_loew)
    


mace_node_feats = np.concatenate(mace_node_feats_chunks, axis=0)
mace_node_es = np.concatenate(mace_node_chunks, axis=0)
sumICOHPs_node_feats = np.concatenate(sum_ICOHP_chunks, axis=0)
sumICOBIs_node_feats = np.concatenate(sum_ICOBI_chunks, axis=0)
chg_loew_feats = np.concatenate(chg_loew_chunks, axis=0)
chg_mull_feats = np.concatenate(chg_mull_chunks, axis=0)

df = pd.DataFrame({
    "MACE_Atomic_Es": mace_node_es,
    "sumICOHPs": sumICOHPs_node_feats,
    "sumICOBI": sumICOBIs_node_feats,
    'chg_loew': chg_loew_feats, 
    'chg_mull': chg_mull_feats, 
})
corr = df.corr(method="pearson")
print(corr)


# Fit a node-level linear model from MACE descriptors to Sum ICOHP.
regression_df = pd.DataFrame(mace_node_feats)
regression_df['target_sum_ICOHP'] = np.asarray(
    sumICOHPs_node_feats
).reshape(-1)
regression_df = regression_df.replace([np.inf, -np.inf], np.nan)
nodes_before_filter = len(regression_df)
regression_df = regression_df.dropna(axis=0, how='any').reset_index(drop=True)
print('\n\nLinear Regression sum ICOHPs...')
print('Removed nodes with non-finite features or target:', nodes_before_filter - len(regression_df), 'out of',nodes_before_filter)
clean_mace_node_feats = regression_df.drop(
    columns='target_sum_ICOHP'
).to_numpy(dtype=np.float32)
clean_sumICOHPs = regression_df['target_sum_ICOHP'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    clean_mace_node_feats,
    clean_sumICOHPs,
    test_size=0.2,
    random_state=42,
)
#linear_model = make_pipeline(StandardScaler(), LinearRegression())
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
r2=r2_score(y_test, y_pred)
print('Linear regression R2:', r2)
#print('Linear regression RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Linear regression MAE:', mean_absolute_error(y_test, y_pred))
joblib.dump(linear_model, "linear_regression_model_sumICOHP.pkl")

plt.figure(figsize=(5, 4))
plt.scatter(y_test, y_pred, c='b', alpha=0.2)
plot_limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(plot_limits, plot_limits, 'k--', linewidth=1)
plt.xlabel('Actual Sum ICOHP (eV)')
plt.ylabel('Predicted Sum ICOHP (eV)')
ax = plt.gca()
ax.text(
    0.05, 0.95,
    f"R$^2$ = {r2:.3f}",
    transform=ax.transAxes,
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
)
plt.tight_layout()
plt.savefig('linear_regression_mace_feats_vs_sumICOHPs.jpg', dpi=700)
plt.close()





# Fit a node-level linear model from MACE descriptors to Sum ICOBI.
regression_df = pd.DataFrame(mace_node_feats)
regression_df['target_sum_ICOBI'] = np.asarray(
    sumICOBIs_node_feats
).reshape(-1)
regression_df = regression_df.replace([np.inf, -np.inf], np.nan)
nodes_before_filter = len(regression_df)
regression_df = regression_df.dropna(axis=0, how='any').reset_index(drop=True)
print('\n\nLinear Regression sum ICOBIs...')
print('Removed nodes with non-finite features or target:', nodes_before_filter - len(regression_df), 'out of',nodes_before_filter)
clean_mace_node_feats = regression_df.drop(
    columns='target_sum_ICOBI'
).to_numpy(dtype=np.float32)
clean_sumICOBIs = regression_df['target_sum_ICOBI'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    clean_mace_node_feats,
    clean_sumICOBIs,
    test_size=0.2,
    random_state=42,
)
#linear_model = make_pipeline(StandardScaler(), LinearRegression())
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
print('Linear regression R2:', r2_score(y_test, y_pred))
#print('Linear regression RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Linear regression MAE:', mean_absolute_error(y_test, y_pred))
joblib.dump(linear_model, "linear_regression_model_sumICOBI.pkl")

plt.figure(figsize=(5, 4))
plt.scatter(y_test, y_pred, c='b', alpha=0.2)
plot_limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(plot_limits, plot_limits, 'k--', linewidth=1)
plt.xlabel('Actual Sum ICOBI')
plt.ylabel('Predicted Sum ICOBI')
ax = plt.gca()
ax.text(
    0.05, 0.95,
    f"R$^2$ = {r2:.3f}",
    transform=ax.transAxes,
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
)
plt.tight_layout()
plt.savefig('linear_regression_mace_feats_vs_sumICOBIs.jpg', dpi=700)
plt.close()





# Fit a node-level linear model from MACE descriptors to Chg Loewdin.
regression_df = pd.DataFrame(mace_node_feats)
regression_df['target_chg_loew'] = np.asarray(
    chg_loew_feats
).reshape(-1)
regression_df = regression_df.replace([np.inf, -np.inf], np.nan)
nodes_before_filter = len(regression_df)
regression_df = regression_df.dropna(axis=0, how='any').reset_index(drop=True)
print('\n\nLinear Regression Loewdin charge...')
print('Removed nodes with non-finite features or target:', nodes_before_filter - len(regression_df), 'out of',nodes_before_filter)
clean_mace_node_feats = regression_df.drop(
    columns='target_chg_loew'
).to_numpy(dtype=np.float32)
clean_mean_maskedICOHPs = regression_df['target_chg_loew'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    clean_mace_node_feats,
    clean_mean_maskedICOHPs,
    test_size=0.2,
    random_state=42,
)
#linear_model = make_pipeline(StandardScaler(), LinearRegression())
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
print('Linear regression R2:', r2_score(y_test, y_pred))
#print('Linear regression RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Linear regression MAE:', mean_absolute_error(y_test, y_pred))
joblib.dump(linear_model, "linear_regression_model_chgLoew.pkl")

plt.figure(figsize=(5, 4))
plt.scatter(y_test, y_pred, c='b', alpha=0.2)
plot_limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(plot_limits, plot_limits, 'k--', linewidth=1)
plt.xlabel('Actual Loewdin Charge (e)')
plt.ylabel('Predicted Loewdin Charge (e)')
plt.tight_layout()
plt.savefig('linear_regression_mace_feats_vs_chgLoew.jpg', dpi=700)
plt.close()






# Fit a node-level linear model from MACE descriptors to Chg Mulliken.
regression_df = pd.DataFrame(mace_node_feats)
regression_df['target_chg_mull'] = np.asarray(
    chg_mull_feats
).reshape(-1)
regression_df = regression_df.replace([np.inf, -np.inf], np.nan)
nodes_before_filter = len(regression_df)
regression_df = regression_df.dropna(axis=0, how='any').reset_index(drop=True)
print('\n\nLinear Regression Mulliken charge...')
print('Removed nodes with non-finite features or target:', nodes_before_filter - len(regression_df), 'out of',nodes_before_filter)
clean_mace_node_feats = regression_df.drop(
    columns='target_chg_mull'
).to_numpy(dtype=np.float32)
clean_mean_maskedICOHPs = regression_df['target_chg_mull'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    clean_mace_node_feats,
    clean_mean_maskedICOHPs,
    test_size=0.2,
    random_state=42,
)
#linear_model = make_pipeline(StandardScaler(), LinearRegression())
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
print('Linear regression R2:', r2_score(y_test, y_pred))
#print('Linear regression RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Linear regression MAE:', mean_absolute_error(y_test, y_pred))
joblib.dump(linear_model, "linear_regression_model_chgMull.pkl")

plt.figure(figsize=(5, 4))
plt.scatter(y_test, y_pred, c='b', alpha=0.2)
plot_limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(plot_limits, plot_limits, 'k--', linewidth=1)
plt.xlabel('Actual Mulliken Charge (e)')
plt.ylabel('Predicted Mulliken Charge (e)')
plt.tight_layout()
plt.savefig('linear_regression_mace_feats_vs_chgMull.jpg', dpi=700)
plt.close()

