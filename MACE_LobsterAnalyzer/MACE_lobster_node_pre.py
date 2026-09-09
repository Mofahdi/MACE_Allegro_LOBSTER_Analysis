import os
import json
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch, dgl
from mace.calculators.mace import MACECalculator
from ase.io import read 
from lobster_node_feats import get_lobster_site_feat
from distance_correlation import evaluate_distance_correlation_matrix_bootstrap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
default_dtype = torch.float32
torch.set_default_dtype(default_dtype)

mace_calc=MACECalculator(model_paths='../../../mace-mpa-0-medium.model', model_dtype='float32', device='cuda')
mace_calc.models[0].eval()

df=pd.read_csv('5_lo_chg_mpInfo_nonmetals_test4.csv')
df['struc_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR.gz'))
df['ICOHP_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster.gz'))
df['ICOBI_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster.gz'))
df['CHRG_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CHARGE.lobster.gz'))

mace_node_chunks=[]; matminer_chunks=[]; lobster_chunks=[]; 
for i, (cont_path, icohp_path, icobi_path, chrg_dir) in enumerate(zip(df['struc_dir'].values, df['ICOHP_dir'].values, 
                                                                      df['ICOBI_dir'].values, df['CHRG_dir'].values)):
    try:
        if i==0:
            lobster_n_feats, lobster_feat_labels = get_lobster_site_feat(structure_path=cont_path, 
                                                            ICOHP_path=icohp_path, ICOBI_path=icobi_path, 
                                                            chrg_path=chrg_dir, output_feat_labels=True)
        else:
            lobster_n_feats = get_lobster_site_feat(structure_path=cont_path, 
                                                    ICOHP_path=icohp_path, ICOBI_path=icobi_path, 
                                                    chrg_path=chrg_dir, output_feat_labels=False) # [# nodes, feats]    
        lobster_chunks.append(lobster_n_feats)

        ase_atoms=read(cont_path)
        mace_descs=mace_calc.get_descriptors(atoms=ase_atoms).astype(np.float32) # [# nodes, feats]
        mace_node_chunks.append(mace_descs)

        del ase_atoms; del lobster_n_feats; del mace_descs;
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e: 
        print(f"FAILED: {cont_path}")
        print(f"ERROR: {e}")

print('done extracting features')

mace_node_feats = np.concatenate(mace_node_chunks, axis=0)
del mace_node_chunks; 
df_lobster = pd.DataFrame(np.concatenate(lobster_chunks, axis=0) , columns=lobster_feat_labels)
df_lobster = df_lobster.dropna(axis=1)
lobster_feat_labels = df_lobster.columns.tolist()
lobster_feats = df_lobster.to_numpy()
del lobster_chunks

print('dimensions of nodes with MACE and lobster features:')
print('[# nodes, # MACE feats]:', mace_node_feats.shape)
print('[# nodes, # lobster feats]:', lobster_feats.shape)




print('started scaling features')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Standardize each node feature
mace_scaler = StandardScaler(copy=False)
mace_node_feats_scaled = mace_scaler.fit_transform(mace_node_feats)
lobster_scaler = StandardScaler(copy=False)
lobster_feats_scaled = lobster_scaler.fit_transform(lobster_feats)
sets = {
    'MACE': mace_node_feats_scaled,
    'LOBSTER': lobster_feats_scaled, 
    }

print('compute sampled dCor....')
mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(sets=sets, sample_size=400, num_bootstrap=100)
#mean_mat, std_mat, mean_pvals = evaluate_distance_correlation_matrix_bootstrap(sets=sets, num_resamples=1000, num_bootstrap=20)
print(mean_mat)
print(std_mat)
print('\n\nICOBI, Xij dcor (mean, std):'); print(mean_mat); print(std_mat);  #EijICOBI_dcor)
mean_mat.to_json("MACE_lobster_dcor_mean.json", orient="records", indent=4)
std_mat.to_json("MACE_lobster_dcor_std.json", orient="records", indent=4)


# dcor_value_lobster = dcor.distance_correlation(
#     mace_node_feats_scaled,
#     lobster_feats_scaled,)
# print('lobster', dcor_value_lobster)
# feat_dcor_scores={}
# for i, feat_type in enumerate(lobster_feat_labels):
#     #dcor_score=dcor.distance_correlation(mace_node_feats_scaled,
#      #   lobster_feats_scaled[:, i])
#     dcor_score, std_mat = evaluate_distance_correlation_matrix_bootstrap(sets=sets, sample_size=400, num_bootstrap=100)
#     feat_dcor_scores.update({feat_type: float(dcor_score[0,1])})
# sorted_feat_dcor_scores = dict(
#     sorted(feat_dcor_scores.items(), key=lambda item: item[1], reverse=True)
# )
# print(sorted_feat_dcor_scores)


#dcor_PCAMACE_lowedin=dcor.distance_correlation(mace_node_feats_scaled, lobster_feats[:, -1])
sets = {
    'MACE': mace_node_feats_scaled,
    'lowedin_chg': lobster_feats[:, -1], 
    }
mean_dcor_MACE_lowedin, std_dcor_MACE_lowedin = evaluate_distance_correlation_matrix_bootstrap(sets=sets, sample_size=500, num_bootstrap=100)
mean_dcor_MACE_lowedin.to_json("MACE_lobster_dcor_mean.json", orient="records", indent=4)
std_dcor_MACE_lowedin.to_json("MACE_lobster_dcor_std.json", orient="records", indent=4)


print('find PCA of MACE features color-coded with Loewdin charges....')
pca = PCA(n_components=2, svd_solver="randomized")
mace_node_feats_pca = pca.fit_transform(mace_node_feats_scaled)
plt.figure(figsize=(4.5,3.5))
plt.scatter(
    mace_node_feats_pca[:, 0],
    mace_node_feats_pca[:, 1],
    c=lobster_feats[:, -1],
    s=5,
    #cmap="rainbow"
    )
plt.xlabel(r"PCA1 (MACE Node Features)", fontsize=12)
plt.ylabel(r"PCA2 (MACE Node Features)", fontsize=12)
cbar = plt.colorbar(label='Charge Loewdin')
#nbar=len(cbar.get_xticks())
cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=len(cbar.ax.get_yticks())+1))
ax=plt.gca()
nx = len(ax.get_xticks())
#ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
ny = len(ax.get_yticks())
#ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
plt.tight_layout()
#if save_plot:
plt.savefig('PCAMACE_LoewdinChg.jpg', dpi=800) # dpi=dpi)
#plt.show()





plt.figure(figsize=(4.5,3.5))
plt.scatter(
    mace_node_feats_pca[:, 0],
    mace_node_feats_pca[:, 1],
    c=lobster_feats[:, -2],
    s=5,
    #cmap="rainbow"
    )
plt.xlabel(r"PCA1 (MACE Node Features)", fontsize=12)
plt.ylabel(r"PCA2 (MACE Node Features)", fontsize=12)
cbar = plt.colorbar(label='Mulliken Charge')
#nbar=len(cbar.get_xticks())
cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=len(cbar.ax.get_yticks())+1))
ax=plt.gca()
nx = len(ax.get_xticks())
#ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
ny = len(ax.get_yticks())
#ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
plt.tight_layout()
#if save_plot:
plt.savefig('PCAMACE_MullChg.jpg', dpi=800) # dpi=dpi)

