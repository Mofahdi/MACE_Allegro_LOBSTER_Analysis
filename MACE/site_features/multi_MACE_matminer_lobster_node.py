import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch, dgl
from mace.calculators.mace import MACECalculator
from ase.io import read 
from matminer_lobster_node_feats import get_lobster_site_feat, get_matminer_site_feats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import dcor
from distance_correlation import evaluate_distance_correlation_matrix_bootstrap


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
default_dtype = torch.float32
torch.set_default_dtype(default_dtype)

mace_calc=MACECalculator(model_paths='../mace-mpa-0-medium.model', model_dtype='float32', device='cuda')
#mace_model=mace_calc.models[0]

df=pd.read_csv('5_lo_chg_mpInfo_nonmetals_test.csv')
df['struc_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR.gz'))
df['ICOHP_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster.gz'))
df['ICOBI_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster.gz'))
df['CHRG_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CHARGE.lobster.gz'))


mace_node1_chunks=[]; mace_node2_chunks=[]; matminer_chunks=[]; lobster_chunks=[]; matminer_lobster_chunks=[]; 
for i, (cont_path, icohp_path, icobi_path, chrg_dir) in enumerate(zip(df['struc_dir'].values, df['ICOHP_dir'].values, 
                                                                      df['ICOBI_dir'].values, df['CHRG_dir'].values)):
    try:
        matminer_n_feats = get_matminer_site_feats(cont_path) # [# nodes, feats]
        if i==0:
            lobster_n_feats, lobster_feat_labels = get_lobster_site_feat(structure_path=cont_path, 
                                                            ICOHP_path=icohp_path, ICOBI_path=icobi_path, 
                                                            chrg_path=chrg_dir, output_feat_labels=True)
        else:
            lobster_n_feats = get_lobster_site_feat(structure_path=cont_path, 
                                                    ICOHP_path=icohp_path, ICOBI_path=icobi_path, 
                                                    chrg_path=chrg_dir, output_feat_labels=False) # [# nodes, feats]
    except: continue
    matminer_chunks.append(matminer_n_feats)
    lobster_chunks.append(lobster_n_feats)
    matminer_lobster_chunks.append(torch.cat([matminer_n_feats, lobster_n_feats], dim=-1))

    ase_atoms=read(cont_path)
    node_descs1=torch.from_numpy(mace_calc.get_descriptors(atoms=ase_atoms, num_layers=1)).float().to(device) # [# nodes, feats]
    node_descs2=torch.from_numpy(mace_calc.get_descriptors(atoms=ase_atoms, num_layers=2)).float().to(device) # [# nodes, feats]
    mace_node1_chunks.append(node_descs1)
    mace_node2_chunks.append(node_descs2)


mace_node1_feats = torch.cat(mace_node1_chunks, dim=0).cpu().numpy()
mace_node2_feats = torch.cat(mace_node2_chunks, dim=0).cpu().numpy()
matminer_feats = torch.cat(matminer_chunks, dim=0).cpu().numpy()
lobster_feats = torch.cat(lobster_chunks, dim=0).cpu().numpy()
matminer_lobster_feats = torch.cat(matminer_lobster_chunks, dim=0).cpu().numpy()



# Standardize each node feature
scaler = StandardScaler()
mace_node1_feats_scaled = scaler.fit_transform(mace_node1_feats)
mace_node2_feats_scaled = scaler.fit_transform(mace_node2_feats)
matminer_feats_scaled = scaler.fit_transform(matminer_feats)
lobster_feats_scaled = scaler.fit_transform(lobster_feats)
matminer_lobster_feats_scaled = scaler.fit_transform(lobster_feats)

sets = {
    'MACE_1': mace_node1_feats_scaled,
    'MACE_2': mace_node2_feats_scaled,
    'MATMINER': matminer_feats_scaled,
    'LOBSTER': lobster_feats_scaled, 
    'matminerLobster': matminer_lobster_feats_scaled,
    }

mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(sets=sets, sample_size=100, num_bootstrap=30)
print(mean_mat)
print(std_mat)

dcor_value_matminer_MACE1 = dcor.distance_correlation(
    mace_node1_feats_scaled,
    matminer_feats_scaled,)
dcor_value_lobster_MACE1 = dcor.distance_correlation(
    mace_node1_feats_scaled,
    lobster_feats_scaled,)
dcor_value_matLobster_MACE1 = dcor.distance_correlation(
    mace_node1_feats_scaled,
    matminer_lobster_feats_scaled,)
print('matminer, lobster, matminer+lobster w/ MACE1 dcors:', dcor_value_matminer_MACE1, dcor_value_lobster_MACE1, dcor_value_matLobster_MACE1)


dcor_value_matminer_MACE2 = dcor.distance_correlation(
    mace_node2_feats_scaled,
    matminer_feats_scaled,)
dcor_value_lobster_MACE2 = dcor.distance_correlation(
    mace_node2_feats_scaled,
    lobster_feats_scaled,)
dcor_value_matLobster_MACE2 = dcor.distance_correlation(
    mace_node2_feats_scaled,
    matminer_lobster_feats_scaled,)
print('matminer, lobster, matminer+lobster w/ MACE2 dcors:', dcor_value_matminer_MACE2, dcor_value_lobster_MACE2, dcor_value_matLobster_MACE2)

exit()

feat_dcor_scores={}
for i, feat_type in enumerate(lobster_feat_labels):
    dcor_score=dcor.distance_correlation(mace_last0_node_feats_scaled,
        lobster_feats_scaled[:, i])
    feat_dcor_scores.update({feat_type: float(dcor_score)})
sorted_feat_dcor_scores = dict(
    sorted(feat_dcor_scores.items(), key=lambda item: item[1], reverse=True)
)
print(sorted_feat_dcor_scores)




pca = PCA(n_components=2)
mace_node_feats_pca = pca.fit_transform(mace_last0_node_feats_scaled)
dcor_PCAMACE_lowedin=dcor.distance_correlation(mace_last0_node_feats_scaled, lobster_feats[:, -2])

plt.figure(figsize=(4.5,3.5))
plt.scatter(
    mace_node_feats_pca[:, 0],
    mace_node_feats_pca[:, 1],
    c=lobster_feats[:, -1],
    s=5,
    #cmap="rainbow"
    )
plt.xlabel(r"PCA1 ($x_{ij}$)", fontsize=12)
plt.ylabel(r"PCA2 ($x_{ij}$)", fontsize=12)
cbar = plt.colorbar(label='Charge Polarity')
#nbar=len(cbar.get_xticks())
cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=len(cbar.ax.get_yticks())+1))
ax=plt.gca()
nx = len(ax.get_xticks())
#ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
ny = len(ax.get_yticks())
#ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
plt.tight_layout()
#if save_plot:
plt.savefig('PCAMACE_chg.jpg',) # dpi=dpi)
plt.show()
