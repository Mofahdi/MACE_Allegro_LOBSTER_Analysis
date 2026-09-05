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
import dcor
from distance_correlation import evaluate_distance_correlation_matrix_bootstrap

process = psutil.Process(os.getpid())
def print_memory(i):
    ram_gb = process.memory_info().rss / (1024 ** 3)
    print(f"After {i} materials: RAM = {ram_gb:.2f} GB", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
default_dtype = torch.float32
torch.set_default_dtype(default_dtype)

mace_calc=MACECalculator(model_paths='../../mace-mpa-0-medium.model', model_dtype='float32', device='cuda')
mace_calc.models[0].eval()

df=pd.read_csv('5_lo_chg_mpInfo_nonmetals_test.csv')
df['struc_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR.gz'))
df['ICOHP_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster.gz'))
df['ICOBI_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster.gz'))
df['CHRG_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CHARGE.lobster.gz'))

mace_node_chunks=[]; matminer_chunks=[]; lobster_chunks=[]; #matminer_lobster_chunks=[]; 
for i, (cont_path, icohp_path, icobi_path, chrg_dir) in enumerate(zip(df['struc_dir'].values, df['ICOHP_dir'].values, 
                                                                      df['ICOBI_dir'].values, df['CHRG_dir'].values)):
    try:
        #matminer_n_feats = get_matminer_site_feats(cont_path) # [# nodes, feats]
        if i==0:
            lobster_n_feats, lobster_feat_labels = get_lobster_site_feat(structure_path=cont_path, 
                                                            ICOHP_path=icohp_path, ICOBI_path=icobi_path, 
                                                            chrg_path=chrg_dir, output_feat_labels=True)
        else:
            lobster_n_feats = get_lobster_site_feat(structure_path=cont_path, 
                                                    ICOHP_path=icohp_path, ICOBI_path=icobi_path, 
                                                    chrg_path=chrg_dir, output_feat_labels=False) # [# nodes, feats]    
        #matminer_chunks.append(matminer_n_feats)
        lobster_chunks.append(lobster_n_feats)

        #print(len(Structure.from_file(cont_path)), cont_path)
        ase_atoms=read(cont_path)
        #with torch.no_grad():
        mace_descs=mace_calc.get_descriptors(atoms=ase_atoms).astype(np.float32) # [# nodes, feats]
        mace_node_chunks.append(mace_descs)

        del ase_atoms; del lobster_n_feats; del mace_descs;
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        #if (i + 1) % 100 == 0:
         #   print_memory(i + 1)

    except Exception as e: 
        print(f"FAILED: {cont_path}")
        print(f"ERROR: {e}")

print('finished extracting features')

mace_node_feats = np.concatenate(mace_node_chunks, axis=0)
del mace_node_chunks; 
#matminer_feats = np.concatenate(matminer_chunks, axis=0) 
#del matminer_chunks;

df_lobster = pd.DataFrame(lobster_chunks, columns=lobster_feat_labels)
df_lobster = df_lobster.dropna(axis=1)
lobster_feat_labels = df_lobster.columns.tolist()
lobster_feats = df_lobster.to_numpy()
#lobster_feats = np.concatenate(lobster_chunks, axis=0) 
del lobster_chunks
#matminer_lobster_feats =  np.concatenate([matminer_feats, lobster_feats], axis=-1)


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
    #'MATMINER': matminer_feats_scaled,
    'LOBSTER': lobster_feats_scaled, 
    #'matminerLobster': matminer_lobster_feats_scaled,
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


print('find PCA of MACE features color-coded with Loewdin charges....')
pca = PCA(n_components=2, svd_solver="randomized")
mace_node_feats_pca = pca.fit_transform(mace_node_feats_scaled)
dcor_PCAMACE_lowedin=dcor.distance_correlation(mace_node_feats_scaled, lobster_feats[:, -1])

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
plt.savefig('PCAMACE_chg.jpg', dpi=800) # dpi=dpi)
#plt.show()
