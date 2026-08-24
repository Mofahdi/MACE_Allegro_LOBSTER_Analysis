import os
import pandas as pd
import matplotlib.pyplot as plt
import dcor

import torch, dgl
#from get_MACE_feats_merge import create_DGL_g, intersect_graphs
#from graph_converter import Icoxx2Graph
from mace.calculators.mace import MACECalculator
from ase.io import read 
from matminer_lobster_feats import get_lobster_struc_feat, get_matminer_struc_feats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
default_dtype = torch.float32
torch.set_default_dtype(default_dtype)

mace_calc=MACECalculator(model_paths='../mace-mpa-0-medium.model', model_dtype='float32', device='cuda')
mace_model=mace_calc.models[0]

df=pd.read_csv('5_lo_chg_mpInfo_nonmetals_test4.csv')
df['struc_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR.gz'))
df['ICOHP_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster.gz'))
df['ICOBI_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster.gz'))
df['CHRG_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CHARGE.lobster.gz'))

mace_chunks=[]; matminer_chunks=[]; lobster_chunks=[]; matminer_lobster_chunks=[]; #
for i, (cont_path, icohp_path, icobi_path, chrg_dir) in enumerate(zip(df['struc_dir'].values, df['ICOHP_dir'].values,
                                                                       df['ICOBI_dir'].values, df['CHRG_dir'].values)):
    ase_atoms=read(cont_path)
    batch=mace_calc._atoms_to_batch(ase_atoms).to(device)
    batch_dict=mace_calc._clone_batch(batch).to_dict()
    node_descs=torch.from_numpy(mace_calc.get_descriptors(atoms=ase_atoms)).float().to(device)
    #mace_out=mace_model(batch_dict, compute_edge_forces=True)
    #g_mace=create_DGL_g(ase_atoms=ase_atoms, batch_dict=batch_dict, mace_out=mace_out, node_descs=node_descs)
    mace_chunks.append(torch.mean(node_descs, dim=0).unsqueeze(0)) # [1, # feats]

    if i==0:
        matminer_feats, matminer_feat_labels = get_matminer_struc_feats(cont_path, output_feat_labels=True)
        lobster_feats, lobster_feat_labels = get_lobster_struc_feat(structure_path=cont_path, ICOHP_path=icohp_path, ICOBI_path=icobi_path, chrg_path=chrg_dir, output_feat_labels=True)
    else: 
        matminer_feats = get_matminer_struc_feats(cont_path, output_feat_labels=False)
        lobster_feats = get_lobster_struc_feat(structure_path=cont_path, ICOHP_path=icohp_path, ICOBI_path=icobi_path, chrg_path=chrg_dir, output_feat_labels=False)

    matminer_chunks.append(matminer_feats)
    lobster_chunks.append(lobster_feats)
    matminer_lobster_chunks.append(torch.cat([matminer_feats, lobster_feats], dim=-1))
    #exit()

mace_feats = torch.cat(mace_chunks, dim=0).cpu().numpy()
matminer_feats = torch.cat(matminer_chunks, dim=0).cpu().numpy()
lobster_feats = torch.cat(lobster_chunks, dim=0).cpu().numpy()
matminer_lobster_feats = torch.cat(matminer_lobster_chunks, dim=0).cpu().numpy()

from sklearn.preprocessing import StandardScaler
from distance_correlation import evaluate_distance_correlation_matrix_bootstrap
# Standardize each edge feature
scaler = StandardScaler()
mace_feats_scaled = scaler.fit_transform(mace_feats)
matminer_feats_scaled = scaler.fit_transform(matminer_feats)
lobster_feats_scaled = scaler.fit_transform(lobster_feats)
matminer_lobster_feats_scaled = scaler.fit_transform(lobster_feats)
num_resamples=mace_feats.shape[0]

sets = {
    'MACE': mace_feats_scaled,
    'MATMINER': matminer_feats_scaled,
    'LOBSTER': lobster_feats_scaled, 
    'matminerLobster': matminer_lobster_feats_scaled,
    }

mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(sets=sets, sample_size=500, num_bootstrap=20)
#mean_mat, std_mat, mean_pvals = evaluate_distance_correlation_matrix_bootstrap(sets=sets, num_resamples=1000, num_bootstrap=20)
print(mean_mat)
print(std_mat)


dcor_value_matminer = dcor.distance_correlation(
    mace_feats_scaled,
    matminer_feats_scaled,)
dcor_value_lobster = dcor.distance_correlation(
    mace_feats_scaled,
    lobster_feats_scaled,)
dcor_value_matLobster = dcor.distance_correlation(
    mace_feats_scaled,
    matminer_lobster_feats_scaled,)
print('matminer, lobster, matminer+lobster dcors:', dcor_value_matminer, dcor_value_lobster, dcor_value_matLobster)


feat_dcor_scores={}
for i, feat_type in enumerate(lobster_feat_labels):
    dcor_score=dcor.distance_correlation(mace_feats_scaled,
        lobster_feats_scaled[:, i])
    feat_dcor_scores.update({feat_type: float(dcor_score)})
sorted_feat_dcor_scores = dict(
    sorted(feat_dcor_scores.items(), key=lambda item: item[1], reverse=True)
)
print(sorted_feat_dcor_scores)


