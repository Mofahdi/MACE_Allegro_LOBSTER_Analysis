import os
import pandas as pd
import matplotlib.pyplot as plt
import dcor

import torch, dgl
from get_MACE_feats_merge import create_DGL_g, intersect_graphs
from graph_converter import Icoxx2Graph
from mace.calculators.mace import MACECalculator
from ase.io import read 
from matminer_lobster_node_feats import get_lobster_site_feat, get_matminer_site_feats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
default_dtype = torch.float32
torch.set_default_dtype(default_dtype)

mace_calc=MACECalculator(model_paths='mace-mpa-0-medium.model', model_dtype='float32', device='cuda')
mace_model=mace_calc.models[0]

df=pd.read_csv('5_lo_chg_mpInfo_nonmetals_test3.csv')
df['struc_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR.gz'))
df['ICOHP_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster.gz'))
df['ICOBI_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster.gz'))
df['CHRG_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CHARGE.lobster.gz'))

mace_chunks=[]; mace_node_chunks=[]; matminer_chunks=[]; lobster_chunks=[]; matminer_lobster_chunks=[]; #energies=[]; inter_es=[]
for i, (cont_path, icoxx_path, chrg_dir) in enumerate(zip(df['struc_dir'].values, df['ICOHP_dir'].values, df['CHRG_dir'].values)):
    ase_atoms=read(cont_path)
    batch=mace_calc._atoms_to_batch(ase_atoms).to(device)
    batch_dict=mace_calc._clone_batch(batch).to_dict()
    node_descs=torch.from_numpy(mace_calc.get_descriptors(atoms=ase_atoms)).float().to(device)
    #mace_out=mace_model(batch_dict, compute_edge_forces=True)
    #g_mace=create_DGL_g(ase_atoms=ase_atoms, batch_dict=batch_dict, mace_out=mace_out, node_descs=node_descs)
    mace_chunks.append(torch.mean(node_descs, dim=0).unsqueeze(0))
    mace_node_chunks.append(node_descs)

    matminer_n_feats = get_matminer_site_feats(cont_path)
    lobster_n_feats = get_lobster_site_feat(structure_path=cont_path, ICOXX_path=icoxx_path, chrg_path=chrg_dir)
    matminer_chunks.append(matminer_n_feats)
    lobster_chunks.append(lobster_n_feats)
    matminer_lobster_chunks.append(torch.cat([matminer_n_feats, lobster_n_feats], dim=-1))
    #energies.append(mace_out['energy'].item())
    #inter_es.append(mace_out['interaction_energy'].item())

mace_feats = torch.cat(mace_chunks, dim=0).cpu().numpy()
mace_node_feats = torch.cat(mace_node_chunks, dim=0).cpu().numpy()
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

mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(sets=sets, num_bootstrap=20)
#mean_mat, std_mat, mean_pvals = evaluate_distance_correlation_matrix_bootstrap(sets=sets, num_resamples=1000, num_bootstrap=20)
print(mean_mat)
print(std_mat)


dcor_value_matminer = dcor.distance_correlation(
    mace_node_feats_scaled,
    matminer_feats_scaled,)
dcor_value_lobster = dcor.distance_correlation(
    mace_node_feats_scaled,
    lobster_feats_scaled,)
dcor_value_matLobster = dcor.distance_correlation(
    mace_node_feats_scaled,
    matminer_lobster_feats_scaled,)
print(dcor_value_matminer, dcor_value_lobster, dcor_value_matLobster)


