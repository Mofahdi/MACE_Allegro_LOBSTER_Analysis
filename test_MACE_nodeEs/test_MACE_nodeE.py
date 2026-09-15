import os
import numpy as np
import pandas as pd
from ase.io import read
import matplotlib.pyplot as plt

import torch
from icoxx_graph import Icoxx2Graph
from mace.calculators import mace_mp


default_dtype = torch.float32
torch.set_default_dtype(default_dtype)
device = "cuda" if torch.cuda.is_available() else "cpu"
mace_calc = mace_mp(model="medium-mpa-0", device=device,) # default_dtype='float32')
df=pd.read_csv('5_lo_chg_mpInfo_nonmetals_test4.csv')
df['struc_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR.gz'))
df['ICOHP_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster.gz'))
df['ICOBI_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster.gz'))
#df['CHRG_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CHARGE.lobster.gz'))

mace_node_chunks=[]; sum_ICOHP_chunks=[]; sum_ICOBI_chunks=[]
for i, (cont_path, icohp_path, icobi_path) in enumerate(zip(df['struc_dir'].values, 
                                                                      df['ICOHP_dir'].values, 
                                                                      df['ICOBI_dir'].values)):
    ase_atoms=read(cont_path)
    ase_atoms.calc = mace_calc
    ase_atoms.get_potential_energy()
    #print(ase_atoms); exit()
    mace_node_chunks.append(ase_atoms.calc.results["node_energy"])
    #print(type(ase_atoms.calc.results["node_energy"]), ase_atoms.calc.results["node_energy"].shape)
    #exit()

    graph_icoxx = Icoxx2Graph(bond_length_cutoff=5.0).get_graph(
        path_to_structure=os.path.join(cont_path),
        path_to_icohplist=os.path.join(icohp_path),
        path_to_icobilist=os.path.join(icobi_path),
    )
    sum_ICOHP_chunks.append(graph_icoxx.ndata['sum_ICOHPs'].to_numpy())
    sum_ICOBI_chunks.append(graph_icoxx.ndata['sum_ICOBIs'].to_numpy())

print(len(mace_node_chunks), torch.tensor(mace_node_chunks).shape)
mace_node_feats = np.concatenate(mace_node_chunks, axis=0)
sumICOHPs_node_feats = np.concatenate(sum_ICOHP_chunks, axis=0)
sumICOBIs_node_feats = np.concatenate(sum_ICOBI_chunks, axis=0)
#mace_node_feats = torch.cat(mace_node_chunks, dim=0).cpu().numpy()
#sumICOHPs_node_feats = torch.cat(sum_ICOHP_chunks, dim=0).cpu().numpy()
#sumICOBIs_node_feats = torch.cat(sum_ICOBI_chunks, dim=0).cpu().numpy()


plt.scatter(
    mace_node_feats,
    sumICOHPs_node_feats,
    #c=lobster_feats[:, -1],
    s=5,
    #cmap="rainbow"
    )
plt.xlabel(r"MACE_nodeE", fontsize=12)
plt.ylabel(r"sum_ICOHPs", fontsize=12)
#cbar = plt.colorbar(label='Charge Polarity')
#nbar=len(cbar.get_xticks())
#cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=len(cbar.ax.get_yticks())+1))
ax=plt.gca()
nx = len(ax.get_xticks())
#ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
ny = len(ax.get_yticks())
#ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
plt.tight_layout()
#if save_plot:
plt.savefig('maceEs_vs_ICOHPs.jpg', dpi=700) # dpi=dpi)
#plt.show()







exit()
atoms = read("CONTCAR.gz")
device = "cuda" if torch.cuda.is_available() else "cpu"

calc = mace_mp(model="medium-mpa-0", device="cpu", default_dtype='float32')
atoms.calc = calc

# Total energy
E = atoms.get_potential_energy()

# Per-node/site energies
node_E = atoms.calc.results["node_energy"]

print("Total:", E)
print("Node energies:", node_E)
print("Sum:", node_E.sum())
