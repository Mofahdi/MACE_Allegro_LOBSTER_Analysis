import os
import pandas as pd
import torch, dgl
from allegroEij_merge import get_allegro_model_transforms, transform_ase_atoms, create_DGL_g, intersect_graphs
from graph_converter import Icoxx2Graph

ids=['Cs3Bi2I9', 'CsCu3S2', 'CsPbBr3']
paths=['/home/malfahdi/RZ-Dienste/hpc-user/malfahdi/tests/test_allegro/Cs3Bi2I9', 
       '/home/malfahdi/RZ-Dienste/hpc-user/malfahdi/tests/test_allegro/CsCu3S2', 
       '/home/malfahdi/RZ-Dienste/hpc-user/malfahdi/tests/test_allegro/CsPbBr3']
df=pd.DataFrame({'id': ids, 'mat_path': paths})
df['struc_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR'))
df['ICOHP_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster'))
df['ICOBI_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
model, transforms=get_allegro_model_transforms(model_path="Allegro-OAM-L-0.1.nequip.zip", device=device)

Eij_chunks=[]; icoxx_chunks=[]
for i, (cont_path, icoxx_path) in enumerate(zip(df['struc_dir'].values, df['ICOHP_dir'].values)):
    ase_atoms, data = transform_ase_atoms(structure_path=cont_path, transforms=transforms, device=device)
    with torch.no_grad():
        output = model(data)

    g_allegro=create_DGL_g(ase_atoms=ase_atoms, output=output)
    g_icoxx = Icoxx2Graph(bond_length_cutoff=5).get_graph(path_to_structure=cont_path,
                    path_to_icoxxlist=icoxx_path,
                ).to(device)
    g_intersect=intersect_graphs(g1=g_allegro, g2=g_icoxx, edge_key="edge_cell_shift")
    g_intersect.edata['g2_ICOXXs'], g_intersect.edata['g1_edge_energy']

    Eij_chunks.append(g_intersect.edata['g1_edge_features']); icoxx_chunks.append(g_intersect.edata['g2_ICOXXs'].unsqueeze(-1))
    #print(g_intersect.edata['g1_edge_energy'].shape, g_intersect.edata['g2_ICOXXs'].shape)
    print(g_intersect.edata['g1_edge_features'].shape, g_intersect.edata['g2_ICOXXs'].unsqueeze(-1).shape)
    #exit()

X = torch.cat(Eij_chunks, dim=0).cpu().numpy()
icoxxs = torch.cat(icoxx_chunks, dim=0).cpu().numpy()
#print(Eijs.shape, icoxxs.shape)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Standardize each edge feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("PCA shape:", X_pca.shape)
# (50000, 2)

print("Explained variance:",
      pca.explained_variance_ratio_)

'''corr = torch.corrcoef(torch.stack([Eijs, icoxxs]))[0, 1]
print(corr)
pd.DataFrame({'ICOXX': icoxxs, 'Eij': Eijs}).to_csv('Eij_icoxx_low_LTC.csv', index=False)

import matplotlib.pyplot as plt
plt.scatter(icoxxs, Eijs)
plt.savefig('eij_vs_icoxx_low_LTC.pdf')
plt.show()
'''

