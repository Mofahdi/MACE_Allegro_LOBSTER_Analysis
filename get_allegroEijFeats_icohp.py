import os
import pandas as pd
import torch, dgl
from allegroEij_merge import get_allegro_model_transforms, transform_ase_atoms, create_DGL_g, intersect_graphs
from graph_converter import Icoxx2Graph

df=pd.read_csv('5_lo_chg_mpInfo_nonmetals.csv')
df['struc_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR.gz'))
df['ICOHP_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster.gz'))
df['ICOBI_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster.gz'))


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
    #print(g_intersect.edata['g1_edge_features'].shape, g_intersect.edata['g2_ICOXXs'].unsqueeze(-1).shape)
    #exit()

X = torch.cat(Eij_chunks, dim=0).cpu().numpy()
icoxxs = torch.cat(icoxx_chunks, dim=0).squeeze().cpu().numpy()


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

import matplotlib.pyplot as plt
plt.figure(figsize=(4,3.5))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=icoxxs,
    s=5,
    #cmap="rainbow"
    )
# [0.0734705  0.05522907]test2 dataset
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="-ICOHP (eV)")
plt.tight_layout()
plt.savefig('Eij_latent_space_ICOHP.png')
plt.show()

