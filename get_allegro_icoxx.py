import os
import pandas as pd
import torch, dgl
from allegroEij_merge import get_allegro_model_transforms, transform_ase_atoms, create_DGL_g, intersect_graphs
from icoxx_graph_converter import Icoxx2Graph

#df=pd.DataFrame({'id': ids, 'mat_dir': paths})
df=pd.read_csv('5_lo_chg_mpInfo_nonmetals.csv')
df['struc_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR.gz'))
df['ICOHP_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster.gz'))
df['ICOBI_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster.gz'))


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model, transforms=get_allegro_model_transforms(model_path="Allegro-OAM-L-0.1.nequip.zip", device=device)

Eij_chunks=[]; icoxx_chunks=[]
for i, (cont_path, icoxx_path) in enumerate(zip(df['struc_dir'].values, df['ICOBI_dir'].values)):
    ase_atoms, data = transform_ase_atoms(structure_path=cont_path, transforms=transforms, device=device)
    with torch.no_grad():
        output = model(data)

    g_allegro=create_DGL_g(ase_atoms=ase_atoms, output=output)
    g_icoxx = Icoxx2Graph(bond_length_cutoff=5).get_graph(path_to_structure=cont_path,
                    path_to_icoxxlist=icoxx_path,
                )
    g_intersect=intersect_graphs(g1=g_allegro, g2=g_icoxx, edge_key="edge_cell_shift")
    g_intersect.edata['g2_ICOXXs'], g_intersect.edata['g1_edge_energy']

    Eij_chunks.append(g_intersect.edata['g1_edge_energy'].squeeze()); icoxx_chunks.append(g_intersect.edata['g2_ICOXXs'])
    #print(g_intersect.edata['g1_edge_energy'].shape, g_intersect.edata['g2_ICOXXs'].shape)

Eijs = torch.cat(Eij_chunks, dim=0)
icoxxs = torch.cat(icoxx_chunks, dim=0)

corr = torch.corrcoef(torch.stack([Eijs, icoxxs]))[0, 1]
print(corr)

pd.DataFrame({'ICOXX': icoxxs, 'Eij': Eijs}).to_csv('Eij_icobi_all.csv', index=False)

import matplotlib.pyplot as plt
plt.scatter(icoxxs, Eijs)
plt.xlabel('ICOBI')
plt.ylabel('Eij')
plt.savefig('eij_vs_icobi_all_data.png')

