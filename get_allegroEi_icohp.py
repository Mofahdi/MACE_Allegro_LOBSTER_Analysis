import os
import pandas as pd
import torch, dgl
from allegroEij_merge import get_allegro_model_transforms, transform_ase_atoms, create_DGL_g, intersect_graphs
from graph_converter import Icoxx2Graph

ids=['Cs3Bi2I9', 'CsCu3S2', 'CsPbBr3']
paths=['/home/malfahdi/RZ-Dienste/hpc-user/malfahdi/tests/test_allegro/Cs3Bi2I9', 
       '/home/malfahdi/RZ-Dienste/hpc-user/malfahdi/tests/test_allegro/CsCu3S2', 
       '/home/malfahdi/RZ-Dienste/hpc-user/malfahdi/tests/test_allegro/CsPbBr3']
#df=pd.DataFrame({'id': ids, 'mat_path': paths})
df=pd.read_csv('5_lo_chg_mpInfo_nonmetals_test3.csv')
df['struc_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR.gz'))
df['ICOHP_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster.gz'))
df['ICOBI_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster.gz'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
model, transforms=get_allegro_model_transforms(model_path="Allegro-OAM-L-0.1.nequip.zip", device=device)

Ei_chunks=[]; n_icoxx_chunks=[]
for i, (cont_path, icoxx_path) in enumerate(zip(df['struc_dir'].values, df['ICOHP_dir'].values)):
    ase_atoms, data = transform_ase_atoms(structure_path=cont_path, transforms=transforms, device=device)
    with torch.no_grad():
        output = model(data)

    g_allegro=create_DGL_g(ase_atoms=ase_atoms, output=output)
    g_icoxx = Icoxx2Graph(bond_length_cutoff=5, noise=-0.0, perc=0.0).get_graph(path_to_structure=cont_path,
                    path_to_icoxxlist=icoxx_path,).to(device)
    g_intersect=intersect_graphs(g1=g_allegro, g2=g_icoxx, edge_key="edge_cell_shift")
    #g_intersect.ndata['g2_sum_ICOXX'], g_intersect.ndata['g1_atomic_energy']
    #print(g_intersect.ndata['g1_atomic_energy'].shape, g_intersect.ndata['g2_sum_ICOXX'].unsqueeze(-1).shape); exit()
    #try:
    Ei_chunks.append(g_intersect.ndata['g1_atomic_energy']); n_icoxx_chunks.append(g_intersect.ndata['g2_sum_ICOXX'].unsqueeze(-1))
    #except: pass

Eis = torch.cat(Ei_chunks, dim=0).squeeze()
icoxxs = torch.cat(n_icoxx_chunks, dim=0).squeeze()
#print(Eis.shape, icoxxs.shape)
corr = torch.corrcoef(torch.stack([Eis, icoxxs]))[0, 1]
print(corr)

pd.DataFrame({'ICOXX': icoxxs.cpu(), 'Ei': Eis.cpu()}).to_csv('Ei_sumICOHP_tes.csv', index=False)

import matplotlib.pyplot as plt
plt.scatter(icoxxs.cpu(), Eis.cpu())
plt.xlabel('ICOHP (eV)')
plt.ylabel('Ei')
plt.savefig('ei_vs_sumICOHP_test.png')
