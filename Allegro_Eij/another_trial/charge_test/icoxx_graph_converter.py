from __future__ import annotations
import dgl
import numpy as np
import torch
#import matgl
import warnings
from tqdm import trange

from ase import Atoms, Atoms
from ase.neighborlist import neighbor_list
from ase.io import read
from jarvis.core.specie import Specie
from jarvis.core.specie import get_node_attributes

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.lobster import Charge

from nbrs_map import icoxx_struc_fxn

warnings.simplefilter("ignore")

default_dtype = torch.float32
torch.set_default_dtype(default_dtype)

class Icoxx2Graph:
    """Construct a DGL graph from Pymatgen Structure and Icohplist objects."""
    def __init__(
        self,
        bond_length_cutoff: float | None = None,
    ):
        """
        Args:
        ----------
        bond_length_cutoff: bond length cutoff radius for graph representation
        """
        #self.element_types = tuple(element_types)
        self.bond_length_cutoff = bond_length_cutoff
    
    def get_graph(self, 
                  path_to_structure: str,
                  path_to_icohplist: str, 
                  path_to_icobilist: str, 
                  path_to_chagcar: str,
                  ) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Get a DGL graph from an input Structure and ICOXXLIST files.
        args:
            param path_to_structure: path to structure, preferably "CONTCAR"
            param path_to_icohplist: path to icohplist.lobster
        :return:
            g: DGL graph
            lat: lattice for periodic systems
        """

        structure=read(path_to_structure)

        mapped_icohp_data_reverse, missing_interactions_reverse=icoxx_struc_fxn(CONTCAR_path=path_to_structure, 
                                                                            r_cutoff = self.bond_length_cutoff, 
                                                                            ICOXX_path = path_to_icohplist, 
                                                                            test_bond_len=False,)

        mapped_icobi_data_reverse, missing_interactions_reverse=icoxx_struc_fxn(CONTCAR_path=path_to_structure, 
                                                                            r_cutoff = self.bond_length_cutoff, 
                                                                            ICOXX_path = path_to_icobilist, 
                                                                            test_bond_len=False,)

        src_ids = []
        dst_ids = []
        images = []
        icohps = []
        icobis = []
        for _, src_dst_dist_img_ic in enumerate(mapped_icohp_data_reverse):
            src=src_dst_dist_img_ic[0][0] 
            dst=src_dst_dist_img_ic[0][1] 
            img = np.array(src_dst_dist_img_ic[2])
            icohp= src_dst_dist_img_ic[-1]
            src_ids.append(src); dst_ids.append(dst); images.append(img); icohps.append(icohp); 

            if (src==mapped_icobi_data_reverse[_][0][0] and dst==mapped_icobi_data_reverse[_][0][1] and np.array_equal(img,mapped_icobi_data_reverse[_][2])): 
                icobis.append(mapped_icobi_data_reverse[_][-1]); 
        lattice_matrix=np.expand_dims(structure.cell.array, axis=0)
        
        g = self.get_graph_from_processed_structure(
            structure,
            src_ids,
            dst_ids,
            images,
            lattice_matrix,
            structure.get_scaled_positions(),
            icohps,
            icobis, 
        )
        chargeobj_loew = Charge(filename=path_to_chagcar).loewdin
        g.ndata['chg_loe']=torch.tensor(chargeobj_loew, dtype=default_dtype).unsqueeze(-1)

        # Node features: [num_nodes, feature_dim]
        h = g.ndata["chg_loe"]
        # Get source and destination node IDs for every edge
        src, dst = g.edges()
        # Compute edge features
        chg_features = torch.abs(h[src] - h[dst])
        g.edata['chg_loe_abs']=chg_features
        return g

    def get_graph_from_processed_structure(
        self,
        structure,
        src_id,
        dst_id,
        images,
        lattice_matrix,
        frac_coords,
        icohps,
        icobis,
    ) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Construct a dgl graph from processed structure and bond strength information.

        Args:
            structure: Input crystals or molecule of pymatgen structure or molecule types.
            src_id: site indices for starting point of bonds.
            dst_id: site indices for destination point of bonds.
            images: the periodic image offsets for the bonds.
            lattice_matrix: lattice information of the structure.
            frac_coords: Fractional coordinates of all atoms in the structure. Note: Cartesian coordinates for molecule
            icoxxs: icoxx values
        Returns:
            DGLGraph object

        """
        u, v = torch.tensor(src_id), torch.tensor(dst_id)
        g = dgl.graph((u, v), num_nodes=len(structure))

        lattice = torch.tensor(np.array(lattice_matrix), dtype=default_dtype)

        g.ndata['Z']=torch.tensor(structure.get_atomic_numbers())
        positions = torch.as_tensor(structure.positions.copy(), dtype=default_dtype);
        g.ndata["pos"] = positions

        # TODO: Need to check if the variable needs to be double or float, now use float
        g.ndata["frac_coords"] = torch.tensor(frac_coords, dtype=default_dtype)
        # Note: pbc_ offshift and pos needs to be float64 to handle cases where bonds are exactly at cutoff
        g.edata["ICOHPs"] = torch.tensor(icohps, dtype=default_dtype) #* -1
        g.edata["ICOBIs"] = torch.tensor(icobis, dtype=default_dtype) #* -1

        pbc_offset = torch.tensor(images, dtype=default_dtype)
        g.edata["edge_cell_shift"] = pbc_offset
        g.edata["pbc_offshift"] = torch.matmul(g.edata["edge_cell_shift"], lattice[0])
        dst_pos = g.ndata["pos"][g.edges()[1]] + g.edata["pbc_offshift"]
        src_pos = g.ndata["pos"][g.edges()[0]]
        bond_vec = dst_pos - src_pos
        bond_dist = torch.norm(bond_vec, dim=1)
        g.edata["bond_vec"] = bond_vec
        g.edata["bond_dist"] = bond_dist

        return g



if __name__=='__main__':
    g = Icoxx2Graph(bond_length_cutoff=5,).get_graph(path_to_structure='CONTCAR.gz',
                path_to_icohplist='ICOHPLIST.lobster.gz', path_to_icobilist='ICOBILIST.lobster.gz',
            )    
    #print(g.edata['bond_strengths'].shape, g.edata["edge_mask"].shape)
    #print(g.edata['bond_strengths'][g.edata["edge_mask"]].shape)


