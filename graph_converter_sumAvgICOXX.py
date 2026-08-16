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
from pymatgen.io.lobster import Grosspop

#from lobsterpy.featurize.core import FeaturizeIcoxxlist
#from matgl.graph.converters import GraphConverter

from nbrs_map import icoxx_struc_fxn

# checklist:
# 1- check missing interactions are zero and skip if they are not
# 2- 

warnings.simplefilter("ignore")

float_np = np.float32
float_th = torch.float32

int_np = np.int32
int_th = torch.int32

default_dtype = torch.float32
torch.set_default_dtype(default_dtype)


class Icoxx2Graph:
    """Construct a DGL graph from Pymatgen Structure and Icohplist objects."""
    def __init__(
        self,
        bond_length_cutoff: float | None = None,
        noise: float = -0.1,
        perc: float = 0.1,
    ):
        """
        Args:
        ----------
        bond_length_cutoff: bond length cutoff radius for graph representation
        """
        #self.element_types = tuple(element_types)
        self.bond_length_cutoff = bond_length_cutoff
        self.noise = noise
        self.perc = perc
    
    def get_graph(self, 
                  path_to_structure: str,
                  path_to_icoxxlist: str, 
                  ) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Get a DGL graph from an input Structure and ICOXXLIST files.
        
        :param path_to_structure: path to structure 
        :param path_to_icohplist: path to icohplist.lobster
        :return:
            g: DGL graph
            lat: lattice for periodic systems
            state_attr: state features
        """

        structure=read(path_to_structure)

        mapped_icoxx_data_reverse, missing_interactions_reverse=icoxx_struc_fxn(CONTCAR_path=path_to_structure, 
                                                                            r_cutoff = self.bond_length_cutoff, 
                                                                            ICOXX_path = path_to_icoxxlist, 
                                                                            test_bond_len=False, missing_check=False, 
                                                                            missing_cutoff=4, missing_icoxx=0)

        src_ids = []
        dst_ids = []
        icoxxs = []
        images = []
        for _, src_dst_dist_img_ic in enumerate(mapped_icoxx_data_reverse):
            src=src_dst_dist_img_ic[0][0] 
            dst=src_dst_dist_img_ic[0][1] 
            img = np.array(src_dst_dist_img_ic[2])
            icoxx= src_dst_dist_img_ic[-1]
            src_ids.append(src); dst_ids.append(dst); images.append(img); icoxxs.append(icoxx); 
        
        lattice_matrix=np.expand_dims(structure.cell.array, axis=0)
        g = self.get_graph_from_processed_structure(
            structure,
            src_ids,
            dst_ids,
            images,
            lattice_matrix,
            structure.get_scaled_positions(),
            icoxxs,
        )

        sum_ICOXX=torch.tensor([]); avg_ICOXX=torch.tensor([])
        for i in range(g.num_nodes()):
            eids = g.in_edges(i, form="eid")
            # Retrieve the edge values
            edge_values = g.edata["ICOXXs"][eids]
            # Filter based on a condition
            mask = ((g.edata["ICOXXs"][eids] < self.noise) &
                        (g.edata["ICOXXs"][eids] < torch.min(g.edata["ICOXXs"]) * self.perc))
            filtered_values = edge_values[mask]
            filtered_eids = eids[mask]
            #print(g.ndata['Z'][i], len(filtered_values), torch.mean(filtered_values))
            #print(sum_ICOXX.shape, torch.sum(filtered_values).unsqueeze(0).shape)
            sum_ICOXX = torch.cat([sum_ICOXX, torch.sum(filtered_values).unsqueeze(0)])
            avg_ICOXX = torch.cat([avg_ICOXX, torch.mean(filtered_values).unsqueeze(0)])
        g.ndata['sum_ICOXX']=sum_ICOXX
        g.ndata['avg_ICOXX']=avg_ICOXX
        #print(sum_ICOXX, avg_ICOXX)
        return g

    def get_graph_from_processed_structure(
        self,
        structure,
        src_id,
        dst_id,
        images,
        lattice_matrix,
        frac_coords,
        icoxxs,
    ) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Construct a dgl graph from processed structure and bond strength information.

        Args:
            structure: Input crystals or molecule of pymatgen structure or molecule types.
            src_id: site indices for starting point of bonds.
            dst_id: site indices for destination point of bonds.
            images: the periodic image offsets for the bonds.
            lattice_matrix: lattice information of the structure.
            frac_coords: Fractional coordinates of all atoms in the structure. Note: Cartesian coordinates for molecule

        Returns:
            DGLGraph object, state_attr

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
        g.edata["ICOXXs"] = torch.tensor(icoxxs, dtype=default_dtype) #* -1

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
    g = Icoxx2Graph(bond_length_cutoff=5, noise=0, perc=0).get_graph(path_to_structure='CONTCAR.gz',
                path_to_icoxxlist='ICOHPLIST.lobster.gz',
            )    
    #print(g.edata['bond_strengths'].shape, g.edata["edge_mask"].shape)
    #print(g.edata['bond_strengths'][g.edata["edge_mask"]].shape)


