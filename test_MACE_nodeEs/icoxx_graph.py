from __future__ import annotations

import warnings
from collections.abc import Sequence

import dgl
import dgl.function as fn
import numpy as np
import torch
from ase import Atoms
from ase.io import read

from nbrs_map import icoxx_struc_fxn

warnings.simplefilter("ignore")

default_dtype: torch.dtype = torch.float32
torch.set_default_dtype(default_dtype)


class Icoxx2Graph:
    """Construct a DGL graph from an ASE structure and ICOXX bonding data."""

    def __init__(self, bond_length_cutoff: float | None = None) -> None:
        """Initialize the graph builder.

        Args:
            bond_length_cutoff: Optional cutoff distance used when matching
                bonded atom pairs.
        """
        self.bond_length_cutoff: float | None = bond_length_cutoff

    def get_graph(
        self,
        path_to_structure: str,
        path_to_icohplist: str,
        path_to_icobilist: str,
    ) -> dgl.DGLGraph:
        """Build a DGL graph from a structure file and ICOHP/ICOBI list files.

        Args:
            path_to_structure: Path to the atomic structure file, typically a
                CONTCAR-like file.
            path_to_icohplist: Path to the ICOHP list file.
            path_to_icobilist: Path to the ICOBI list file.

        Returns:
            A DGL graph whose nodes contain atomic attributes and whose edges
            contain bond features such as ICOHP, ICOBI, periodic offsets, and
            bond vectors/distances.
        """
        atoms: Atoms = read(path_to_structure)

        mapped_icohp_data_reverse, _ = icoxx_struc_fxn(
            CONTCAR_path=path_to_structure,
            r_cutoff=self.bond_length_cutoff,
            ICOXX_path=path_to_icohplist,
            test_bond_len=False,
        )

        mapped_icobi_data_reverse, _ = icoxx_struc_fxn(
            CONTCAR_path=path_to_structure,
            r_cutoff=self.bond_length_cutoff,
            ICOXX_path=path_to_icobilist,
            test_bond_len=False,
        )

        src_ids: list[int] = []
        dst_ids: list[int] = []
        images: list[np.ndarray] = []
        icohps: list[float] = []
        icobis: list[float] = []

        for idx, src_dst_dist_img_ic in enumerate(mapped_icohp_data_reverse):
            src = src_dst_dist_img_ic[0][0]
            dst = src_dst_dist_img_ic[0][1]
            img = np.array(src_dst_dist_img_ic[2])
            icohp = src_dst_dist_img_ic[-1]

            src_ids.append(src)
            dst_ids.append(dst)
            images.append(img)
            icohps.append(icohp)

            if (
                src == mapped_icobi_data_reverse[idx][0][0]
                and dst == mapped_icobi_data_reverse[idx][0][1]
                and np.array_equal(img, mapped_icobi_data_reverse[idx][2])
            ):
                icobis.append(mapped_icobi_data_reverse[idx][-1])

        lattice_matrix = np.expand_dims(atoms.cell.array, axis=0)

        return self.get_graph_from_atoms(
            atoms=atoms,
            src_id=src_ids,
            dst_id=dst_ids,
            images=images,
            lattice_matrix=lattice_matrix,
            frac_coords=atoms.get_scaled_positions(),
            icohps=icohps,
            icobis=icobis,
        )

    def get_graph_from_atoms(
        self,
        atoms: Atoms,
        src_id: Sequence[int],
        dst_id: Sequence[int],
        images: Sequence[np.ndarray],
        lattice_matrix: np.ndarray,
        frac_coords: np.ndarray,
        icohps: Sequence[float],
        icobis: Sequence[float],
    ) -> dgl.DGLGraph:
        """Construct a DGL graph from processed atomic and bond-level data.

        Args:
            atoms: ASE structure object containing atomic species and positions.
            src_id: Source atom indices for each bond.
            dst_id: Destination atom indices for each bond.
            images: Periodic image offsets for each bond.
            lattice_matrix: Lattice matrix of the periodic cell.
            frac_coords: Fractional coordinates of all atoms in the structure.
            icohps: ICOHP values associated with each bond.
            icobis: ICOBI values associated with each bond.

        Returns:
            A DGL graph with atom and edge attributes for downstream ML or
            bonding analysis workflows.
        """
        u = torch.tensor(src_id, dtype=torch.long)
        v = torch.tensor(dst_id, dtype=torch.long)
        graph: dgl.DGLGraph = dgl.graph((u, v),) #num_nodes=len(atoms))

        lattice = torch.tensor(np.asarray(lattice_matrix), dtype=default_dtype)
        graph.ndata["Z"] = torch.tensor(atoms.get_atomic_numbers())
        graph.ndata["pos"] = torch.as_tensor(atoms.positions.copy(), dtype=default_dtype)
        graph.ndata["frac_coords"] = torch.tensor(frac_coords, dtype=default_dtype)

        graph.edata["ICOHPs"] = torch.tensor(icohps, dtype=default_dtype)
        graph.edata["ICOBIs"] = torch.tensor(icobis, dtype=default_dtype)

        # sum_filtered_icohps=[]
        # sum_filtered_icobis=[]
        # mean_filtered_icohps=[]
        # mean_filtered_icobis=[]
        # noise = -0.1
        # perc = 0.1
        # for i in range(graph.num_nodes()):
        #     #eids = graph.in_edges(i, form="eid")
        #     eids = graph.out_edges(i, form="eid")
        #     # Retrieve the edge values
        #     edge_ICOHPs = graph.edata["ICOHPs"][eids]
        #     edge_ICOBIs = graph.edata["ICOBIs"][eids]
        #     # Filter based on a condition
        #     #mask = ((graph.edata["ICOHPs"][eids] < noise) &
        #      #           (graph.edata["ICOHPs"][eids] < torch.min(graph.edata["ICOHPs"]) * perc))
        #     mask_COHP = graph.edata["ICOHPs"][eids] < torch.min(graph.edata["ICOHPs"]) * perc
        #     mask_COBI = graph.edata["ICOBIs"][eids] > torch.max(graph.edata["ICOBIs"]) * perc
        #     #filtered_eids = eids[mask_COHP]
        #     filtered_ICOHPs = edge_ICOHPs[mask_COHP]
        #     filtered_ICOBIs = edge_ICOBIs[mask_COBI]
        #     mean_filtered_icohps.append(torch.mean(filtered_ICOHPs)); 
        #     mean_filtered_icobis.append(torch.mean(filtered_ICOBIs))
        #     sum_filtered_icohps.append(torch.sum(filtered_ICOHPs)); 
        #     sum_filtered_icobis.append(torch.sum(filtered_ICOBIs))

        # graph.ndata['mean_maskedICOHPs'] = torch.tensor(mean_filtered_icohps)
        # graph.ndata['mean_maskedICOBIs'] = torch.tensor(mean_filtered_icobis)
        # graph.ndata['sum_maskedICOHPs'] = torch.tensor(sum_filtered_icohps)
        # graph.ndata['sum_maskedICOBIs'] = torch.tensor(sum_filtered_icobis)

        graph.update_all(fn.copy_e("ICOHPs", "m"), 
                         fn.sum("m", "sum_ICOHPs"))
        graph.update_all(fn.copy_e("ICOBIs", "m"), 
                         fn.sum("m", "sum_ICOBIs"))
        #print(graph.ndata['sum_ICOBIs'].shape, graph.ndata['mean_maskedICOHPs'].shape)

        pbc_offset = torch.tensor(np.asarray(images), dtype=default_dtype)
        graph.edata["edge_cell_shift"] = pbc_offset
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["edge_cell_shift"], lattice[0])

        dst_pos = graph.ndata["pos"][graph.edges()[1]] + graph.edata["pbc_offshift"]
        src_pos = graph.ndata["pos"][graph.edges()[0]]
        bond_vec = dst_pos - src_pos
        bond_dist = torch.norm(bond_vec, dim=1)

        graph.edata["bond_vec"] = bond_vec
        graph.edata["bond_dist"] = bond_dist

        return graph


if __name__ == "__main__":
    graph = Icoxx2Graph(bond_length_cutoff=5.0).get_graph(
        path_to_structure="CONTCAR.gz",
        path_to_icohplist="ICOHPLIST.lobster.gz",
        path_to_icobilist="ICOBILIST.lobster.gz",
    )


