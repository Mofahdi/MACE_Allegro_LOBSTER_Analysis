from __future__ import annotations

from typing import Any

import dgl
import torch
from ase import Atoms
from ase.io import read

from nequip.data import AtomicDataDict, from_ase
from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform
from nequip.model import ModelFromPackage
from nequip.nn._graph_mixin import SequentialGraphNetwork
from nequip.nn.graph_model import GraphModel
from nequip.nn.misc import SaveForOutput


#############################################
# Load the trained foundation model
#############################################
def get_allegro_model_transforms(
    model_path: str,
    device: torch.device,
    use_model_type_names: bool = True,
) -> tuple[GraphModel, list[Any]]:
    """Load the Allegro model and prepare transformation steps for ASE inputs.

    Args:
        model_path: Path to the serialized Allegro model package.
        device: Target PyTorch device used for inference.
        use_model_type_names: If True, include the model's atom type names in the
            chemical-species mapper.

    Returns:
        A tuple with the loaded graph model and the transform list used to convert
        structures into model-ready graph data.
    """
    model = ModelFromPackage(model_path)
    graph_model = model['sole_model']

    model_type_names_str = graph_model.metadata.get('type_names', '')
    model_type_names = model_type_names_str.split(' ') if model_type_names_str else []

    force_stress_wrapper = graph_model.model
    sequential_net = force_stress_wrapper.func

    modified_modules: dict[str, Any] = {}
    saved_layer_names: list[str] = []
    for name, module in sequential_net.named_children():
        #print(name)
        modified_modules[name] = module
        #if name == "edge_readout":
        #   break
        if name == "per_type_energy_scale_shift":
            break
        # Determine if we should save this layer
        should_save = False
        if should_save:
            save_module = SaveForOutput(
                field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=f"saved_{name}_embeddings",
                irreps_in=module.irreps_out,
            )
            modified_modules[f"save_{name}"] = save_module
            saved_layer_names.append(f"saved_{name}_embeddings")

    # Rebuild model
    model = GraphModel(SequentialGraphNetwork(modified_modules))
    model = model.to(device)
    model.eval()

    transforms: list[Any] = [
        ChemicalSpeciesToAtomTypeMapper(model_type_names) if use_model_type_names else ChemicalSpeciesToAtomTypeMapper([]),
        NeighborListTransform(r_max=7),
    ]

    return model, transforms


def transform_ase_atoms(
    structure_path: str,
    transforms: list[Any],
    device: torch.device,
) -> tuple[Atoms, Any]:
    """Read a structure and apply the configured NequIP transforms.

    Args:
        structure_path: File path to the ASE-readable structure.
        transforms: Sequence of transform functions applied to the atomic data.
        device: Device to which the transformed data should be moved.

    Returns:
        The original ASE atoms object and the transformed atomic data container.
    """
    ase_atoms = read(structure_path)
    data = from_ase(ase_atoms)

    for transform in transforms:
        data = transform(data)
    data = AtomicDataDict.to_(data, device)

    return ase_atoms, data


def create_DGL_g(ase_atoms: Atoms, output: dict[str, Any]) -> dgl.DGLGraph:
    """Build a DGL graph from a model output dictionary.

    Args:
        ase_atoms: ASE structure used to determine the graph node count.
        output: Dictionary containing node and edge attributes returned by the
            Allegro model.

    Returns:
        A DGL graph populated with the model's node and edge features.
    """
    g = dgl.graph((output['edge_index'][0], output['edge_index'][1]), num_nodes=len(list(ase_atoms.symbols)))
    g.ndata['pos']=output['pos']
    g.ndata['atomic_energy']=output['atomic_energy']
    g.ndata['atom_types']=output['atom_types']
    #g.ndata['']=output['']

    g.edata['edge_cell_shift']=output['edge_cell_shift']
    g.edata['edge_vectors']=output['edge_vectors']
    g.edata['edge_lengths']=output['edge_lengths']
    g.edata['normed_edge_lengths']=output['normed_edge_lengths']
    g.edata['edge_features']=output['edge_features']
    g.edata['edge_energy']=output['edge_energy']
    #g.edata['']=output['']

    return g

def intersect_graphs(
    g1: dgl.DGLGraph,
    g2: dgl.DGLGraph,
    edge_key: str,
) -> dgl.DGLGraph:
    """Create a graph containing the edge intersection between two DGL graphs.

    Args:
        g1: Reference graph whose edges are used to define the match set.
        g2: Graph whose edges are filtered to only those matching ``g1``.
        edge_key: Edge attribute used as the matching key.

    Returns:
        A new graph containing the matched edges from ``g2`` and the associated
        node and edge features from both input graphs.
    """
    src1, dst1 = g1.edges()
    src2, dst2 = g2.edges()

    key1 = g1.edata[edge_key]
    key2 = g2.edata[edge_key]

    # Map:
    # (source, destination, edge_key_vector) -> G1 edge ID
    g1_edge_map = {
        (int(s), int(d), tuple(k.tolist())): i
        for i, (s, d, k) in enumerate(zip(src1, dst1, key1))
    }

    g2_keep = []
    g1_matching_ids = []

    for i, (s, d, k) in enumerate(zip(src2, dst2, key2)):

        lookup = (
            int(s),
            int(d),
            tuple(k.tolist())
        )

        if lookup in g1_edge_map:
            g2_keep.append(i)
            g1_matching_ids.append(g1_edge_map[lookup])

    g2_keep = torch.tensor(
        g2_keep,
        dtype=torch.long,
        device=g2.device,
    )

    g1_matching_ids = torch.tensor(
        g1_matching_ids,
        dtype=torch.long,
        device=g1.device,
    )

    # Build graph using the matching edges from G2
    g3 = dgl.graph(
        (
            src2[g2_keep],
            dst2[g2_keep],
        ),
        num_nodes=g2.num_nodes(),
        device=g2.device,
    )

    # --------------------------------------------------
    # Node features
    # --------------------------------------------------

    for name, feat in g2.ndata.items():
        g3.ndata[f"g2_{name}"] = feat

    for name, feat in g1.ndata.items():
        g3.ndata[f"g1_{name}"] = feat

    # --------------------------------------------------
    # Edge features
    # --------------------------------------------------

    for name, feat in g2.edata.items():
        g3.edata[f"g2_{name}"] = feat[g2_keep]

    for name, feat in g1.edata.items():
        g3.edata[f"g1_{name}"] = feat[g1_matching_ids]

    return g3
