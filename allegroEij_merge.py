import os
import torch
import dgl
from ase.io import read

from nequip.data import AtomicDataDict, from_ase
from nequip.data.transforms import NeighborListTransform

from nequip.model import ModelFromPackage
from nequip.nn.misc import SaveForOutput
from nequip.nn._graph_mixin import SequentialGraphNetwork
from nequip.nn.graph_model import GraphModel
from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform


#############################################
# Load your trained model
#############################################
def get_allegro_model_transforms(model_path, device, model_type_names: bool = True):
    model = ModelFromPackage(model_path)
    graph_model = model['sole_model']

    model_type_names_str = graph_model.metadata.get('type_names', '')
    model_type_names = model_type_names_str.split(' ') if model_type_names_str else []

    # Get sequential network
    force_stress_wrapper = graph_model.model
    sequential_net = force_stress_wrapper.func

    # Insert SaveForOutput modules
    modified_modules = {}
    saved_layer_names = []
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
                    irreps_in=module.irreps_out
                    )
            modified_modules[f"save_{name}"] = save_module
            saved_layer_names.append(f"saved_{name}_embeddings")

    # Rebuild model
    model = GraphModel(SequentialGraphNetwork(modified_modules))
    model = model.to(device)
    model.eval()

    transforms = [
        ChemicalSpeciesToAtomTypeMapper(model_type_names),
        NeighborListTransform(r_max=7)
        ]

    return model, transforms

def transform_ase_atoms(structure_path, transforms, device):
    ase_atoms=read(structure_path)
    data = from_ase(
        ase_atoms,)
    
    for transform in transforms:
        data = transform(data)
    data = AtomicDataDict.to_(data, device)

    return ase_atoms, data


def create_DGL_g(ase_atoms, output):
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

def intersect_graphs(g1, g2, edge_key):
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

"""
exit()
for name, module in sequential_net.named_children():
    modified_modules[name] = module

    # Determine if we should save this layer
    should_save = False
    #if layers_to_save == "all":
     #   should_save = "type_embed" in name or "layer" in name
    #elif layers_to_save == "layer":
     #   should_save = "layer" in name and "convnet" in name
    #elif layers_to_save == "type_embed":
     #   should_save = "type_embed" in name

    if should_save:
        save_module = SaveForOutput(
                field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=f"saved_{name}_embeddings",
                irreps_in=module.irreps_out
                )
        modified_modules[f"save_{name}"] = save_module
        saved_layer_names.append(f"saved_{name}_embeddings")

"""
