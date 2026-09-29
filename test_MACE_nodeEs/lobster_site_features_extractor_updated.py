import os
import pandas as pd
import numpy as np
from pymatgen.io.lobster import Icohplist
from pymatgen.core import Structure

def get_lobster_site_features(structure, icoxxlist, max_length, site_index):
    # Get all neighbors data in a single list using distance based algorithm
    # This is used as a reference to map the icoxx data
    rdf_nb_lst = structure.get_neighbor_list(r=max_length)

    if site_index is None:
        site_indices = list(range(structure.num_sites))
        return_list = True
    elif isinstance(site_index, int):
        site_indices = [site_index]
        return_list = False
        if site_index not in range(structure.num_sites):
            raise ValueError(f"{site_index} is not a valid site index for the structure")
    else:
        site_indices = list(site_index)
        return_list = True

    bond_labels_all = np.array(icoxxlist.icohpcollection._list_labels)
    bond_lengths_all = np.array(icoxxlist.icohpcollection._list_length)
    atoms1_all = np.array(icoxxlist.icohpcollection._list_atom1)
    atoms2_all = np.array(icoxxlist.icohpcollection._list_atom2)
    icoxxs_all = np.array([sum(item.values()) for item in icoxxlist.icohpcollection._list_icohp])
    pairs_all = [[at1, at2] for at1, at2 in zip(atoms1_all, atoms2_all)]
    trans_all = np.array(icoxxlist.icohpcollection._list_translation)

    stats_fns_icohp = {
        #   "max": np.max,
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "sum": np.sum,
    }
    stats_fns_icobi = {
        "max": np.max,
        "mean": np.mean,
        "std": np.std,
    #    "min": np.min,
        "sum": np.sum,
    }

    all_feats=[]
    all_mapped_icoxx_data=[]
    for current_site_index in site_indices:

        # Get ICOXX labels relevant to this site
        site_icohp_dict = (
            icoxxlist.icohpcollection.get_icohp_dict_of_site(
                current_site_index,
                minbondlength=0.5,
                maxbondlength=max_length,
            )
        )

        indices = np.where(
            np.isin(
                bond_labels_all,
                list(site_icohp_dict.keys()),
            )
        )[0]

        # Select from the arrays we already computed
        bond_lengths = bond_lengths_all[indices].tolist()
        atoms1 = atoms1_all[indices].tolist()
        atoms2 = atoms2_all[indices].tolist()

        icoxxs = icoxxs_all[indices].tolist()

        trans = [list(item) for item in trans_all[indices]]

        pairs = [[at1, at2] for at1, at2 in zip(atoms1, atoms2)]

        relevant_site_indices = [
            ix for ix, (org, dest) in enumerate(zip(rdf_nb_lst[0], rdf_nb_lst[1])) if org == current_site_index
        ]

        rdf_distances = [round(rdf_nb_lst[3][ix], 5) for ix in relevant_site_indices]

        rdf_trans = [[int(i) for i in rdf_nb_lst[2][ix]] for ix in relevant_site_indices]

        rdf_pairs = [
            [structure[org].species_string + str(org + 1), structure[dest].species_string + str(dest + 1)]
            for ix, (org, dest) in enumerate(zip(rdf_nb_lst[0], rdf_nb_lst[1])) 
            if ix in relevant_site_indices
            ]


        # Assimilate icoxx data in tuples
        input_icoxx_list = list(zip(pairs, bond_lengths, trans))

        # 0: pair, 1: bond length, 2: translation
        ref_rdf_data = list(zip(rdf_pairs, rdf_distances, rdf_trans))

        # Create a temp dict for faster lookup
        icoxx_dict = {
            (tuple(icoxx_p_d_t[0]), tuple(icoxx_p_d_t[2])): (icoxx_p_d_t[1], icoxxs[idx])
            for idx, icoxx_p_d_t in enumerate(input_icoxx_list)
        }
        # Initialize an empty list to store the data that goes in BWDF computation
        mapped_icoxx_data, missing_interactions = [], []

        # Check if interaction exists in both rdf data / icoxx data, then add to complete data
        for rdf_p_d_t in ref_rdf_data:
            pair_translation_key = (tuple(rdf_p_d_t[0]), tuple(rdf_p_d_t[2]))
            bond_length = rdf_p_d_t[1]
            if pair_translation_key in icoxx_dict:
                icoxx_bond_length, icoxx_value = icoxx_dict[pair_translation_key]
                #if round(np.absolute(bond_length - icoxx_bond_length), 5) <= 1e-5:
                complete_entry = (*rdf_p_d_t, icoxx_value)
                if complete_entry not in mapped_icoxx_data:
                    mapped_icoxx_data.append(complete_entry)
                else:
                    missing_interactions.append(rdf_p_d_t)
            else:
                missing_interactions.append(rdf_p_d_t)

        # Check if the missing interactions are reverse images in ref neighbours data which LOBSTER sometimes eliminates
        for rdf_p_d_t in missing_interactions[:]:
            reversed_translation = tuple(-1 * np.array(rdf_p_d_t[2]))
            reversed_pair = tuple(reversed(rdf_p_d_t[0]))
            reversed_key = (reversed_pair, reversed_translation)

            if reversed_key in icoxx_dict:
                icoxx_bond_length, icoxx_value = icoxx_dict[reversed_key]
                #if round(np.absolute(rdf_p_d_t[1] - icoxx_bond_length), 5) <= 1e-5:
                complete_entry = (*rdf_p_d_t, icoxx_value)
                if complete_entry not in mapped_icoxx_data:
                    mapped_icoxx_data.append((*rdf_p_d_t, icoxx_value))
                    missing_interactions.remove(rdf_p_d_t)
        all_mapped_icoxx_data.append(mapped_icoxx_data)

    return all_mapped_icoxx_data

if __name__ == '__main__':
    icoxxlist = Icohplist(
        filename='ICOHPLIST.lobster.gz',
        are_cobis=False,
        are_coops=False,
    )
    struc = Structure.from_file('CONTCAR.gz')

    all_mapped_icoxx_data=get_lobster_site_features(structure=struc, icoxxlist=icoxxlist, max_length=5, site_index=None)
    print(len(all_mapped_icoxx_data))


