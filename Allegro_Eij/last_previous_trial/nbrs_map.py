import os, numpy as np, pandas as pd

from pymatgen.core import Structure
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.io.lobster.outputs import Icohplist

from ase import Atoms, Atoms
from ase.neighborlist import neighbor_list
from ase.io import read



def get_icoxx_pairs_S_l(ICOXX_file_path):
    """ 
    Args:
        path to icoxxlist.lobster
    return:
        icoxx_info_list: (pairs, bond length, images)
        icoxx_info_list: {(pairs, images): bond length, ICOXX}
    """

    icohp_list=Icohplist(filename= ICOXX_file_path);
    list_atom1=list(icohp_list._icohpcollection._list_atom1)
    list_atom2=list(icohp_list._icohpcollection._list_atom2)
    list_lens=list(icohp_list._icohpcollection._list_length)
    list_trans=list(list(trans) for trans in icohp_list._icohpcollection._list_translation)
    list_icohp=[float(sum(cohp_dict.values())) for cohp_dict in icohp_list._icohpcollection._list_icohp] 

    list_atom1_src_index_lobster=[int("".join(filter(str.isdigit, at1))) - 1 for at1 in list_atom1]
    list_atom2_dst_index_lobster=[int("".join(filter(str.isdigit, at2))) - 1 for at2 in list_atom2]

    arr_lens=np.asarray(list_lens)
    pairs_lobster = [[at1, at2] for at1, at2 in zip(list_atom1_src_index_lobster, list_atom2_dst_index_lobster)]
    icoxx_info_list=list(zip(pairs_lobster, arr_lens, list_trans))
    icoxx_info_dict = {
        (tuple(icoxx_info[0]), tuple(icoxx_info[2])): (icoxx_info[1], list_icohp[idx])
        for idx, icoxx_info in enumerate(icoxx_info_list)
        }

    return icoxx_info_list, icoxx_info_dict

def get_struc_pairs_S_l(CONTCAR_path, r_max):
    """ 
    Args:
        path to CONTCAR
        cutoff radius
    return:
        ase_info_list: (pairs, bond length, images)
    """

    struc=read(CONTCAR_path)
    edge_src, edge_dst, edge_shift, edge_len = neighbor_list("ijSd", a=struc, cutoff=r_max, self_interaction=False)
    pairs_ase=[[int(at1), int(at2)] for at1, at2 in zip(edge_src, edge_dst)]
    ase_struc_list=list(zip(pairs_ase, edge_len, edge_shift.tolist()))

    return ase_struc_list



def compare_pairs(struc_list_pairs_S_l, icoxx_dict_pairsS_l, test_bond_len=False):
    """ 
    Args:
        struc_list_pairs_S_l: ase_info_list: pairs, bond length, images
        icoxx_dict_pairsS_l: (pairs, images): [bond length, icoxx]
        test_bond_len: whether to map ase and icoxx interactions using bond length as criteria along w/ pairs and images
    return:
        mapped_icoxx_data: (ase_info_list, icoxx)
        missing_interactions: (ase_info_list)
    """
    mapped_icoxx_data, missing_interactions = [], []
    for ind_ase, ase_site in enumerate(struc_list_pairs_S_l):
        comp_pair=(tuple(ase_site[0]), tuple(ase_site[2]))
        if comp_pair in icoxx_dict_pairsS_l:
            icoxx_bond_length, icoxx_value = icoxx_dict_pairsS_l[comp_pair]
            if test_bond_len:
                if round(np.absolute(ase_site[1] - icoxx_bond_length), 5) <= 1e-3:
                #if np.absolute(ase_site[1] - icoxx_bond_length) <= 1e-3: # if 1e-6, mapped_icoxx_data = 0
                    all_info = (*ase_site, icoxx_value)
                    mapped_icoxx_data.append(all_info)
                else: 
                    missing_interactions.append(ase_site)
            else:
                all_info = (*ase_site, icoxx_value)
                mapped_icoxx_data.append(all_info)
        else: 
            missing_interactions.append(ase_site)
    return mapped_icoxx_data, missing_interactions

def reverse_missed_inter(mapped_icoxx_data, missing_interactions, icoxx_dict_pairsS_l, test_bond_len=False):
    """ 
    function to assign the same ICOXX with reversed pairs (i.e, pair (1,2): icoxx --> reversed pair (2,1): icoxx)
    Args:
        mapped_icoxx_data: (ase_info_list, icoxx)
        missing_interactions: (ase_info_list)
        icoxx_dict_pairsS_l: (pairs, images): [bond length, icoxx]
        test_bond_len: whether to map ase and icoxx interactions using bond length as criteria along w/ pairs and images
    return:
        mapped_icoxx_data: updated (ase_info_list, icoxx)
        missing_interactions: updated (ase_info_list) after reversing interactions
    """
    for ase_site in missing_interactions[:]:
        reversed_translation = tuple(map(int, -1 * np.array(ase_site[2])))
        reversed_pair = tuple(reversed(ase_site[0]))
        reversed_key = (reversed_pair, reversed_translation)
        if reversed_key in icoxx_dict_pairsS_l:
            icoxx_bond_length, icoxx_value = icoxx_dict_pairsS_l[reversed_key]
            if test_bond_len:
                if round(np.absolute(ase_site[1] - icoxx_bond_length), 5) <= 1e-3:
                #if np.absolute(ase_site[1] - icoxx_bond_length) <= 1e-6: # (maybe) if 1e-6, mapped_icoxx_data = 0
                    #print('ran')
                    all_info = (*ase_site, icoxx_value)
                    if all_info not in mapped_icoxx_data:
                        mapped_icoxx_data.append(all_info)
                        missing_interactions.remove(ase_site)
            else:
                all_info = (*ase_site, icoxx_value)
                if all_info not in mapped_icoxx_data:
                    mapped_icoxx_data.append(all_info)
                    missing_interactions.remove(ase_site)
    return mapped_icoxx_data, missing_interactions

def zero_missing_icoxx_cutoff(mapped_icoxx_data, missing_interactions, cutoff=4):
    """ 
    zero icoxx the missing interactions >4 A (was not used and probably not needed)
    Args:
        mapped_icoxx_data: (ase_info_list, icoxx)
        missing_interactions: (ase_info_list)
        cutoff: any missing interaction > 4A is zeroed 
    return:
        mapped_icoxx_data: updated (ase_info_list, icoxx)
        missing_interactions: updated (ase_info_list)
    """

    for ase_site in missing_interactions[:]:
        if ase_site[1] > cutoff:
            all_info = (*ase_site, 0)
            if all_info not in mapped_icoxx_data:
                mapped_icoxx_data.append((*ase_site, 0))
                missing_interactions.remove(ase_site)
    return mapped_icoxx_data, missing_interactions


def icoxx_struc_fxn(CONTCAR_path, r_cutoff, ICOXX_path, test_bond_len=False,):
    """ 
    function that combines all the above functions to map ICOXX values into ase_atoms
    Args:
        CONTCAR_path: path to CONTCAR
        r_cutoff: cutoff radius
        ICOXX_path: path to ICOXX file
        test_bond_len: whether to map ase and icoxx interactions using bond length as criteria along w/ pairs and images
    return:
        mapped_icoxx_data: (ase_info_list, icoxx)
        missing_interactions: (ase_info_list)
    """

    icoxx_list_pairs_l_S, icoxx_dict_pairsS_l = get_icoxx_pairs_S_l(ICOXX_file_path=ICOXX_path)
    ase_struc_list = get_struc_pairs_S_l(CONTCAR_path=CONTCAR_path, r_max=r_cutoff)
    mapped_icoxx_data, missing_interactions = compare_pairs(
                                                    struc_list_pairs_S_l=ase_struc_list, 
                                                    icoxx_dict_pairsS_l=icoxx_dict_pairsS_l, 
                                                    test_bond_len=test_bond_len,
                                                    )
    mapped_icoxx_data_reverse, missing_interactions_reverse = reverse_missed_inter(
                                                                    mapped_icoxx_data=mapped_icoxx_data, 
                                                                    missing_interactions=missing_interactions,
                                                                    icoxx_dict_pairsS_l=icoxx_dict_pairsS_l,
                                                                    test_bond_len=test_bond_len,
                                                                    )
    #if missing_check:
    #    mapped_icoxx_data_reverse, missing_interactions_reverse 

    return mapped_icoxx_data_reverse, missing_interactions_reverse
    

if __name__ == "__main__":
    r_max=5; print(r_max, 'Angstroms cutoff R')
    mapped_icoxx_data_reverse, missing_interactions_reverse=icoxx_struc_fxn(CONTCAR_path='CONTCAR', 
                                                                            r_cutoff=5, 
                                                                            ICOXX_path='ICOHPLIST.lobster', 
                                                                            test_bond_len=False, missing_check=False, 
                                                                            missing_cutoff=4, missing_icoxx=0)
    #edge_src, edge_dst, edge_shift, edge_len = neighbor_list("ijSd", a=read('CONTCAR'), cutoff=r_max, self_interaction=False)
    #print(edge_shift)
    exit()
    icoxx_list_pairs_l_S, icoxx_dict_pairsS_l = get_icoxx_pairs_S_l(ICOXX_file_path='ICOHPLIST.lobster')
    ase_struc_list = get_struc_pairs_S_l(CONTCAR_path='CONTCAR', r_max=r_max)


    mapped_icoxx_data_w_length, missing_interactions_w_length = compare_pairs(
                                                    struc_list_pairs_S_l=ase_struc_list, 
                                                    icoxx_dict_pairsS_l=icoxx_dict_pairsS_l, 
                                                    test_bond_len=True,
                                                    )
    print('mapped and missed icoxx w bond length', len(mapped_icoxx_data_w_length), len(missing_interactions_w_length))
    mapped_icoxx_data_no_length, missing_interactions_no_length = compare_pairs(
                                                    struc_list_pairs_S_l=ase_struc_list, 
                                                    icoxx_dict_pairsS_l=icoxx_dict_pairsS_l, 
                                                    test_bond_len=False,
                                                    )
    print('mapped and missed icoxx wout bond length', len(mapped_icoxx_data_no_length), len(missing_interactions_no_length))


    #exit()
    mapped_icoxx_data_w_length_updated, missing_interactions_w_length_updated = reverse_missed_inter(
                                                                        mapped_icoxx_data=mapped_icoxx_data_w_length, 
                                                                        missing_interactions=missing_interactions_w_length,
                                                                        icoxx_dict_pairsS_l=icoxx_dict_pairsS_l,
                                                                        test_bond_len=True,
                                                                        )
    print('mapped and missed icoxx w bond length post-reverse', len(mapped_icoxx_data_w_length_updated), len(missing_interactions_w_length_updated))
    mapped_icoxx_data_no_length_updated, missing_interactions_no_length_updated = reverse_missed_inter(
                                                                        mapped_icoxx_data=mapped_icoxx_data_no_length, 
                                                                        missing_interactions=missing_interactions_no_length,
                                                                        icoxx_dict_pairsS_l=icoxx_dict_pairsS_l,
                                                                        test_bond_len=False,
                                                                        )
    print('mapped and missed icoxx wout bond length post-reverse', len(mapped_icoxx_data_no_length_updated), len(missing_interactions_no_length_updated))



    mapped_icoxx_data_w_length_0_far, missing_interactions_w_length_0_far=zero_missing_icoxx_cutoff(
                                                                        mapped_icoxx_data=mapped_icoxx_data_w_length_updated, 
                                                                        missing_interactions=missing_interactions_w_length_updated, 
                                                                        cutoff=4)
    mapped_icoxx_data_no_length_0_far, missing_interactions_no_length_0_far=zero_missing_icoxx_cutoff(
                                                                        mapped_icoxx_data=mapped_icoxx_data_no_length_updated, 
                                                                        missing_interactions=missing_interactions_no_length_updated, 
                                                                        cutoff=4)

    print('mapped and missed icoxx w bond length post-reverse', len(mapped_icoxx_data_w_length_0_far), len(missing_interactions_w_length_0_far))    
    print('mapped and missed icoxx wout bond length post-reverse', len(mapped_icoxx_data_no_length_0_far), len(missing_interactions_no_length_0_far))
    #print(mapped_icoxx_data_no_length_0_far[-6:])


