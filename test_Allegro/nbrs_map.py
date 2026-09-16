from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from pymatgen.io.lobster.outputs import Icohplist

BondEntry = tuple[list[int], float, list[int]]
MappedBond = tuple[list[int], float, list[int], float]


def get_icoxx_pairs_S_l(
    ICOXX_file_path: str,
) -> tuple[list[BondEntry], dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, float]]]:
    """Parse an ICOXX Lobster file into pair, bond-length, and image metadata.

    Args:
        ICOXX_file_path: Path to the ICOXX Lobster file.

    Returns:
        A pair containing the raw ICOXX bond metadata and a dictionary keyed by
        ``(pair, translation)`` that maps to ``(bond_length, icoxx_value)``.
    """
    icohp_list = Icohplist(filename=ICOXX_file_path)
    list_atom1 = list(icohp_list._icohpcollection._list_atom1)
    list_atom2 = list(icohp_list._icohpcollection._list_atom2)
    list_lens = list(icohp_list._icohpcollection._list_length)
    list_trans = [list(trans) for trans in icohp_list._icohpcollection._list_translation]
    list_icohp = [float(sum(cohp_dict.values())) for cohp_dict in icohp_list._icohpcollection._list_icohp]

    list_atom1_src_index_lobster = [int("".join(filter(str.isdigit, at1))) - 1 for at1 in list_atom1]
    list_atom2_dst_index_lobster = [int("".join(filter(str.isdigit, at2))) - 1 for at2 in list_atom2]

    arr_lens = np.asarray(list_lens)
    pairs_lobster = [[at1, at2] for at1, at2 in zip(list_atom1_src_index_lobster, list_atom2_dst_index_lobster)]
    icoxx_info_list: list[BondEntry] = list(zip(pairs_lobster, arr_lens, list_trans))
    icoxx_info_dict: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, float]] = {
        (tuple(icoxx_info[0]), tuple(icoxx_info[2])): (icoxx_info[1], list_icohp[idx])
        for idx, icoxx_info in enumerate(icoxx_info_list)
    }

    return icoxx_info_list, icoxx_info_dict


def get_struc_pairs_S_l(CONTCAR_path: str, r_max: float) -> list[BondEntry]:
    """Get bond pairs, bond lengths, and cell translations from an ASE structure.

    Args:
        CONTCAR_path: Path to the structure file.
        r_max: Maximum cutoff distance used for neighbour detection.

    Returns:
        A list of tuples of the form ``(pair, bond_length, cell_shift)``.
    """
    struc: Atoms = read(CONTCAR_path)
    edge_src, edge_dst, edge_shift, edge_len = neighbor_list(
        "ijSd",
        a=struc,
        cutoff=r_max,
        self_interaction=False,
    )
    pairs_ase = [[int(at1), int(at2)] for at1, at2 in zip(edge_src, edge_dst)]
    ase_struc_list: list[BondEntry] = list(zip(pairs_ase, edge_len, edge_shift.tolist()))

    return ase_struc_list



def compare_pairs(
    struc_list_pairs_S_l: Sequence[BondEntry],
    icoxx_dict_pairsS_l: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, float]],
    test_bond_len: bool = False,
) -> tuple[list[MappedBond], list[BondEntry]]:
    """Match ASE neighbor pairs to ICOXX values using pair and image keys.

    Args:
        struc_list_pairs_S_l: Structure bond entries of the form
            ``(pair, bond_length, periodic_translation)``.
        icoxx_dict_pairsS_l: Mapping from ``(pair, translation)`` to
            ``(bond_length, icoxx_value)``.
        test_bond_len: If True, also require matching bond lengths within a small
            tolerance.

    Returns:
        A tuple containing the mapped ICOXX entries and the remaining missing
        interactions.
    """
    mapped_icoxx_data: list[MappedBond] = []
    missing_interactions: list[BondEntry] = []
    for ase_site in struc_list_pairs_S_l:
        comp_pair = (tuple(ase_site[0]), tuple(ase_site[2]))
        if comp_pair in icoxx_dict_pairsS_l:
            icoxx_bond_length, icoxx_value = icoxx_dict_pairsS_l[comp_pair]
            if test_bond_len:
                if round(np.absolute(ase_site[1] - icoxx_bond_length), 5) <= 1e-3:
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


def reverse_missed_inter(
    mapped_icoxx_data: list[MappedBond],
    missing_interactions: list[BondEntry],
    icoxx_dict_pairsS_l: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, float]],
    test_bond_len: bool = False,
) -> tuple[list[MappedBond], list[BondEntry]]:
    """Assign ICOXX values to reversed atom pairs when they match the same bond.

    Args:
        mapped_icoxx_data: Already-mapped ICOXX results for forward interactions.
        missing_interactions: Bonds that did not match in the initial pass.
        icoxx_dict_pairsS_l: Dictionary mapping pair and image keys to ICOXX data.
        test_bond_len: If True, require bond-length agreement before accepting the
            reversed match.

    Returns:
        Updated mapped data and the remaining unmatched interactions.
    """
    for ase_site in missing_interactions[:]:
        reversed_translation = tuple(map(int, -1 * np.array(ase_site[2])))
        reversed_pair = tuple(reversed(ase_site[0]))
        reversed_key = (reversed_pair, reversed_translation)
        if reversed_key in icoxx_dict_pairsS_l:
            icoxx_bond_length, icoxx_value = icoxx_dict_pairsS_l[reversed_key]
            if test_bond_len:
                if round(np.absolute(ase_site[1] - icoxx_bond_length), 5) <= 1e-3:
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


# def zero_missing_icoxx_cutoff(
#     mapped_icoxx_data: list[MappedBond],
#     missing_interactions: list[BondEntry],
#     cutoff: float = 4.0,
# ) -> tuple[list[MappedBond], list[BondEntry]]:
#     """Set distant missing interactions to zero beyond a given distance.

#     Args:
#         mapped_icoxx_data: Current mapped ICOXX entries.
#         missing_interactions: Remaining unmatched structure interactions.
#         cutoff: Distance cutoff in Angstroms above which missing data is zeroed.

#     Returns:
#         Updated mapped entries and the remaining missing interactions.
#     """
#     for ase_site in missing_interactions[:]:
#         if ase_site[1] > cutoff:
#             if (*ase_site, 0) not in mapped_icoxx_data:
#                 mapped_icoxx_data.append((*ase_site, 0))
#                 missing_interactions.remove(ase_site)
#     return mapped_icoxx_data, missing_interactions


def icoxx_struc_fxn(
    CONTCAR_path: str,
    r_cutoff: float,
    ICOXX_path: str,
    test_bond_len: bool = False,
    #missing_check: bool = False,
    #missing_cutoff: float = 4.0,
    #missing_icoxx: float = 0.0,
) -> tuple[list[MappedBond], list[BondEntry]]:
    """Map ICOXX values from a Lobster file onto an ASE structure.

    Args:
        CONTCAR_path: Path to the structure file.
        r_cutoff: Neighbor cutoff used for structure bond detection.
        ICOXX_path: Path to the ICOXX Lobster file.
        test_bond_len: If True, apply a bond-length check during matching.
        missing_check: If True, assign zeros to distant missing bonds.
        missing_cutoff: Distance cutoff used when zeroing missing interactions.
        missing_icoxx: Value assigned to missing interactions beyond the cutoff.

    Returns:
        A tuple containing mapped ICOXX entries and the list of unmatched bonds.
    """
    _, icoxx_dict_pairsS_l = get_icoxx_pairs_S_l(ICOXX_file_path=ICOXX_path)
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

    # if missing_check:
    #     mapped_icoxx_data_reverse, missing_interactions_reverse = zero_missing_icoxx_cutoff(
    #         mapped_icoxx_data=mapped_icoxx_data_reverse,
    #         missing_interactions=missing_interactions_reverse,
    #         cutoff=missing_cutoff,
    #     )

    return mapped_icoxx_data_reverse, missing_interactions_reverse
    
