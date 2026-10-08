from __future__ import annotations

import random
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dgl.data.utils import Subset
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


SPLIT_NAMES = ("train", "test")


def load_structures(
    df: pd.DataFrame,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
):
    structures = []
    grouped_indices = defaultdict(list)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for index, structure_path in enumerate(df["struc_dir"]):
            structure = Structure.from_file(str(structure_path))

            analyzer = SpacegroupAnalyzer(
                structure,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )

            reduced_formula = structure.composition.reduced_formula
            space_group_number = analyzer.get_space_group_number()

            key = (reduced_formula, space_group_number)

            structures.append(structure)
            grouped_indices[key].append(index)

    return structures, grouped_indices

def structure_groups(
    df: pd.DataFrame,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
):
    structures, grouped_indices = load_structures(
        df,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    #print(len(grouped_indices.values()))
    #print(len(list(set(grouped_indices.keys()))))

    #df['']
    #print(grouped_indices.values())

    grouped_indices_list = list(map(lambda x: x[0], grouped_indices.values()))
    new_df = df.iloc[grouped_indices_list]
    print(new_df.shape)
    exit()
    matcher = StructureMatcher(
        ltol=0.3,
        stol=0.5,
        angle_tol=angle_tolerance,
        primitive_cell=True,
        scale=True, # True False
        attempt_supercell=False,
        allow_subset=False,
    )

    groups = []

    for indices in grouped_indices.values():
        parents = {index: index for index in indices}

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left, right):
            left_root = find(left)
            right_root = find(right)

            if left_root != right_root:
                parents[right_root] = left_root

        for position, left in enumerate(indices):
            for right in indices[position + 1:]:
                if matcher.fit(
                    structures[left],
                    structures[right],
                ):
                    union(left, right)

        matched_groups = defaultdict(list)

        for index in indices:
            matched_groups[find(index)].append(index)

        groups.extend(matched_groups.values())

    return groups

def structure_split_indices(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
    seed: int = 42,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
):
    groups = structure_groups(
        df,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)

    target_sizes = {
        "train": len(df) * train_fraction,
        "test": len(df) * (1.0 - train_fraction),
    }

    split_indices = {
        "train": [],
        "test": [],
    }

    for group in groups:
        split_name = min(
            SPLIT_NAMES,
            key=lambda name: (
                len(split_indices[name]) / target_sizes[name]
            ),
        )

        split_indices[split_name].extend(group)

    return {
        "train": sorted(split_indices["train"]),
        "test": sorted(split_indices["test"]),
    }

def write_split_csv(
    df: pd.DataFrame,
    split_indices,
    output_path: str | Path,
):
    rows = []

    for split_name in SPLIT_NAMES:
        for index in split_indices[split_name]:
            rows.append(
                {
                    "id": str(df.iloc[index]["id"]),
                    "split": split_name,
                    "index": index,
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def read_split_csv(
    df: pd.DataFrame,
    split_path,
    ):

    split_df = pd.read_csv(split_path)

    id_to_index = {
        str(material_id): index
        for index, material_id in enumerate(df["id"])
    }

    split_indices = {
        "train": [],
        "test": [],
    }

    for _, row in split_df.iterrows():
        material_id = str(row["id"])
        split_name = str(row["split"])

        split_indices[split_name].append(
            id_to_index[material_id]
        )

    return {
        "train": sorted(split_indices["train"]),
        "test": sorted(split_indices["test"]),
    }


def load_or_create_split(
    df: pd.DataFrame,
    split_path: str | Path,
    train_fraction: float = 0.8,
    seed: int = 42,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
):
    split_path = Path(split_path)

    if split_path.is_file():
        return read_split_csv(df, split_path)

    split_indices = structure_split_indices(
        df,
        train_fraction=train_fraction,
        seed=seed,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    # write_split_csv(
    #     df,
    #     split_indices,
    #     split_path,
    # )

    return split_indices


if __name__ == '__main__':
    df = pd.read_csv('nonmetals_mp_info.csv')

    #from structure_split import load_or_create_split

    structures, grouped_indices = load_structures(
        df,
        symprec=1e-4,
        angle_tolerance=1.0,
    )
    #print(len(grouped_indices.values()))
    #print(len(list(set(grouped_indices.keys()))))

    #df['']
    #print(grouped_indices.values())

    grouped_indices_list = list(map(lambda x: x[0], grouped_indices.values()))
    new_df = df.loc[grouped_indices_list, ['id' , 'formula_pretty', 'chg_spill_1','chg_spill_2','mat_path']]
    print(new_df.shape)

    new_df.to_csv('mp_data_paths.csv', index=False)

    exit()
    split_indices = load_or_create_split(
        df,
        split_path="train_test_split.csv",
        train_fraction=0.8,
        seed=42,
        symprec=1e-4, #1e-3
        angle_tolerance=1.0,
    )

    print("Train:", len(split_indices["train"]))
    print("Test :", len(split_indices["test"]))

    train_df = df.iloc[split_indices["train"]]
    test_df = df.iloc[split_indices["test"]]

    print(train_df.shape, test_df.shape)

    # write_split_csv(
    #     df,
    #     split_indices,
    #     output_path="train_test_split.csv",
    # )


