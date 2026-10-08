from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


def _load_structures_by_formula(df: pd.DataFrame):
    structures = []
    by_formula = defaultdict(list)

    for idx, path in enumerate(df["struc_dir"]):
        structure = Structure.from_file(path)
        structures.append(structure)
        by_formula[structure.composition.reduced_formula].append(idx)

    return structures, by_formula


def structure_groups(df: pd.DataFrame):
    """
    Group structures that are equivalent according to StructureMatcher.
    Structures in the same group will always stay together.
    """

    structures, by_formula = _load_structures_by_formula(df)

    matcher = StructureMatcher(
        primitive_cell=True,
        scale=True,
    )

    groups = []

    for indices in by_formula.values():

        parents = {i: i for i in indices}

        def find(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        def union(i, j):
            ri = find(i)
            rj = find(j)
            if ri != rj:
                parents[rj] = ri

        for pos, left in enumerate(indices):
            for right in indices[pos + 1:]:
                if matcher.fit(structures[left], structures[right]):
                    union(left, right)

        formula_groups = defaultdict(list)
        for idx in indices:
            formula_groups[find(idx)].append(idx)

        groups.extend(formula_groups.values())

    return groups


def train_test_split_indices(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    seed: int = 42,
):
    """
    Structure-aware split.

    Equivalent structures are kept together and assigned
    either to train or test.
    """

    groups = structure_groups(df)

    rng = random.Random(seed)
    rng.shuffle(groups)

    total_size = len(df)

    target_train = round(total_size * train_frac)

    splits = {
        "train": [],
        "test": [],
    }

    for group in sorted(groups, key=len, reverse=True):

        if len(splits["train"]) < target_train:
            splits["train"].extend(group)
        else:
            splits["test"].extend(group)

    return {
        "train": sorted(splits["train"]),
        "test": sorted(splits["test"]),
    }


def write_split_csv(df, split_indices, output_file="splits.csv"):

    rows = []

    for split_name in ("train", "test"):
        for idx in split_indices[split_name]:
            rows.append(
                {
                    "id": df.iloc[idx]["id"],
                    "split": split_name,
                    "index": idx,
                }
            )

    pd.DataFrame(rows).to_csv(output_file, index=False)
