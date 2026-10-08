from __future__ import annotations

import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np 
import pandas as pd
from dgl.data.utils import Subset
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "test")



def structure_split_indices(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
    seed: int = 42,
):

    #random.Random(seed).shuffle(df)
    #df.sort(key=len, reverse=True)

    target_sizes = {
        "train": len(df) * train_fraction,
        "test": len(df) * (1.0 - train_fraction),
    }

    split_indices = {
        "train": [],
        "test": [],
    }

    for group in df:
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
    df = pd.read_csv('mp_data_paths_test.csv')
    #print(df.shape)

    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42
    )

    print(df.shape, df.loc[train_idx, :].shape)
    #print(split_indices.shape)
    exit()
    print("Train:", len(split_indices["train"]))
    print("Test :", len(split_indices["test"]))

    #print(split_indices)

    #train_df = df.iloc[split_indices["train"]]
    #test_df = df.iloc[split_indices["test"]]

    #print(train_df.shape, test_df.shape)


    #from structure_split import load_or_create_split

    # structures, grouped_indices = load_structures(
    #     df,
    #     symprec=1e-4,
    #     angle_tolerance=1.0,
    # )
    # #print(len(grouped_indices.values()))
    # #print(len(list(set(grouped_indices.keys()))))

    # #df['']
    # #print(grouped_indices.values())

    # grouped_indices_list = list(map(lambda x: x[0], grouped_indices.values()))
    # new_df = df.loc[grouped_indices_list, ['id' , 'formula_pretty', 'chg_spill_1','chg_spill_2','mat_path']]
    # print(new_df.shape)

    # new_df.to_csv('mp_data_paths.csv', index=False)

#    exit()

    # write_split_csv(
    #     df,
    #     split_indices,
    #     output_path="train_test_split.csv",
    # )


