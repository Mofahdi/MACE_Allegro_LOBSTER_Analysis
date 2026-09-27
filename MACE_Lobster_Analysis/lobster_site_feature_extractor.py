from typing import List, Tuple, Union
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.lobster import Charge
from lobsterpy.featurize.core import FeaturizeIcoxxlist

def extract_lobster_site_features(
    structure_path: str,
    icohp_path: str,
    icobi_path: str,
    chrg_path: str,
    output_feat_labels: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    Extract site-level LOBSTER features for all atoms in a crystal structure.

    This function computes local ICOHP and ICOBI summary statistics for each site,
    including binned distribution metrics, asymmetry indices, and Mulliken/Loewdin
    charges from the LOBSTER charge file. It returns the feature matrix for all
    sites, optionally together with the feature names.

    Parameters
    ----------
    structure_path : str
        Path to the structure file (for example, a CONTCAR or POSCAR).
    icohp_path : str
        Path to the LOBSTER ICOHP file.
    icobi_path : str
        Path to the LOBSTER ICOBI file.
    chrg_path : str
        Path to the LOBSTER charge file.
    output_feat_labels : bool, optional
        If True, also return the feature names corresponding to each column in
        the feature matrix. Default is False.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, list[str]]
        A 2D array of shape (n_sites, n_features). If ``output_feat_labels`` is
        True, returns a tuple containing the feature matrix and the feature names.

    Notes
    -----
    The feature set includes:
    - ICOHP and ICOBI binned-distribution statistics
    - asymmetry indices for each site
    - Mulliken and Loewdin charge values
    """
    structure = Structure.from_file(structure_path)
    site_features: list[list[float]] = []

    icohp_featurizer = FeaturizeIcoxxlist(
        path_to_icoxxlist=icohp_path,
        path_to_structure=structure_path,
        bin_width=0.1,
        normalization="counts",
        max_length=5,
    )
    icobi_featurizer = FeaturizeIcoxxlist(
        path_to_icoxxlist=icobi_path,
        path_to_structure=structure_path,
        bin_width=0.1,
        normalization="counts",
        max_length=5,
    )
    charge_obj = Charge(filename=chrg_path)

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

    for site_index in range(len(structure)):

        site_icohps = [bond[3] for bond in icohp_featurizer.get_icoxx_neighbors_data(site_index = site_index)['mapped_icoxx_data']]
        site_icobis = [bond[3] for bond in icobi_featurizer.get_icoxx_neighbors_data(site_index = site_index)['mapped_icoxx_data']]

        site_feat_dict = {
            f"site_icohp_{name}": fn(site_icohps)
            for name, fn in stats_fns_icohp.items()
        }
        site_feat_dict.update({
            f"site_icobi_{name}": fn(site_icobis)
            for name, fn in stats_fns_icobi.items()
        })

        site_feat_dict["charge_mull"] = charge_obj.mulliken[site_index]
        site_feat_dict["charge_loew"] = charge_obj.loewdin[site_index]

        site_features.append(list(site_feat_dict.values()))

    all_site_features = np.asarray(site_features, dtype=np.float32)

    if output_feat_labels:
        feature_labels = list(site_feat_dict.keys())
        return all_site_features, feature_labels

    return all_site_features

if __name__ == '__main__':
    test_func, labels = extract_lobster_site_features(structure_path = 'CONTCAR.gz',
        icohp_path = 'ICOHPLIST.lobster.gz',
        icobi_path = 'ICOBILIST.lobster.gz',
        chrg_path = 'CHARGE.lobster.gz',
        output_feat_labels = True,
    )
    print(type(test_func), test_func.shape, labels)

