import os
from tqdm import tqdm
import numpy as np
import torch

from pymatgen.core import Structure
import pandas as pd
import numpy as np
import warnings
from pymatgen.io.lobster import Charge
#from lobsterpy.featurize.batch import BatchSummaryFeaturizer, BatchIcoxxlistFeaturizer
from lobsterpy.featurize.core import FeaturizeIcoxxlist

import warnings
warnings.filterwarnings("ignore")

default_dtype = torch.float32
torch.set_default_dtype(default_dtype)


# ------------------------------------------------------------
# Featurizer setup
# ------------------------------------------------------------
def get_lobster_site_feat(structure_path, ICOHP_path, ICOBI_path, chrg_path, output_feat_labels: bool = False):
    """
    Get featurized dataframe using path provided containing LOBSTER calc files.

    Uses LobsterPy featurizer implementations.

    Args:
        path_to_lobster_calc: Path to directory containing LOBSTER calcs output files
        site_index: site index of structure

    Returns:
        A pandas DataFrame with site features extracted from LOBSTER calculation files
    """

    struc=Structure.from_file(structure_path)
    sites_feats=[]
    for site_index in range(len(struc)):
        icohp_featurizer = FeaturizeIcoxxlist(
        path_to_icoxxlist=ICOHP_path,
        path_to_structure=structure_path,
        bin_width=0.1,
        normalization="counts",
        max_length=5,
        )
        icobi_featurizer = FeaturizeIcoxxlist(
        path_to_icoxxlist=ICOBI_path,
        path_to_structure=structure_path,
        bin_width=0.1,
        normalization="counts",
        max_length=5,
        )
        
        site_icohp_bwdf = icohp_featurizer.calc_site_bwdf(site_index=site_index)
        bwdf_icohp_values = site_icohp_bwdf[f"{site_index}"]["icoxx_binned"]
        stats_fns = {
            "max": np.max,
            "mean": np.mean,
            "std": np.std,
            "min": np.min,
            "sum": np.sum,
        }
        site_feats = {
            f"site_icohp_bwdf_{name}": fn(bwdf_icohp_values) for name, fn in stats_fns.items()
        }
        site_feats.update(
            {
            "site_icohp_asi": icohp_featurizer.calc_site_asymmetry_index(site_index),
            }
        )
        
        site_icobi_bwdf = icobi_featurizer.calc_site_bwdf(site_index=site_index)
        bwdf_icobi_values = site_icobi_bwdf[f"{site_index}"]["icoxx_binned"]
        site_feats.update(
            {f"site_icobi_bwdf_{name}": fn(bwdf_icobi_values) for name, fn in stats_fns.items()
             }
        )
        site_feats.update(
            {
            "site_icobi_asi": icobi_featurizer.calc_site_asymmetry_index(site_index),
            }
        )

        chargeobj = Charge(filename=chrg_path)
        site_feats.update(
            {
            "charge_mull": chargeobj.mulliken[site_index],
            "charge_loew": chargeobj.loewdin[site_index],
            }
        )

        sites_feats.append(list(site_feats.values()))

    #all_sites_feats=torch.tensor(sites_feats, dtype=default_dtype); 
    all_sites_feats = np.array(sites_feats, dtype=np.float32)
    if output_feat_labels:
        siteFeats_labels = list(site_feats.keys())
        return all_sites_feats, siteFeats_labels
    return all_sites_feats


if __name__=='__main__':
    import pandas as pd
    df=pd.read_csv('5_lo_chg_mpInfo_nonmetals_test4.csv')
    df['structure']=df['struc_dir'].apply(lambda x: Structure.from_file(x, sort = False))

    struc=df.loc[0, 'structure']
    struc_dir=df.loc[0, 'struc_dir']
    ICOHP_dir=df.loc[0, 'ICOHP_dir'] # ICOHP_dir,ICOBI_dir
    ICOBI_dir=df.loc[0, 'ICOBI_dir'] # ICOHP_dir,ICOBI_dir
    chrg_dir=df.loc[0, 'chrg_dir'] # ICOHP_dir,ICOBI_dir

    lobster_n_feats=get_lobster_site_feat(structure_path=struc_dir, ICOXX_path=ICOHP_dir, chrg_path=chrg_dir); #exit()
    print(lobster_n_feats.shape, )
