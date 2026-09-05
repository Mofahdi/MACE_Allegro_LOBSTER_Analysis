import os
from tqdm import tqdm
import numpy as np
import torch

from pymatgen.core import Structure
from matminer.featurizers.base import MultipleFeaturizer
import pandas as pd
import numpy as np
import warnings
from monty.serialization import MontyDecoder
from matminer.featurizers.site import (
    AGNIFingerprints,
    AverageBondAngle,
    AverageBondLength,
    BondOrientationalParameter,
    ChemEnvSiteFingerprint,
    CoordinationNumber,
    CrystalNNFingerprint,
    GaussianSymmFunc,
    ChemicalSRO,
    LocalPropertyDifference,
    VoronoiFingerprint,
    SiteElementalProperty,
    
    GeneralizedRadialDistributionFunction,
    OPSiteFingerprint,
)

from matminer.featurizers.site.external import SOAP
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure import ElectronicRadialDistributionFunction

from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.lobster import Charge
#from lobsterpy.featurize.batch import BatchSummaryFeaturizer, BatchIcoxxlistFeaturizer
from lobsterpy.featurize.core import FeaturizeIcoxxlist
#from lobsterpy.featurize.utils import get_file_paths

import warnings
warnings.filterwarnings("ignore")

default_dtype = torch.float32
torch.set_default_dtype(default_dtype)


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
        # site_feats.update(
        #     {
        #     "site_icohp_asi": icohp_featurizer.calc_site_asymmetry_index(site_index),
        #     }
        # )

        site_icobi_bwdf = icobi_featurizer.calc_site_bwdf(site_index=site_index)
        bwdf_icobi_values = site_icobi_bwdf[f"{site_index}"]["icoxx_binned"]
        site_feats.update(
            {f"site_icobi_bwdf_{name}": fn(bwdf_icobi_values) for name, fn in stats_fns.items()
             }
        )
        # site_feats.update(
        #     {
        #     "site_icobi_asi": icobi_featurizer.calc_site_asymmetry_index(site_index),
        #     }
        # )

        chargeobj = Charge(filename=chrg_path)
        site_feats.update(
            {
            "charge_loew": chargeobj.loewdin[site_index],
            #"charge_mull": chargeobj.mulliken[site_index],
            }
        )

        sites_feats.append(list(site_feats.values()))
        
    all_sites_feats=torch.tensor(sites_feats, dtype=default_dtype); 
    if output_feat_labels:
        siteFeats_labels = list(site_feats.keys())
        return all_sites_feats, siteFeats_labels
    return all_sites_feats

# ------------------------------------------------------------
# Featurizer setup
# ------------------------------------------------------------
def get_matminer_site_feats(
    structure_path: str, site_featurizers: list | None = None, #n_jobs: int = 8
) -> list:
    """
    Get site featurized dataframe using pymatgen structure object .

    Uses Matminers MultipleFeaturizer

    Args:
        structure: pymatgen structure
        site_featurizers: list of matminer site based featurizes to apply on input structures_df
        n_jobs: number of parallel jobs to run for featurization

    Returns:
        A pandas dataframe with structure and composition based features for sites
    """
    #print(structure_path)
    #structures_df['structure']=structures_df['struc_dir'].apply(lambda x: Structure.from_file(x, sort = False))
    structure=Structure.from_file(structure_path, sort = False)
    if site_featurizers is None:
        site_featurizers = [
            AGNIFingerprints(cutoff=5), # good
            AverageBondAngle(VoronoiNN()), # good
            AverageBondLength(VoronoiNN()), # good
            #BondOrientationalParameter(max_l=5), # good
            CoordinationNumber(), # good
            CrystalNNFingerprint.from_preset("cn"), # CN feats, prob. good, prob better for our use
            #GaussianSymmFunc(), # good   
            LocalPropertyDifference(properties=("Electronegativity", "CovalentRadius")), #"Number", 
            #VoronoiFingerprint(), # good
            SiteElementalProperty(properties=("Electronegativity", "CovalentRadius", 'ElectronAffinity', 'AllenElectronegativity')), #"Number", 
            #SOAP.from_preset("formation_energy"),
            
            #ChemicalSRO.from_preset('VoronoiNN'),
            #ChemEnvSiteFingerprint.from_preset("simple"), # one-hot motifs, good vector, prob. bad for our use
            #GeneralizedRadialDistributionFunction.from_preset("gaussian"), # same as above
            #CrystalNNFingerprint.from_preset("ops"), # CN feats, prob. good, prob bad for our use
            #OPSiteFingerprint(), # CN feats prob. not ok
        ]

    multi_feat = MultipleFeaturizer(featurizers=site_featurizers)

    site_features = [
    multi_feat.featurize(structure, i)
    for i in range(len(structure))
    ]
    node_feats=torch.tensor(site_features)
    #print(node_feats.dtype)
    return node_feats

'''def get_soap_feats(structures_df):
    structures_df['structure']=structures_df['struc_dir'].apply(lambda x: Structure.from_file(x, sort = False))
    #soap = SOAP.from_preset("formation_energy")
    soap = SOAP(rcut=5, nmax=6, lmax=4, sigma=0.4, periodic=True, rbf="gto", crossover=True, compression=False)
    # df["structure"] contains pymatgen Structure objects
    soap.fit(df["structure"])
    features = soap.featurize(df["structure"].iloc[0], 1)
    #print(len(features), max(features))
    return features
'''

if __name__=='__main__':
    import pandas as pd
    df=pd.read_csv('5_lo_chg_mpInfo_nonmetals_test4.csv')
    df['structure']=df['struc_dir'].apply(lambda x: Structure.from_file(x, sort = False))

    struc=df.loc[0, 'structure']
    struc_dir=df.loc[0, 'struc_dir']
    ICOHP_dir=df.loc[0, 'ICOHP_dir'] # ICOHP_dir,ICOBI_dir
    ICOBI_dir=df.loc[0, 'ICOBI_dir'] # ICOHP_dir,ICOBI_dir
    chrg_dir=df.loc[0, 'chrg_dir'] # ICOHP_dir,ICOBI_dir

    matminer_n_feats=get_matminer_site_feats(struc_dir); #exit()
    lobster_n_feats=get_lobster_site_feat(structure_path=struc_dir, ICOXX_path=ICOHP_dir, chrg_path=chrg_dir); #exit()
    matminer_lobster_n_feats = torch.cat([matminer_n_feats, lobster_n_feats], dim=-1)
    print(matminer_n_feats.shape, lobster_n_feats.shape, matminer_lobster_n_feats.shape)
    #get_soap_feats(structures_df=df)
    #exit()
    #feats.to_csv('feats.csv', index=False)