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
from matminer.featurizers.structure import SiteStatsFingerprint
from matminer.featurizers.structure import (
    CoulombMatrix,
    OrbitalFieldMatrix,
    SineCoulombMatrix,
    ElectronicRadialDistributionFunction,
)
#from matminer.featurizers.site.external import SOAP
from matminer.featurizers.base import MultipleFeaturizer

from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.lobster import Charge
#from lobsterpy.featurize.batch import BatchSummaryFeaturizer, BatchIcoxxlistFeaturizer
from lobsterpy.featurize.core import FeaturizeIcoxxlist, FeaturizeCharges
#from lobsterpy.featurize.utils import get_file_paths

import warnings
warnings.filterwarnings("ignore")

default_dtype = torch.float32
torch.set_default_dtype(default_dtype)


def get_lobster_struc_feat(structure_path, ICOHP_path, ICOBI_path, chrg_path, output_feat_labels: bool = False):
    """
    Get featurized dataframe using path provided containing LOBSTER calc files.

    Uses LobsterPy featurizer implementations.

    Args:
        path_to_lobster_calc: Path to directory containing LOBSTER calcs output files
        site_index: site index of structure

    Returns:
        A pandas DataFrame with site features extracted from LOBSTER calculation files
    """

    #struc=Structure.from_file(structure_path)
    icohp_featurizer = FeaturizeIcoxxlist(
        path_to_icoxxlist=ICOHP_path,
        path_to_structure=structure_path,
        bin_width=0.1,
        normalization="counts",
        max_length=5,
        )
    icohp_asym=icohp_featurizer.get_asymmetry_index_stats_df()
    icohp_featurizer.get_binned_bwdf_df()
    icohp_stats=icohp_featurizer.get_stats_df()
    #print(icohp_stats.columns.tolist())

    icobi_featurizer = FeaturizeIcoxxlist(
        path_to_icoxxlist=ICOBI_path,
        path_to_structure=structure_path,
        bin_width=0.1,
        normalization="counts",
        max_length=5,
        )
    icobi_asym=icobi_featurizer.get_asymmetry_index_stats_df()
    icobi_featurizer.get_binned_bwdf_df()
    icobi_stats=icobi_featurizer.get_stats_df()

    loew_chg=FeaturizeCharges(path_to_structure=structure_path, path_to_charge=chrg_path, charge_type='loewdin').get_df() # 'mulliken' 'loewdin'
    mulk_chg=FeaturizeCharges(path_to_structure=structure_path, path_to_charge=chrg_path, charge_type='mulliken').get_df() # 'mulliken' 'loewdin'

    all_feat_labels = ['icohp_' + x for x in icohp_stats.columns.tolist()]
    all_feat_labels.extend(['icobi_' + x for x in icobi_stats.columns.tolist()])
    all_feat_labels.extend(loew_chg.columns.tolist()); all_feat_labels.extend(mulk_chg.columns.tolist())

    all_feats = np.concatenate([icohp_stats.values, icobi_stats.values, loew_chg.values, mulk_chg.values], axis=-1).astype(np.float32) # 
    #print(all_feats); exit()
    #all_feats = torch.from_numpy(all_feats).float()
    if output_feat_labels:
        return all_feats, all_feat_labels
    return all_feats

    
def get_matminer_struc_feats(
    structure_path: str, site_featurizers: list | None = None, output_feat_labels: bool = False, #n_jobs: int = 8
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
            GaussianSymmFunc(), # good   
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
    ssf = SiteStatsFingerprint(
        site_featurizer=multi_feat,
        stats=("mean", "std_dev", "minimum", "maximum"),
    )
    all_feat_labels = ssf.feature_labels()

    all_feats = ssf.featurize(structure)
    all_feats = np.expand_dims(np.asarray(all_feats, dtype=np.float32), axis=0)
    #all_feats = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
    if output_feat_labels:
        return all_feats, all_feat_labels
    return all_feats 


'''    for featurizer in site_featurizers:
        ssf = SiteStatsFingerprint(
            site_featurizer=featurizer,
            stats=("mean", "std_dev", "minimum", "maximum"),
        )

        features = ssf.featurize(structure)
        feature_names = ssf.feature_labels()

        all_features.extend(features)
        all_feature_names.extend(feature_names)'''

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

    lob_feats, lob_labels = get_lobster_struc_feat(structure_path=struc_dir, ICOHP_path=ICOHP_dir, ICOBI_path=ICOBI_dir, chrg_path=chrg_dir, output_feat_labels = True)
    mat_feats, mat_labels=get_matminer_struc_feats(structure_path=struc_dir, output_feat_labels = True, )
    #matLob_feats = torch.cat([mat_feats, lob_feats], dim=-1)
    matLob_feats = np.concatenate([mat_feats, lob_feats], axis=-1)
    print(lob_feats.shape, len(lob_labels))
    print(mat_feats.shape, len(mat_labels))
    print(matLob_feats.shape)

    #print(lob_feats.shape, mat_feats.shape)
    #get_matminer_struc_feats()

    #matminer_n_feats=get_matminer_site_feats(struc); #exit()
    #lobster_n_feats=get_lobster_site_feat(structure_path=struc_dir, ICOXX_path=ICOHP_dir, chrg_path=chrg_dir); #exit()
    #matminer_lobster_n_feats = torch.cat([matminer_n_feats, lobster_n_feats], dim=-1)
    #print(matminer_n_feats.shape, lobster_n_feats.shape, matminer_lobster_n_feats.shape)
    #get_soap_feats(structures_df=df)
    #exit()
    #feats.to_csv('feats.csv', index=False)