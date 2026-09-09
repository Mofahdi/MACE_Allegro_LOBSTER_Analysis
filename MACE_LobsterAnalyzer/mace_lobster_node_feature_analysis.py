import os
import json
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch, dgl
from mace.calculators.mace import MACECalculator
from ase.io import read 
from lobster_site_feature_extractor import extract_lobster_site_features
from distance_correlation import evaluate_distance_correlation_matrix_bootstrap

default_dtype = torch.float32
torch.set_default_dtype(default_dtype)

class MACE_LOBSTER_NodeFeatureExtractor:
    def __init__(self, df_path: str, model_path: str, device: torch.device) -> None:
        """Initialize the feature-extraction pipeline for a material dataset.

        Parameters
        ----------
        df_path : str
            Path to the CSV file containing material metadata and paths.
        model_path : str
            Path to the trained MACE model used to compute descriptors.
        device : torch.device
            Device on which the MACE calculator should run.
        """
        self.df_path: str = df_path
        self.model_path: str = model_path
        self.device: torch.device = device
        self.process_df_and_model()

    def process_df_and_model(self) -> None:
        """Load the dataset and initialize the MACE calculator.

        This method reads the CSV metadata table, creates paths to the required
        structure and LOBSTER files, and stores the prepared dataframe together
        with a configured MACE calculator instance.
        """
        df: pd.DataFrame = pd.read_csv(self.df_path)
        df['struc_dir'] = df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR.gz'))
        df['ICOHP_dir'] = df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster.gz'))
        df['ICOBI_dir'] = df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster.gz'))
        df['CHRG_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CHARGE.lobster.gz'))

        self.df = df

    def extract_features(self) -> None:
        """Extract MACE and LOBSTER node-level features for all entries in the dataset.

        For each structure, this method loads the relevant LOBSTER outputs,
        computes the local site features, and evaluates the corresponding MACE
        descriptors. The extracted arrays are concatenated and stored on the
        instance for downstream analysis.
        """
        self.mace_calc = MACECalculator(model_paths=self.model_path, model_dtype='float32', device=self.device)

        mace_node_chunks=[]; lobster_chunks=[]; 
        for i, (cont_path, icohp_path, icobi_path, chrg_dir) in enumerate(
                                                                            zip(
                                                                                self.df['struc_dir'].values, 
                                                                                self.df['ICOHP_dir'].values, 
                                                                                self.df['ICOBI_dir'].values, 
                                                                                self.df['CHRG_dir'].values
                                                                                )
                                                                            ):
            try:
                if i==0:
                    lobster_n_feats, lobster_feat_labels = extract_lobster_site_features(structure_path=cont_path, 
                                                                    icohp_path=icohp_path, icobi_path=icobi_path, 
                                                                    chrg_path=chrg_dir, output_feat_labels=True)
                else:
                    lobster_n_feats = extract_lobster_site_features(structure_path=cont_path, 
                                                            icohp_path=icohp_path, icobi_path=icobi_path, 
                                                            chrg_path=chrg_dir, output_feat_labels=False) # [# nodes, feats]    
                lobster_chunks.append(lobster_n_feats)

                ase_atoms=read(cont_path)
                mace_descs=self.mace_calc.get_descriptors(atoms=ase_atoms).astype(np.float32) # [# nodes, feats]
                mace_node_chunks.append(mace_descs)

                del ase_atoms; del lobster_n_feats; del mace_descs;
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e: 
                print(f"FAILED: {cont_path}")
                print(f"ERROR: {e}")

        print('done extracting features')

        mace_node_feats = np.concatenate(mace_node_chunks, axis=0)
        del mace_node_chunks; 
        df_lobster = pd.DataFrame(np.concatenate(lobster_chunks, axis=0) , columns=lobster_feat_labels)
        df_lobster = df_lobster.dropna(axis=1)
        lobster_feat_labels = df_lobster.columns.tolist()
        lobster_feats = df_lobster.to_numpy()
        del lobster_chunks

        self.mace_node_feats=mace_node_feats
        self.lobster_feats=lobster_feats
        print('dimensions of nodes with MACE and lobster features:')
        print('[# nodes, # MACE feats]:', mace_node_feats.shape)
        print('[# nodes, # lobster feats]:', lobster_feats.shape)

    def compute_distance_correlation(self, sample_size: int = 400, num_bootstrap: int = 100) -> None:
        """Compute the distance-correlation matrix between MACE and LOBSTER node features.

        The method standardizes the extracted MACE and LOBSTER feature matrices,
        samples matched subsets from both datasets, and evaluates the bootstrap-
        based distance correlation between the two feature spaces. The resulting
        mean and standard-deviation matrices are printed and saved as JSON files.

        Parameters
        ----------
        sample_size : int, optional
            Number of node-feature samples used in each bootstrap resampling.
        num_bootstrap : int, optional
            Number of bootstrap iterations used to estimate the mean and standard
            deviation of the distance-correlation matrix.

        Returns
        -------
        None
            The method stores the standardized MACE features on the instance as
            ``self.mace_node_feats_scaled`` and writes summary JSON outputs to disk.
        """
        mace_scaler = StandardScaler(copy=False)
        mace_node_feats_scaled = mace_scaler.fit_transform(self.mace_node_feats)
        lobster_scaler = StandardScaler(copy=False)
        lobster_feats_scaled = lobster_scaler.fit_transform(self.lobster_feats)
        sets = {
            'MACE': mace_node_feats_scaled,
            'LOBSTER': lobster_feats_scaled, 
            }

        print('computing dCor from sampled data ....')
        mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(sets=sets, 
                                                                           sample_size = sample_size, 
                                                                           num_bootstrap = num_bootstrap)
        
        print('\n\ndCor between LOBSTER, MACE node features (mean, std):'); print(mean_mat); print(std_mat); 

        mean_mat.to_json("MACE_lobster_dcor_mean.json", orient="records", indent=4)
        std_mat.to_json("MACE_lobster_dcor_std.json", orient="records", indent=4)

        self.mace_node_feats_scaled = mace_node_feats_scaled

    def plot_pca_charge_distributions(self) -> None:
        """Create 2D PCA visualizations for MACE features colored by charge values.

        This method projects the standardized MACE node features into two
        principal components and plots them twice: once colored by the Loewdin
        charge and once by the Mulliken charge. Both plots are saved as image
        files on disk.
        """
        pca = PCA(n_components=2,) #svd_solver="randomized")
        mace_node_feats_pca = pca.fit_transform(self.mace_node_feats_scaled)

        plt.figure(figsize=(4.5,3.5))
        plt.scatter(
            mace_node_feats_pca[:, 0],
            mace_node_feats_pca[:, 1],
            c=self.lobster_feats[:, -1],
            s=5,
            #cmap="rainbow"
            )
        plt.xlabel(r"PCA1 (MACE Node Features)", fontsize=12)
        plt.ylabel(r"PCA2 (MACE Node Features)", fontsize=12)
        cbar = plt.colorbar(label='Charge Loewdin')
        #nbar=len(cbar.get_xticks())
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=len(cbar.ax.get_yticks())+1))
        ax=plt.gca()
        nx = len(ax.get_xticks())
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
        ny = len(ax.get_yticks())
        #ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
        plt.tight_layout()
        #if save_plot:
        plt.savefig('PCAMACE_2D_LoewdinChg.jpg', dpi=800) # dpi=dpi)

        plt.figure(figsize=(4.5,3.5))
        plt.scatter(
            mace_node_feats_pca[:, 0],
            mace_node_feats_pca[:, 1],
            c=self.lobster_feats[:, -2],
            s=5,
            #cmap="rainbow"
            )
        plt.xlabel(r"PCA1 (MACE Node Features)", fontsize=12)
        plt.ylabel(r"PCA2 (MACE Node Features)", fontsize=12)
        cbar = plt.colorbar(label='Mulliken Charge')
        #nbar=len(cbar.get_xticks())
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=len(cbar.ax.get_yticks())+1))
        ax=plt.gca()
        nx = len(ax.get_xticks())
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
        ny = len(ax.get_yticks())
        #ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
        plt.tight_layout()
        #if save_plot:
        plt.savefig('PCAMACE_2D_MullChg.jpg', dpi=800) # dpi=dpi)



