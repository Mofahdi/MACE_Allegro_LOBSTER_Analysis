import os
import json
import pickle
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score

from scipy.ndimage import gaussian_filter, map_coordinates

import torch, dgl
from mace.calculators.mace import MACECalculator
from ase.io import read 
from pymatgen.core.structure import Structure

from lobster_site_feature_extractor import extract_lobster_site_features
from distance_correlation import evaluate_distance_correlation_matrix_bootstrap

default_dtype = torch.float32
torch.set_default_dtype(default_dtype)

class MACE_LOBSTER_NodeFeatureAnalysis:
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


        # set paths to distinguish outputs and create those paths if do not exist
        self.lobster_corr = os.path.join(os.getcwd(), 'MACE_Lobster_corr')
        if not os.path.isdir(self.lobster_corr):
            os.mkdir(self.lobster_corr)

        self.chrg_corr = os.path.join(os.getcwd(), 'MACE_chrg_corr')
        if not os.path.isdir(self.chrg_corr):
            os.mkdir(self.chrg_corr)

        self.lp = os.path.join(os.getcwd(), 'LinearProbe')
        if not os.path.isdir(self.lp):
            os.mkdir(self.lp)


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

        mat_id_chunks=[]; 
        mace_node_chunks=[]; 
        lobster_chunks=[]; 
        for i, (mat_id, cont_path, icohp_path, icobi_path, chrg_dir) in enumerate(
                                                                            zip(
                                                                                self.df['id'].values,
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
                mat_id_chunks.append(np.array([mat_id]*lobster_n_feats.shape[0]))
                ase_atoms=read(cont_path)
                mace_descs=self.mace_calc.get_descriptors(atoms=ase_atoms).astype(np.float32) # [# nodes, feats]                
                mace_node_chunks.append(mace_descs)

                del ase_atoms; del lobster_n_feats; del mace_descs;
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e: 
                raise RuntimeError(
                    f"Feature extraction failed for material {mat_id}: {cont_path}"
                ) from e

        mace_node_feats = np.concatenate(mace_node_chunks, axis=0)
        del mace_node_chunks; 
        df_lobster = pd.DataFrame(np.concatenate(lobster_chunks, axis=0) , columns=lobster_feat_labels)
        df_lobster = df_lobster.dropna(axis=1)
        lobster_feat_labels = df_lobster.columns.tolist()
        lobster_feats = df_lobster.to_numpy()
        del lobster_chunks

        self.ids = np.concatenate(mat_id_chunks) 
        self.mace_node_feats=mace_node_feats
        self.lobster_feats=lobster_feats
        self.df_lobster = df_lobster
        print('done extracting features')
        print('dimensions of nodes with MACE and lobster features:')
        print('[# nodes, # MACE feats]:', mace_node_feats.shape)
        print('[# nodes, # lobster feats]:', lobster_feats.shape)

        mace_scaler = StandardScaler()
        self.mace_node_feats_scaled = mace_scaler.fit_transform(self.mace_node_feats)
        pca = PCA(n_components=2,) #svd_solver="randomized")
        self.mace_node_feats_pca = pca.fit_transform(self.mace_node_feats_scaled)
        print('\nvariances in PCA of MACE node embeddings are:', pca.explained_variance_ratio_)

        self.df_all: pd.DataFrame = self.df_lobster.copy()
        self.df_all.insert(0, 'PC2_MACE', self.mace_node_feats_pca[:, 1])
        self.df_all.insert(0, 'PC1_MACE', self.mace_node_feats_pca[:, 0])
        self.df_all.to_csv(os.path.join(self.lobster_corr, 'Pearson_data.csv'), index=False)



    def plot_pearson_heatmap(self, extension: str = '.jpg', dpi: int = 800,) -> None:
        """Plot the Pearson correlation matrix between PCA-compressed MACE features and LOBSTER features.

        The first two principal components of the MACE node embeddings are appended to the
        LOBSTER feature dataframe, the pairwise Pearson correlation matrix is computed, and the
        resulting heatmap is saved to disk as an image and JSON summary.

        Returns
        -------
        None
            Saves a correlation heatmap and the raw correlation table as output files.
        """
        new_df: pd.DataFrame = self.df_all.copy()
        corr = new_df.corr(method="pearson")
        #print(corr)

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        #plt.figure(figsize=(10, 6))
        #plt.figure(figsize=(6.5, 4.5))
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.5,
            linecolor='white',
            #cbar_kws={'label': 'Pearson r',}, #'fontsize': 9},
            annot_kws={'size': 13}
        )

        # Colorbar font sizes
        cbar = ax.collections[0].colorbar
        cbar.set_label('Pearson Correlation', fontsize=13)  # colorbar label
        cbar.ax.tick_params(labelsize=13)         # colorbar tick numbers

        #plt.title('Pearson Correlation Matrix', fontsize=16, pad=16)
        plt.xticks(rotation=45, ha='right', fontsize=13)
        plt.yticks(rotation=0, fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.lobster_corr, 'mace_lobster_pearson_heatmap'+extension), dpi=dpi, bbox_inches='tight')
        plt.close()
        corr.to_json(os.path.join(self.lobster_corr, "mace_lobster_pearson_correlation.json"), orient="records", indent=4)



    def compute_distance_correlation(
            self, sample_size: int = 400, 
            num_bootstrap: int = 100, 
            extension: str = '.jpg', 
            dpi: int = 800,
            ) -> None:
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
        lobster_scaler = StandardScaler()
        lobster_feats_scaled = lobster_scaler.fit_transform(self.lobster_feats)
        sets = {
            'MACE Node\nEmbeddings': self.mace_node_feats_scaled,
            'LOBSTER Features\n(Charge + ICOXX)': lobster_feats_scaled, 
            }

        mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(sets=sets, 
                                                                           sample_size = sample_size, 
                                                                           num_bootstrap = num_bootstrap)
        
        print('dCor between LOBSTER, MACE node features (mean, std):'); print(mean_mat); print(std_mat); 

        mean_mat.to_json(os.path.join(self.lobster_corr, "mace_lobster_dcor_mean.json"), orient="records", indent=4)
        std_mat.to_json(os.path.join(self.lobster_corr, "mace_lobster_dcor_std.json"), orient="records", indent=4)


        mask = np.triu(np.ones_like(mean_mat, dtype=bool), k=1)
        #plt.figure(figsize=(3.8, 3.8))
        #plt.figure(figsize=(3.2, 3.2))
        #plt.figure(figsize=(4, 4))
        plt.figure(figsize=(3.5, 3.5))
        ax = sns.heatmap(
            mean_mat,
            mask=mask,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 12},
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Mean dCor"},
        )
        #Colorbar font sizes
        cbar = ax.collections[0].colorbar
        cbar.set_label('Mean dCor', fontsize=11)  # colorbar label
        cbar.ax.tick_params(labelsize=8)         # colorbar tick numbers

        #plt.title('Pearson Correlation Matrix', fontsize=16, pad=16)
        #plt.xticks(rotation=45, ha='right', fontsize=13)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)

        #ax.set_title("MACE–LOBSTER Mean Distance Correlation")
        plt.tight_layout()
        plt.savefig(os.path.join(self.lobster_corr, "mace_lobster_dcor_mean_heatmap"+extension), dpi=dpi, bbox_inches="tight")
        plt.close()



    def plot_pc_space_and_charge_dcor(
        self,
        #plot_name_loewdin: str = ,
        #plot_name_mulliken: str = ,
        sample_size: int = 400,
        num_bootstrap: int = 100,
        extension: str = '.jpg', 
        dpi: int = 800,
    ) -> None:
        """Project MACE node features into PCA space and quantify their relationship to charge.

        This method standardizes the MACE node-feature matrix, reduces it to the
        first two principal components, and creates 2D scatter plots colored by
        Loewdin and Mulliken charges. It then estimates bootstrap-based distance
        correlation between the PCA embedding and each charge vector and saves the
        summary statistics as JSON files.

        Parameters
        ----------
        # plot_name_loewdin : str, optional
        #     Output filename for the PCA plot colored by Loewdin charge.
        # plot_name_mulliken : str, optional
        #     Output filename for the PCA plot colored by Mulliken charge.
        sample_size : int, optional
            Number of node-feature samples used in each bootstrap iteration.
        num_bootstrap : int, optional
            Number of bootstrap replicates used to estimate the dCor statistics.

        Returns
        -------
        None
            Saves the PCA plots and the mean/std distance-correlation summaries.
        """
        loewdin_charge = self.df_all['charge_loew']
        mulliken_charge = self.df_all['charge_mull']
        ylabels= ['Loewdin Charge (e)', 'Mulliken Charge (e)']
        plot_names = ['mace_pca_loewdin_charge_scatter', 'mace_pca_mulliken_charge_scatter']

        for i, (chg_label, plt_name) in enumerate(zip(ylabels, plot_names)):
            if i==0:
                color_dat = loewdin_charge.copy()
            if i==1:
                color_dat = mulliken_charge.copy()
            
            #plt.figure(figsize=(4.5,3.5))
            #plt.figure(figsize=(3.8 ,3.8))
            #plt.figure(figsize=(4.4, 3.5))
            plt.figure(figsize=(4.4, 3.5))
            plt.scatter(
                self.df_all['PC1_MACE'], #self.mace_node_feats_pca[:, 0],
                self.df_all['PC2_MACE'], #self.mace_node_feats_pca[:, 1],
                c=color_dat,
                s=5,
                #cmap="rainbow"
                )
            plt.xlabel(r"PC1 (MACE Node Embeddings)", fontsize=11)
            plt.ylabel(r"PC2 (MACE Node Embeddings)", fontsize=11)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            cbar = plt.colorbar()
            cbar.set_label(chg_label, fontsize=12)
            cbar.ax.tick_params(labelsize=11)
            #nbar=len(cbar.get_xticks())
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=len(cbar.ax.get_yticks())+1))
            ax=plt.gca()
            nx = len(ax.get_xticks())
            #ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
            ny = len(ax.get_yticks())
            #ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
            plt.tight_layout()
            #if save_plot:
            plt.savefig(os.path.join(self.chrg_corr, plt_name+extension), dpi=dpi) # dpi=dpi)
            plt.close()


        sets = {
            'PC1-PC2 MACE': self.mace_node_feats_pca,
            'Loewdin': loewdin_charge,
            'Mulliken': mulliken_charge,
            }
        
        mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(
                                                                            sets=sets, 
                                                                            sample_size = sample_size, 
                                                                            num_bootstrap = num_bootstrap,
                                                                            )
        
        print('dCor between charge and PC1-PC2 MACE node features (mean, std):'); print(mean_mat); print(std_mat); 

        mean_mat.to_json(os.path.join(self.chrg_corr, "mace_pca_charge_dcor_mean.json"), orient="records", indent=4)
        std_mat.to_json(os.path.join(self.chrg_corr, "mace_pca_charge_dcor_std.json"), orient="records", indent=4)

        pd.DataFrame(
            {
                "PC1 MACE": self.df_all['PC1_MACE'],
                "PC2 MACE": self.df_all['PC2_MACE'],
                'Loewdin': loewdin_charge,
                'Mulliken': mulliken_charge, 
            }
        ).to_csv(os.path.join(self.chrg_corr, "mace_pca_charge_summary.csv"), index=False)



    def plot_pc1_pc2_vs_charge(self, extension: str = '.jpg', dpi: int = 800) -> None:
        """Plot both MACE principal components against Loewdin and Mulliken charge values.

        Parameters
        ----------
        dpi : int
            Output image resolution in dots per inch for each saved plot.

        Returns
        -------
        None
            Saves a scatter plot for each principal-component and charge-type pair.
        """
        new_df: pd.DataFrame = self.df_all.copy()

        # the order here: column in df, label in plot, plot name suffix
        components = [
            ("PC1_MACE", "PC1 (MACE Node Embeddings)", "PC1"),
            ("PC2_MACE", "PC2 (MACE Node Embeddings)", "PC2"),
        ]

        # the order here: charge type, column in df, label in plot
        charge_targets = [
            ("chg_loew", "charge_loew", "Loewdin Charge (e)"),
            ("chg_mull", "charge_mull", "Mulliken Charge (e)"),
        ]

        for pc_column, pc_label, pc_suffix in components:
            for chrg_type, charge_column, chrg_label in charge_targets:
                chrg_dat: pd.Series = new_df[charge_column]

                plt.figure(figsize=(4.4, 3.5))
                #plt.scatter(new_df[pc_column], chrg_dat, c='blue', alpha=0.1) 
                #plt.scatter(new_df[pc_column], chrg_dat, s = 10, c="#00A651", alpha=0.1)
                #plt.scatter(new_df[pc_column], chrg_dat, s = 10, c="#00C853", alpha=0.1)
                
                x, y = new_df[pc_column].to_numpy().reshape(-1), chrg_dat.to_numpy().reshape(-1)
                counts, xedges, yedges = np.histogram2d(x, y, bins=500)

                smoothed = gaussian_filter(
                    counts, sigma=1.5, mode="constant"
                )
                rng = np.random.default_rng(42)
                idx = rng.choice(
                    len(x), size=min(100_000, len(x)), replace=False
                )
                xp, yp = x[idx], y[idx]
                gx = (xp - xedges[0]) / (xedges[1] - xedges[0]) - 0.5
                gy = (yp - yedges[0]) / (yedges[1] - yedges[0]) - 0.5

                colors = map_coordinates(
                    smoothed,
                    [gx, gy],
                    order=1,
                    mode="nearest",
                    prefilter=False,
                )

                # Draw sparse points first and dense points last.
                order = np.argsort(colors)

                fig, ax = plt.subplots(figsize=(4.4, 3.5))

                scatter = ax.scatter(
                    xp[order],
                    yp[order],
                    c=colors[order],
                    cmap="plasma",
                    s=5,
                    alpha=1,
                    edgecolors="none",
                    rasterized=True,
                )

                cbar = fig.colorbar(
                    scatter, ax=ax,
                    #label="Atom Point Density"
                )
                cbar.set_label("Atom Point Density", fontsize=11)

                
                plt.xlabel(pc_label, fontsize=12)
                plt.ylabel(chrg_label, fontsize=12)
                plt.xticks(fontsize=11)
                plt.yticks(fontsize=11)
                ax = plt.gca()
                r = chrg_dat.corr(new_df[pc_column])
                plt.plot([], [], ' ', label=f'$r$ = {r:.2f}')
                ax.legend(
                    loc='best',
                    frameon=True,
                    framealpha=1,
                    edgecolor='black',
                    borderpad=0.3,
                    handlelength=0,
                    handletextpad=0,
                    labelspacing=0,
                    fontsize=10,
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.chrg_corr, f'mace_{pc_suffix.lower()}_vs_{chrg_type.replace("chg_", "")}_charge'+extension),
                    dpi=dpi,
                )
                plt.close()



    def plot_pc2_vs_sum_icoxx(self, extension: str = '.jpg', dpi: int = 800) -> None:
        """Plot the second MACE principal component versus summed ICOHP/ICOBI values.

        Parameters
        ----------
        dpi : int
            Output image resolution in dots per inch for the saved scatter plots.

        Returns
        -------
        None
            Saves one PCA-vs-sum-ICOXX summary figure for each considered property.
        """
        new_df: pd.DataFrame = self.df_all.copy()

        icoxx_types = ['icohp', 'icobi']
        icoxx_labels = ['Sum ICOHP (eV)', 'Sum ICOBI']

        for i, (icoxx_type, icoxx_label) in enumerate(zip(icoxx_types, icoxx_labels)):
            if icoxx_type == 'icohp':
                icoxx_data = new_df['site_icohp_sum'].copy()
                #color = "#E4432E"
                color = "#DD2D2D"
            if icoxx_type == 'icobi':
                icoxx_data = new_df['site_icobi_sum'].copy()
                #color = "#286AD4"
                color = "#0852F1"

            #plt.figure(figsize=(4.4,3.5))
            #plt.figure(figsize=(3.8, 3.8))
            plt.figure(figsize=(4.3, 3.5))
            #plt.scatter(new_df['PC2_MACE'], icoxx_data, c='blue', alpha=0.1)
            #plt.scatter(new_df['PC2_MACE'], icoxx_data, s = 10, c=color, alpha=0.1)
            
            x, y = new_df['PC2_MACE'].to_numpy().reshape(-1), icoxx_data.to_numpy().reshape(-1)
            counts, xedges, yedges = np.histogram2d(x, y, bins=500)

            smoothed = gaussian_filter(
                counts, sigma=1.5, mode="constant"
            )
            rng = np.random.default_rng(42)
            idx = rng.choice(
                len(x), size=min(100_000, len(x)), replace=False
            )
            xp, yp = x[idx], y[idx]
            gx = (xp - xedges[0]) / (xedges[1] - xedges[0]) - 0.5
            gy = (yp - yedges[0]) / (yedges[1] - yedges[0]) - 0.5

            colors = map_coordinates(
                smoothed,
                [gx, gy],
                order=1,
                mode="nearest",
                prefilter=False,
            )

            # Draw sparse points first and dense points last.
            order = np.argsort(colors)

            fig, ax = plt.subplots(figsize=(4.4, 3.5))

            scatter = ax.scatter(
                xp[order],
                yp[order],
                c=colors[order],
                cmap="plasma",
                s=5,
                alpha=1,
                edgecolors="none",
                rasterized=True,
            )

            cbar = fig.colorbar(
                scatter, ax=ax,
                #label="Atom Point Density"
            )
            cbar.set_label("Atom Point Density", fontsize=11)
            
            plt.xlabel(r'PC2 (MACE Node Embeddings)', fontsize=12)
            plt.ylabel(icoxx_label, fontsize=12) #fontsize=12
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            #ax=plt.gca()
            ax = plt.gca()
            r = icoxx_data.corr(new_df["PC2_MACE"])
            plt.plot([], [], ' ', label=f'$r$ = {r:.2f}')
            ax.legend(
                loc='best',
                frameon=True,
                framealpha=1,
                edgecolor='black',
                borderpad=0.3,
                handlelength=0,
                handletextpad=0,
                labelspacing=0, 
                fontsize=10,
            )
            plt.tight_layout()
            #if save_plot:
            plt.show()
            plt.savefig(os.path.join(self.lobster_corr, f'mace_pc2_vs_sum_{icoxx_type}{extension}'), dpi=dpi)
            plt.close()




    def fit_linProb(self, feature_type: str, extension: str = '.jpg', dpi: int = 800) -> None:
        """Train a linear regression model to predict a target scalar property from MACE descriptors.

        Parameters
        ----------
        feature_type : str
            Name of the target variable to predict. Supported values are
            ``'chg_loew'``, ``'chg_mull'``, ``'sum_icohp'``, ``'sum_icobi'``,
            ``'mean_icohp'``, and ``'mean_icobi'``.
        dpi : int, optional
            Resolution used when saving the predicted-versus-actual plot.

        Returns
        -------
        None
            Trains a linear model, saves the fitted model and predictions, prints
            MAE and R^2, and saves the actual-vs-predicted diagnostic plot.
        """
        mat_df=self.df.copy()

        ids = self.ids
        X = np.asarray(self.mace_node_feats, dtype=np.float32)

        if feature_type == 'chg_loew':
            y = self.df_lobster['charge_loew']
            xlabel = 'Actual Loewdin Charge (e)'
            ylabel = 'Predicted Loewdin Charge (e)'
            color = "#00C853"

        elif feature_type == 'chg_mull':
            y = self.df_lobster['charge_mull']
            xlabel = 'Actual Mulliken Charge (e)'
            ylabel = 'Predicted Mulliken Charge (e)'
            color = "#00C853"

        elif feature_type == 'sum_icohp':
            y = self.df_lobster['site_icohp_sum']
            xlabel = 'Actual Sum ICOHP (eV)'
            ylabel = 'Predicted Sum ICOHP (eV)'
            #color = "#C83F3F"
            color = '#DD2D2D'

        elif feature_type == 'sum_icobi':
            y = self.df_lobster['site_icobi_sum']
            xlabel = 'Actual Sum ICOBI'
            ylabel = 'Predicted Sum ICOBI'
            #color = "#2979FF"
            color = "#0852F1"

        else:
            raise ValueError("you can only enter one of the following options: " \
            "1- 'charge_loew'," \
            "2- 'chg_mull', " \
            "3- 'sum_icohp', " \
            "4- 'sum_icobi' " )

        # Remove invalid or non-finite samples
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]
        ids = ids[valid]


        # Build an node-level dataframe
        node_df = pd.DataFrame(
            {
                "id": ids,
                "target": y,
                #"xij_feat": X,
            }
        )
        # add features to node_df
        for i in range(X.shape[1]):
            node_df[f"feature_{i}"] = X[:, i]

        # this path saves
        split_path = os.path.join(self.lp, "material_train_test_split.csv")
        if not os.path.exists(split_path):

            mat_df["reduced_formula"] = mat_df["struc_dir"].apply(
                lambda x: Structure.from_file(x).composition.reduced_formula
            )

            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=0.2,
                random_state=42
            )

            train_idx, test_idx = next(
                splitter.split(mat_df, groups=mat_df["reduced_formula"])
            )

            train_df = mat_df.iloc[train_idx].copy()
            test_df = mat_df.iloc[test_idx].copy()

            # Verify no composition leakage
            assert set(train_df["reduced_formula"]).isdisjoint(
                set(test_df["reduced_formula"])
            )
            train_ids = mat_df["id"].iloc[train_idx].tolist()
            test_ids = mat_df["id"].iloc[test_idx].tolist()
            split_df = pd.DataFrame(
                {
                    "id": np.concatenate([train_ids, test_ids]),
                    "split": ["train"] * len(train_ids) + ["test"] * len(test_ids),
                }
            )
            print("# Train materials:", len(train_df), "# Test materials:", len(test_df))
            split_df.to_csv(split_path, index=False)
            print(f'saved "./LinearProbe/material_train_test_split.csv" split for a later Linear Probe')
            
        else:
            split_df = pd.read_csv(split_path)
            train_ids = split_df.loc[split_df["split"] == "train", "id"].to_numpy()
            test_ids = split_df.loc[split_df["split"] == "test", "id"].to_numpy()
            print(f'loaded "./LinearProbe/material_train_test_split.csv" to ensure same split')

        train_edge_df = node_df[node_df["id"].isin(train_ids)].copy()
        test_edge_df = node_df[node_df["id"].isin(test_ids)].copy()

        train_edge_df = train_edge_df.merge(
            mat_df, #[material_cols],
            on="id",
            how="left",
        )
        test_edge_df = test_edge_df.merge(
            mat_df, #[material_cols],
            on="id",
            how="left",
        )

        X_train = train_edge_df[[f"feature_{i}" for i in range(X.shape[1])]].to_numpy()
        y_train = train_edge_df["target"].to_numpy()
        X_test = test_edge_df[[f"feature_{i}" for i in range(X.shape[1])]].to_numpy()
        y_test = test_edge_df["target"].to_numpy()
        ids_test = test_edge_df["id"].to_numpy()



        # ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(
        #     ids, X, y, test_size=0.2, random_state=42
        # )

        model = LinearRegression()
        model.fit(X_train, y_train)
        with open(os.path.join(self.lp, f"mace_linear_probe_{feature_type}_model.pkl"), 'wb') as f:
            pickle.dump(model, f)
        y_pred = model.predict(X_test)

        pd.DataFrame({
                        'id': ids_test, 
                        'true_'+feature_type: y_test,
                        'pred_'+feature_type: y_pred,
                      }).to_csv(os.path.join(self.lp, f'mace_linear_probe_{feature_type}_results.csv'),
                                index=False,)
        
        #print("Regression coefficients:", model.coef_)
        #print("Intercept:", model.intercept_)
        print("MAE:", mean_absolute_error(y_test, y_pred))
        #print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
        r2 = r2_score(y_test, y_pred)
        print("R2:", r2)

        # Predicted versus actual ICOHP
        plt.figure(figsize=(4.4, 3.5)) #(5, 4)
        #plt.scatter(y_test, y_pred, c='b', alpha=0.1,) #label=f"R$^2$ = {r2:.3f}")
        plt.scatter(y_test, y_pred, c=color, alpha=0.1, s = 10) #label=f"R$^2$ = {r2:.3f}")
        limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        plt.plot(limits, limits, "k--",) #label="ideal prediction")
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        ax = plt.gca()
        # ax.text(
        #     0.05, 0.9,
        #     f"R$^2$ = {r2:.3f}",
        #     transform=ax.transAxes,
        #     fontsize=10,
        #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=1)
        # )
        plt.plot([], [], ' ', label=f"R$^2$ = {r2:.2f}")
        ax.legend(
            loc='best',
            frameon=True,
            framealpha=1,
            edgecolor='black',
            borderpad=0.3,
            handlelength=0,
            handletextpad=0,
            labelspacing=0, 
            fontsize=10,
        )

        #plt.legend(loc='upper left', frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.lp, f"mace_linear_probe_{feature_type}{extension}"), dpi=dpi)
        #plt.show()
        plt.close()


