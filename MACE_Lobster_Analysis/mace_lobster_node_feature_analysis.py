import os
import json
import psutil
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


import torch, dgl
from mace.calculators.mace import MACECalculator
from ase.io import read 
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

        mat_id_chunks=[]; mace_node_chunks=[]; lobster_chunks=[]; 
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
                print(f"FAILED: {cont_path}")
                print(f"ERROR: {e}")

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

        mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(sets=sets, 
                                                                           sample_size = sample_size, 
                                                                           num_bootstrap = num_bootstrap)
        
        print('dCor between LOBSTER, MACE node features (mean, std):'); print(mean_mat); print(std_mat); 

        mean_mat.to_json("MACE_lobster_dcor_mean.json", orient="records", indent=4)
        std_mat.to_json("MACE_lobster_dcor_std.json", orient="records", indent=4)

        self.mace_node_feats_scaled = mace_node_feats_scaled

    def plot_pca_charge_dcor_relationship(
        self,
        plot_name_loewdin: str = 'PCAMACE_2D_LoewdinChg.jpg',
        plot_name_mulliken: str = 'PCAMACE_2D_MullChg.jpg',
        sample_size: int = 400,
        num_bootstrap: int = 100,
    ) -> None:
        """Project MACE node features into PCA space and quantify their relationship to charge.

        This method standardizes the MACE node-feature matrix, reduces it to the
        first two principal components, and creates 2D scatter plots colored by
        Loewdin and Mulliken charges. It then estimates bootstrap-based distance
        correlation between the PCA embedding and each charge vector and saves the
        summary statistics as JSON files.

        Parameters
        ----------
        plot_name_loewdin : str, optional
            Output filename for the PCA plot colored by Loewdin charge.
        plot_name_mulliken : str, optional
            Output filename for the PCA plot colored by Mulliken charge.
        sample_size : int, optional
            Number of node-feature samples used in each bootstrap iteration.
        num_bootstrap : int, optional
            Number of bootstrap replicates used to estimate the dCor statistics.

        Returns
        -------
        None
            Saves the PCA plots and the mean/std distance-correlation summaries.
        """
        pca = PCA(n_components=2,) #svd_solver="randomized")
        self.mace_node_feats_pca = pca.fit_transform(self.mace_node_feats_scaled)
        print('\nvariances in PCA of MACE node embeddings are:', pca.explained_variance_ratio_)
        loewdin_charge = self.df_lobster['charge_mull']
        mulliken_charge = self.df_lobster['charge_loew']

        plt.figure(figsize=(4.5,3.5))
        plt.scatter(
            self.mace_node_feats_pca[:, 0],
            self.mace_node_feats_pca[:, 1],
            c=loewdin_charge,
            s=5,
            #cmap="rainbow"
            )
        plt.xlabel(r"PC1 (MACE Node Embeddings)", fontsize=11)
        plt.ylabel(r"PC2 (MACE Node Embeddings)", fontsize=11)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        cbar = plt.colorbar()
        cbar.set_label('Loewdin Charge (e)', fontsize=12)
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
        plt.savefig(plot_name_loewdin, dpi=800) # dpi=dpi)


        plt.figure(figsize=(4.5,3.5))
        plt.scatter(
            self.mace_node_feats_pca[:, 0],
            self.mace_node_feats_pca[:, 1],
            c=mulliken_charge,
            s=5,
            #cmap="rainbow"
            )
        plt.xlabel(r"PC1 (MACE Node Embeddings)", fontsize=11)
        plt.ylabel(r"PC2 (MACE Node Embeddings)", fontsize=11)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        cbar = plt.colorbar()
        cbar.set_label('Mulliken Charge (e)', fontsize=12)
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
        plt.savefig(plot_name_mulliken, dpi=800) # dpi=dpi)


        sets = {
            'MACE': self.mace_node_feats_pca,
            'Loewdin': loewdin_charge,
            'Mulliken': mulliken_charge,
            }
        
        mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(
                                                                            sets=sets, 
                                                                            sample_size = sample_size, 
                                                                            num_bootstrap = num_bootstrap,
                                                                            )
        
        print('dCor between charge and PCA MACE node features (mean, std):'); print(mean_mat); print(std_mat); 

        mean_mat.to_json("MACE_charges_dcor_mean.json", orient="records", indent=4)
        std_mat.to_json("MACE_charges_dcor_std.json", orient="records", indent=4)

        pd.DataFrame(
            {
                "PC1 MACE": self.mace_node_feats_pca[:, 0],
                "PC2 MACE": self.mace_node_feats_pca[:, 1],
                'Loewdin': loewdin_charge,
                'Mulliken': mulliken_charge, 
            }
        ).to_csv("PCA_MACE_Loew_Mull.csv", index=False)



    def linProb(self, feature_type: str, dpi: int = 800):
        """
        Train a linear model for ICOHP or ICOBI prediction.

        Args:
            feature_type (bool): If True, predict ICOHP; otherwise predict ICOBI.
            dpi (int): DPI for the saved regression plot.

        Returns:
            None. Prints MAE and R^2, saves the model, and saves the
            actual-vs-predicted plot.
        """
        # self.ids = np.concatenate(mat_id_chunks) 
        # self.mace_node_feats=mace_node_feats
        # self.lobster_feats=lobster_feats
        # self.df_lobster = df_lobster
        # 'site_icohp_sum', 'site_icobi_sum', 'charge_mull', 'charge_loew'
        ids = self.ids
        X = np.asarray(self.mace_node_feats, dtype=np.float32)

        if feature_type == 'chg_loew':
            y = self.df_lobster['charge_loew']
            xlabel = 'Actual Loewdin Charge (e)'
            ylabel = 'Predicted Loewdin Charge (e)'

        elif feature_type == 'chg_mull':
            y = self.df_lobster['charge_mull']
            xlabel = 'Actual Mulliken Charge (e)'
            ylabel = 'Predicted Mulliken Charge (e)'

        elif feature_type == 'sum_icohp':
            y = self.df_lobster['site_icohp_sum']
            xlabel = 'Actual Sum ICOBI'
            ylabel = 'Predicted Sum ICOBI'

        elif feature_type == 'sum_icobi':
            y = self.df_lobster['site_icobi_sum']
            xlabel = 'Actual Sum ICOHP (eV)'
            ylabel = 'Predicted Sum ICOHP (eV)'

        else:
            raise ValueError("you can only enter one of the following options: " \
            "1- 'charge_loew'," \
            "2- 'chg_mull', " \
            "3- 'sum_icohp', " \
            "4- 'sum_icobi' ")

        # Remove invalid or non-finite samples
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]
        ids = ids[valid]

        ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(
            ids, X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        joblib.dump(model, feature_type+"_linear_regression_model.pkl")
        y_pred = model.predict(X_test)

        pd.DataFrame({
                        'id': ids_test, 
                        'true_'+feature_type: y_test,
                        'pred_'+feature_type: y_pred,
                      }).to_csv(feature_type+'_LinProb_results.csv',
                                index=False,)
        
        #print("Regression coefficients:", model.coef_)
        #print("Intercept:", model.intercept_)
        print("MAE:", mean_absolute_error(y_test, y_pred))
        #print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
        r2 = r2_score(y_test, y_pred)
        print("R2:", r2)

        # Predicted versus actual ICOHP
        plt.figure(figsize=(4.4, 3.5)) #(5, 4)
        plt.scatter(y_test, y_pred, c='b', alpha=0.1,) #label=f"R$^2$ = {r2:.3f}")
        limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        plt.plot(limits, limits, "k--",) #label="ideal prediction")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax = plt.gca()
        # ax.text(
        #     0.05, 0.9,
        #     f"R$^2$ = {r2:.3f}",
        #     transform=ax.transAxes,
        #     fontsize=10,
        #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=1)
        # )
        plt.plot([], [], ' ', label=f"R$^2$ = {r2:.3f}")
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
        plt.savefig(feature_type+"_linearProbe.jpg", dpi=dpi)
        #plt.show()
        plt.close()




