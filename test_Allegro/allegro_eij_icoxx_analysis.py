from __future__ import annotations

import os
from typing import Any

import dcor
import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from allegro_graph_pipeline import (
    create_DGL_g,
    get_allegro_model_transforms,
    intersect_graphs,
    transform_ase_atoms,
)
from distance_correlation import evaluate_distance_correlation_matrix_bootstrap
from icoxx_graph import Icoxx2Graph

default_dtype: torch.dtype = torch.float32
torch.set_default_dtype(default_dtype)


class AllegroEijIcoxxAnalyzer:
    """Load a structure table, run the Allegro model, and analyze Eij/ICOXX relationships."""

    def __init__(self, df_path: str, model_path: str, device: torch.device) -> None:
        """Initialize the analysis object and prepare the model and dataframe."""
        self.df_path: str = df_path
        self.model_path: str = model_path
        self.device: torch.device = device
        self.process_df_and_model()

    def process_df_and_model(self) -> None:
        """Load the structure dataframe and initialize the Allegro model transforms."""
        df = pd.read_csv(self.df_path)
        df["struc_dir"] = df["mat_path"].apply(lambda x: os.path.join(x, "CONTCAR.gz"))
        df["ICOHP_dir"] = df["mat_path"].apply(lambda x: os.path.join(x, "ICOHPLIST.lobster.gz"))
        df["ICOBI_dir"] = df["mat_path"].apply(lambda x: os.path.join(x, "ICOBILIST.lobster.gz"))
        self.df = df

        model, transforms=get_allegro_model_transforms(model_path=self.model_path, device=self.device)
        self.model=model
        self.transforms=transforms

    def get_Eijs_EijFeats_icoxxs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract matching Allegro Eij values and ICOXX features across all structures."""
        mat_id_chunks: list[torch.Tensor] = []
        Eij_chunks: list[torch.Tensor] = []
        EijFeats_chunks: list[torch.Tensor] = []
        icohp_chunks: list[torch.Tensor] = []
        icobi_chunks: list[torch.Tensor] = []
        d_chunks: list[torch.Tensor] = []

        #
        for mat_id, cont_path, icohp_path, icobi_path in zip(
            self.df['id'].values,
            self.df["struc_dir"].values,
            self.df["ICOHP_dir"].values,
            self.df["ICOBI_dir"].values,
        ):
            ase_atoms, data = transform_ase_atoms(
                structure_path=cont_path,
                transforms=self.transforms,
                device=self.device,
            )
            with torch.no_grad():
                output = self.model(data)

            g_allegro = create_DGL_g(ase_atoms=ase_atoms, output=output)
            g_icoxx = Icoxx2Graph(bond_length_cutoff=6).get_graph(
                path_to_structure=cont_path,
                path_to_icohplist=icohp_path,
                path_to_icobilist=icobi_path,
            ).to(self.device)
            g_intersect = intersect_graphs(g1=g_allegro, g2=g_icoxx, edge_key="edge_cell_shift")

            Eij_chunks.append(g_intersect.edata['g1_edge_energy']); # [# edges, 1]
            EijFeats_chunks.append(g_intersect.edata['g1_edge_features']) # [# edges, # channels]
            icohp_chunks.append(g_intersect.edata['g2_ICOHPs'].unsqueeze(-1)) # [# edges, 1]
            icobi_chunks.append(g_intersect.edata['g2_ICOBIs'].unsqueeze(-1)) # [# edges, 1]
            d_chunks.append(g_intersect.edata['g2_bond_dist'].unsqueeze(-1)) # [# edges, 1]
            mat_id_chunks.append(np.array([mat_id]*g_intersect.num_edges()))

        self.ids = np.concatenate(mat_id_chunks) # [# edges, 1]
        self.Eijs = torch.cat(Eij_chunks, dim=0).float().cpu() # [# edges, 1]
        self.EijFeats = torch.cat(EijFeats_chunks, dim=0).float().cpu() # [# edges, # channels]
        self.icohps = torch.cat(icohp_chunks, dim=0).float().cpu() # [# edges, 1]
        self.icobis = torch.cat(icobi_chunks, dim=0).float().cpu() # [# edges, 1]
        self.d = torch.cat(d_chunks, dim=0).float().cpu() # [# edges, 1]
        return self.ids, self.Eijs, self.EijFeats, self.icohps, self.icobis, self.d

    def PCA_Xij(self) -> np.ndarray:
        """Standardize the Eij feature matrix and project it into a 2D PCA space."""
        scaler = StandardScaler()
        EijFeats_scaled = np.array(scaler.fit_transform(self.EijFeats), dtype=np.float32)

        # PCA
        pca = PCA(n_components=2)
        self.EijFeats_pca = pca.fit_transform(EijFeats_scaled)
        return pca.explained_variance_ratio_

    def plot_Eij_d(
        self,
        save_plot: bool = True,
        Eii_icoxx_plot_name: str = "Eij_d.png",
        dpi: float = 400,
    ) -> tuple[float, float]:
        """Plot Eij against bond distance and return correlation metrics.

        Args:
            save_plot: If True, save the figure to disk.
            Eii_icoxx_plot_name: File name used when saving the figure.
            dpi: Resolution used for saved plots.

        Returns:
            A tuple of Pearson correlation and distance-correlation values.
        """

        plt.figure(figsize=(4,3.5))
        plt.scatter(self.d, self.Eijs, c='blue')
        plt.xlabel('Bond Distance (Å)')
        plt.ylabel(r'Allegro $E_{ij}$ (eV)', fontsize=12)
        ax=plt.gca()
        #nx = len(ax.get_xticks())
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
        #ny = len(ax.get_yticks())
        #ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
        plt.tight_layout()
        if save_plot:
            plt.savefig(Eii_icoxx_plot_name, dpi=dpi)
        plt.show()

        pd.DataFrame({"bond distance": self.d.squeeze(), "Eij": self.Eijs.squeeze()}).to_csv(
            "Dist_Eij_all.csv",
            index=False,
        )

        scaler = StandardScaler()
        Eijs_scaled = scaler.fit_transform(self.Eijs)
        d_scaled = scaler.fit_transform(self.d)
        return (
            torch.corrcoef(torch.stack([self.Eijs.reshape(-1), self.d.reshape(-1)]))[0, 1].item(),
            dcor.distance_correlation(Eijs_scaled, d_scaled),
        )

    def plot_Eij_icoxx(
        self,
        icohp: bool = True,
        save_plot: bool = True,
        Eii_icoxx_plot_name: str = "Eij_ICOXX.png",
        dpi: float = 400,
    ) -> tuple[float, float]:
        """Plot Eij against ICOHP or ICOBI and return correlation metrics.

        Args:
            icohp: When True, use ICOHP; otherwise use ICOBI.
            save_plot: If True, save the figure to disk.
            Eii_icoxx_plot_name: File name used when saving the figure.
            dpi: Resolution used for saved plots.

        Returns:
            A tuple of Pearson correlation and distance-correlation values.
        """

        icoxxs = self.icohps if icohp else self.icobis
        xlabel = 'ICOHP (eV)' if icohp else 'ICOBI'
        plt.figure(figsize=(4,3.5))
        plt.scatter(icoxxs, self.Eijs, c='blue')
        plt.xlabel(xlabel)
        plt.ylabel(r'Allegro $E_{ij}$ (eV)', fontsize=12)
        ax=plt.gca()
        nx = len(ax.get_xticks())
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
        ny = len(ax.get_yticks())
        #ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
        plt.tight_layout()
        if save_plot:
            plt.savefig(Eii_icoxx_plot_name, dpi=dpi)
        plt.show()

        label = "ICOHP" if icohp else "ICOBI"
        pd.DataFrame({'id': self.ids, xlabel: icoxxs.squeeze(), "Eij": self.Eijs.squeeze()}).to_csv(
            label + "_Eij_all.csv",
            index=False,
        )

        scaler = StandardScaler()
        Eijs_scaled = scaler.fit_transform(self.Eijs)
        icoxxs_scaled = scaler.fit_transform(icoxxs)
        return (
            torch.corrcoef(torch.stack([self.Eijs.reshape(-1), icoxxs.reshape(-1)]))[0, 1].item(),
            dcor.distance_correlation(Eijs_scaled, icoxxs_scaled),
        )

    def plot_PCAXijFeats_d_and_dcor(
        self,
        num_dcor_samples: int,
        dcor_sample_size: int,
        save_plot: bool = True,
        PCAXijFeats_dist_plot_name: str = "PCAXijFeatureSpace_dist.png",
        dpi: float = 400,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plot the PCA manifold of xij features against bond distance.

        Args:
            num_dcor_samples: Number of bootstrap samples used for distance-correlation
                estimation.
            dcor_sample_size: Sample size for each bootstrap draw.
            save_plot: If True, save the figure to disk.
            PCAEiiFeats_dist_plot_name: Output filename for the saved figure.
            dpi: Saved-figure resolution.

        Returns:
            Mean and standard-deviation distance-correlation matrices for the PCA and
            distance variables.
        """
        plt.figure(figsize=(4.5, 3.5))
        plt.scatter(
            self.EijFeats_pca[:, 0],
            self.EijFeats_pca[:, 1],
            c=self.d,
            s=5,
            #cmap="rainbow"
            )
        plt.xlabel(r"PC1 of Allegro Edge Embeddings", fontsize=11)
        plt.ylabel(r"PC2 of Allegro Edge Embeddings", fontsize=11)
        cbar = plt.colorbar(label='Bond Distance (Å)')
        #nbar=len(cbar.get_xticks())
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=len(cbar.ax.get_yticks())+1))
        ax=plt.gca()
        nx = len(ax.get_xticks())
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
        ny = len(ax.get_yticks())
        #ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
        plt.tight_layout()
        if save_plot:
            plt.savefig(PCAXijFeats_dist_plot_name, dpi=dpi)
        plt.show()

        pd.DataFrame(
            {
                "PCA1XijFeats": self.EijFeats_pca[:, 0],
                "PCA2XijFeats": self.EijFeats_pca[:, 1],
                "bond distance": self.d.squeeze(),
            }
        ).to_csv("Dist_PCAXijFeats.csv", index=False)
        sets = {"PCAXijfeats": self.EijFeats_pca, "d": StandardScaler().fit_transform(self.d)}
        mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(
            sets=sets,
            num_bootstrap=num_dcor_samples,
            sample_size=dcor_sample_size,
        )
        return mean_mat, std_mat
        #return dcor.distance_correlation(self.EijFeats_pca, StandardScaler().fit_transform(self.d))

    def plot_PCAXij_icoxx_and_dcor(
        self,
        icohp: bool,
        num_dcor_samples: int,
        dcor_sample_size: int,
        save_plot: bool = True,
        PCAXijFeats_icoxx_plot_name: str = "PCAXijFeatureSpace_ICOXX.png",
        dpi: float = 400,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plot the PCA manifold of xij features against ICOHP/ICOBI values."""
        icoxxs = self.icohps if icohp else self.icobis
        colorBar_label = "ICOHP (eV)" if icohp else "ICOBI"

        plt.figure(figsize=(4.5, 3.5))
        plt.scatter(
            self.EijFeats_pca[:, 0],
            self.EijFeats_pca[:, 1],
            c=icoxxs,
            s=5,
            #cmap="rainbow"
            )
        plt.xlabel(r"PC1 of Allegro Edge Embeddings", fontsize=11)
        plt.ylabel(r"PC2 of Allegro Edge Embeddings", fontsize=11)
        cbar = plt.colorbar(label=colorBar_label)
        #nbar=len(cbar.get_xticks())
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=len(cbar.ax.get_yticks())+1))
        ax=plt.gca()
        nx = len(ax.get_xticks())
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
        ny = len(ax.get_yticks())
        #ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
        plt.tight_layout()
        if save_plot:
            plt.savefig(PCAXijFeats_icoxx_plot_name, dpi=dpi)
        plt.show()

        label = "ICOHP" if icohp else "ICOBI"
        pd.DataFrame(
            {
                "PCA1XijFeats": self.EijFeats_pca[:, 0],
                "PCA2XijFeats": self.EijFeats_pca[:, 1],
                colorBar_label: icoxxs.squeeze(),
            }
        ).to_csv(label + "_PCAXijFeats.csv", index=False)
        sets = {"PCAXijfeats": self.EijFeats_pca, colorBar_label: StandardScaler().fit_transform(icoxxs)}
        mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(
            sets=sets,
            num_bootstrap=num_dcor_samples,
            sample_size=dcor_sample_size,
        )
        return mean_mat, std_mat

