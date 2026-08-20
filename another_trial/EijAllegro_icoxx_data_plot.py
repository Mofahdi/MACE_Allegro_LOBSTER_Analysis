import os
import pandas as pd
import numpy as np
import torch, dgl
import dcor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from allegroEij_merge import get_allegro_model_transforms, transform_ase_atoms, create_DGL_g, intersect_graphs
from icoxx_graph_converter import Icoxx2Graph
from distance_correlation import evaluate_distance_correlation_matrix_bootstrap

default_dtype = torch.float32
torch.set_default_dtype(default_dtype)

class Allegro_ICOXX:
    def __init__(self, df_path, model_path, device):
        self.df_path=df_path
        self.model_path=model_path
        self.device=device
        self.process_df_and_model() 

    def process_df_and_model(self):
        """ get df with absolute paths and get Allegro's model with ase atoms transforms"""
        df=pd.read_csv(self.df_path)
        df['struc_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'CONTCAR.gz'))
        df['ICOHP_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOHPLIST.lobster.gz'))
        df['ICOBI_dir']=df['mat_path'].apply(lambda x: os.path.join(x, 'ICOBILIST.lobster.gz'))
        self.df=df

        model, transforms=get_allegro_model_transforms(model_path=self.model_path, device=self.device)
        self.model=model
        self.transforms=transforms

    def get_Eijs_EijFeats_icoxxs(self,): # df, model, transforms, device, ICOXX):
        """function to get 'Eijs, EijFeats, icohps, icobis' from Allegro and ICOXX files"""
        Eij_chunks=[]; EijFeats_chunks=[]; icohp_chunks=[]; icobi_chunks=[]; d_chunks=[]
        for i, (cont_path, icohp_path, icobi_path) in enumerate(zip(self.df['struc_dir'].values, self.df['ICOHP_dir'].values, self.df['ICOBI_dir'].values)):
            ase_atoms, data = transform_ase_atoms(structure_path=cont_path, 
                                                  transforms=self.transforms, 
                                                  device=self.device)
            with torch.no_grad():
                output = self.model(data)

            g_allegro=create_DGL_g(ase_atoms=ase_atoms, output=output)
            g_icoxx = Icoxx2Graph(bond_length_cutoff=6).get_graph(path_to_structure=cont_path,
                            path_to_icohplist=icohp_path, path_to_icobilist=icobi_path,
                        ).to(device)
            g_intersect=intersect_graphs(g1=g_allegro, g2=g_icoxx, edge_key="edge_cell_shift")

            Eij_chunks.append(g_intersect.edata['g1_edge_energy']); # [# edges, 1]
            EijFeats_chunks.append(g_intersect.edata['g1_edge_features']) # [# edges, # channels]
            icohp_chunks.append(g_intersect.edata['g2_ICOHPs'].unsqueeze(-1)) # [# edges, 1]
            icobi_chunks.append(g_intersect.edata['g2_ICOBIs'].unsqueeze(-1)) # [# edges, 1]
            d_chunks.append(g_intersect.edata['g2_bond_dist'].unsqueeze(-1)) # [# edges, 1]
        
        self.Eijs = torch.cat(Eij_chunks, dim=0).float().cpu() # [# edges, 1]
        self.EijFeats = torch.cat(EijFeats_chunks, dim=0).float().cpu() # [# edges, # channels]
        self.icohps = torch.cat(icohp_chunks, dim=0).float().cpu() # [# edges, 1]
        self.icobis = torch.cat(icobi_chunks, dim=0).float().cpu() # [# edges, 1]
        self.d = torch.cat(d_chunks, dim=0).float().cpu() # [# edges, 1]
        return self.Eijs, self.EijFeats, self.icohps, self.icobis, self.d

    def PCA_Xij(self):
        """function to get conduct PCA on Eij features"""
        scaler = StandardScaler()
        EijFeats_scaled = np.array(scaler.fit_transform(self.EijFeats), dtype=np.float32)

        # PCA
        pca = PCA(n_components=2)
        self.EijFeats_pca = pca.fit_transform(EijFeats_scaled)
        return pca.explained_variance_ratio_

    def plot_Eij_icoxx(self,
                        icohp: bool=True, 
                        save_plot: bool=True, 
                        Eii_icoxx_plot_name:str = 'Eij_ICOXX.png', 
                        dpi: float = 400):
        """ plot Eij vs. ICOXX
        args:
            icohp: whether to plots ICOHP or ICOBI, 
            save_plot: option to save plot, 
            Eii_icoxx_plot_name: filename of the plot, 
            dpi: plot's resolution
        :return:
            g: csv file "label+'_Eij_all.csv'" contains ICOXX and Eij raw data
            coeff: Pearson's correlation coefficient between Eij and ICOXX 
        """

        icoxxs = self.icohps if icohp else self.icobis
        xlabel = 'ICOHP (eV)' if icohp else 'ICOBI'
        plt.figure(figsize=(4,3.5))
        plt.scatter(icoxxs, self.Eijs, c='blue')
        plt.xlabel(xlabel)
        plt.ylabel(r'$E_{ij}$', fontsize=12)
        ax=plt.gca()
        nx = len(ax.get_xticks())
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
        ny = len(ax.get_yticks())
        #ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
        plt.tight_layout()
        if save_plot:
            plt.savefig(Eii_icoxx_plot_name, dpi=dpi)
        plt.show()

        label = 'ICOHP' if icohp else 'ICOBI'
        pd.DataFrame({xlabel: icoxxs.squeeze(), 'Eij': self.Eijs.squeeze()}).to_csv(label+'_Eij_all.csv', index=False)
        
        scaler = StandardScaler()
        Eijs_scaled = scaler.fit_transform(self.Eijs)
        icoxxs_scaled = scaler.fit_transform(icoxxs)
        return torch.corrcoef(torch.stack([Eijs.reshape(-1), icoxxs.reshape(-1)]))[0, 1].item(), dcor.distance_correlation(Eijs_scaled, icoxxs_scaled)

    def plot_Eij_d(self,
                        save_plot: bool=True, 
                        Eii_icoxx_plot_name:str = 'Eij_d.png', 
                        dpi: float = 400):
        """ plot Eij vs. ICOXX
        args:
            save_plot: option to save plot, 
            Eii_icoxx_plot_name: filename of the plot, 
            dpi: plot's resolution
        :return:
            g: csv file "label+'_Eij_all.csv'" contains ICOXX and Eij raw data
            coeff: Pearson's correlation coefficient between Eij and ICOXX 
        """

        plt.figure(figsize=(4,3.5))
        plt.scatter(self.d, self.Eijs, c='blue')
        plt.xlabel('Bond Distance')
        plt.ylabel(r'$E_{ij}$', fontsize=12)
        ax=plt.gca()
        nx = len(ax.get_xticks())
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
        ny = len(ax.get_yticks())
        #ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
        plt.tight_layout()
        if save_plot:
            plt.savefig(Eii_icoxx_plot_name, dpi=dpi)
        plt.show()

        pd.DataFrame({'bond distance': self.d.squeeze(), 'Eij': self.Eijs.squeeze()}).to_csv('Dist_Eij_all.csv', index=False)

        scaler = StandardScaler()
        Eijs_scaled = scaler.fit_transform(self.Eijs)
        d_scaled = scaler.fit_transform(self.d)
        return torch.corrcoef(torch.stack([Eijs.reshape(-1), self.d.reshape(-1)]))[0, 1].item(), dcor.distance_correlation(Eijs_scaled, d_scaled)

    def plot_PCAEij_icoxx_and_dcor(self, 
                          icohp: bool=True, 
                          save_plot: bool=True, 
                          PCAEiiFeats_icoxx_plot_name:str = 'PCAEijFeatureSpace_ICOXX.png', 
                          dpi: float=400):
        """ plot PCA1 vs. PCA2 of Eij features color-coded with ICOXX
        args:
            icohp: whether to plots ICOHP or ICOBI, 
            save_plot: option to save plot, 
            PCAEiiFeats_icoxx_plot_name: filename of the plot, 
            dpi: plot's resolution
        :return:
            g: csv file "label+'_PCAEijFeats.csv'" contains PCA1 and PCA2 of Eij features and ICOXX raw data
            coeff: PCA1 and PCA2 variances
        """

        icoxxs = self.icohps if icohp else self.icobis
        colorBar_label = "ICOHP (eV)" if icohp else "ICOBI"

        plt.figure(figsize=(4.5,3.5))
        plt.scatter(
            self.EijFeats_pca[:, 0],
            self.EijFeats_pca[:, 1],
            c=icoxxs,
            s=5,
            #cmap="rainbow"
            )
        plt.xlabel(r"PCA1 ($x_{ij}$)", fontsize=12)
        plt.ylabel(r"PCA2 ($x_{ij}$)", fontsize=12)
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
            plt.savefig(PCAEiiFeats_icoxx_plot_name, dpi=dpi)
        plt.show()

        label = 'ICOHP' if icohp else 'ICOBI'
        pd.DataFrame({'PCA1EijFeats': self.EijFeats_pca[:, 0], 'PCA2EijFeats': self.EijFeats_pca[:, 1], colorBar_label: icoxxs.squeeze(),}).to_csv(label+'_PCAEijFeats.csv', index=False)
        #return pca.explained_variance_ratio_, dcor.distance_correlation(X_pca, icoxx_scaled)
        sets={'PCAEijfeats': self.EijFeats_pca, colorBar_label: StandardScaler().fit_transform(icoxxs)}
        mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(sets=sets, num_resamples=100, num_bootstrap=20)
        return mean_mat, std_mat

    def plot_PCAEij_d_and_dcor(self, 
                          save_plot: bool=True, 
                          PCAEiiFeats_dist_plot_name:str = 'PCAEijFeatureSpace_dist.png', 
                          dpi: float=400):
        """ plot PCA1 vs. PCA2 of Eij features color-coded with ICOXX
        args:
            save_plot: option to save plot, 
            PCAEiiFeats_dist_plot_name: filename of the plot, 
            dpi: plot's resolution
        :return:
            g: csv file 'Dist_PCAEijFeats.csv' contains PCA1 and PCA2 of Eij features and distance raw data
            coeff: PCA1 and PCA2 variances
        """

        plt.figure(figsize=(4.5,3.5))
        plt.scatter(
            self.EijFeats_pca[:, 0],
            self.EijFeats_pca[:, 1],
            c=self.d,
            s=5,
            #cmap="rainbow"
            )
        plt.xlabel(r"PCA1 ($x_{ij}$)", fontsize=12)
        plt.ylabel(r"PCA2 ($x_{ij}$)", fontsize=12)
        cbar = plt.colorbar(label='Bond Distance')
        #nbar=len(cbar.get_xticks())
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=len(cbar.ax.get_yticks())+1))
        ax=plt.gca()
        nx = len(ax.get_xticks())
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=nx + 1))
        ny = len(ax.get_yticks())
        #ax.yaxis.set_major_locator(MaxNLocator(nbins=ny + 1))
        plt.tight_layout()
        if save_plot:
            plt.savefig(PCAEiiFeats_dist_plot_name, dpi=dpi)
        plt.show()

        pd.DataFrame({'PCA1EijFeats': self.EijFeats_pca[:, 0], 'PCA2EijFeats': self.EijFeats_pca[:, 1], 'bond distance': self.d.squeeze(),}).to_csv('Dist_PCAEijFeats.csv', index=False)
        #return pca.explained_variance_ratio_, dcor.distance_correlation(X_pca, d_scaled)
        sets={'PCAEijfeats': self.EijFeats_pca, 'd': StandardScaler().fit_transform(self.d)}
        mean_mat, std_mat = evaluate_distance_correlation_matrix_bootstrap(sets=sets, num_resamples=100, num_bootstrap=100)
        return mean_mat, std_mat
        #return dcor.distance_correlation(self.EijFeats_pca, StandardScaler().fit_transform(self.d))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
# initialize class
alleg_icoxx=Allegro_ICOXX(df_path='5_lo_chg_mpInfo_nonmetals.csv', 
                          model_path="../Allegro-OAM-L-0.1.nequip.zip", 
                          device=device)

# i outputted "Eijs, EijFeats, icohps, icobis" 
# because if others want to plot the data differently from how we plot.
# Howerver, our object can plot and will reproduce our work's plots
Eijs, EijFeats, icohps, icobis, dists = alleg_icoxx.get_Eijs_EijFeats_icoxxs()
PCAEijFeats_var=alleg_icoxx.PCA_Xij()
print('PCA EijFeats:', PCAEijFeats_var)

# plot Eij vs. d and get Pearson and dcor correlations
Pcorr_d, dcor_d=alleg_icoxx.plot_Eij_d(save_plot = True, Eii_icoxx_plot_name = 'Eij_d.jpg', dpi = 800)
print('Eij and bond length Pearson and dcor Correlations:',Pcorr_d, dcor_d)
# plot Eij vs. ICOHP and get Pearson and dcor correlations
PCorr_ICOHP, dcor_ICOHP = alleg_icoxx.plot_Eij_icoxx(icohp=True, save_plot=True, Eii_icoxx_plot_name = 'Eij_ICOHP.jpg', dpi = 800)
print('Eij and ICOHP Pearson and dcor Correlations:', PCorr_ICOHP, dcor_ICOHP)
# plot Eij vs. ICOBI and get Pearson and dcor correlations
PCorr_ICOBI, dcor_ICOBI=alleg_icoxx.plot_Eij_icoxx(icohp=False, save_plot=True, Eii_icoxx_plot_name = 'Eij_ICOBI.jpg', dpi = 800)
print('Eij and ICOBI Pearson and dcor Correlations:', PCorr_ICOBI, dcor_ICOBI)


XijD_dcor_mean_mat, XijD_dcor_std_mat = alleg_icoxx.plot_PCAEij_d( 
                        save_plot=True, 
                        PCAEiiFeats_dist_plot_name = 'PCAEijFeatureSpace_dist.jpg', # PCAEijFeatureSpace_ICOHP PCAEijFeatureSpace_ICOBI
                        dpi=700)
#print('d: PCA variances, Eij dcor:', Xij_variances,) # EijD_dcor)
print('\n\nd, Xij dcor (mean, std, p-val):'); print(XijD_dcor_mean_mat); print(XijD_dcor_std_mat); print(XijD_dcor_mean_pvals);
#exit()
# plot PCA(Eij Features) color-coded with ICOHP
#Xij_variances, EijICOHP_dcor = alleg_icoxx.plot_PCAEij_icoxx(
XijICOHP_dcor_mean, XijICOHP_dcor_std = alleg_icoxx.plot_PCAEij_icoxx_and_dcor(
                        icohp=True, 
                        save_plot=True, 
                        PCAEiiFeats_icoxx_plot_name = 'PCAEijFeatureSpace_ICOHP.jpg', # PCAEijFeatureSpace_ICOHP PCAEijFeatureSpace_ICOBI
                        dpi=700)
#print('ICOHP: PCA variances, Eij dcor:', Xij_variances,) # EijICOHP_dcor) 
print('\n\nICOHP, Xij dcor (mean, std, p-val):'); print(XijICOHP_dcor_mean); print(XijICOHP_dcor_std); # EijICOHP_dcor) 

# plot PCA(Eij Features) color-coded with ICOBI
#Xij_variances, EijICOBI_dcor = alleg_icoxx.plot_PCAEij_icoxx(
XijICOBI_dcor_mean, XijICOBI_dcor_std = alleg_icoxx.plot_PCAEij_icoxx_and_dcor(
                        icohp=False, 
                        save_plot=True, 
                        PCAEiiFeats_icoxx_plot_name = 'PCAEijFeatureSpace_ICOBI.jpg', # PCAEijFeatureSpace_ICOHP PCAEijFeatureSpace_ICOBI
                        dpi=700)
#print('ICOBI: PCA variances, Eij dcor:', Xij_variances,) #EijICOBI_dcor)
print('\n\nICOBI, Xij dcor (mean, std, p-val):'); pritn(XijICOBI_dcor_mean); print(XijICOBI_dcor_std);  #EijICOBI_dcor)

