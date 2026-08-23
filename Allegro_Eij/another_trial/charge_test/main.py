import torch
from EijAllegro_icoxx_data_plot import Allegro_ICOXX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
# initialize class
alleg_icoxx=Allegro_ICOXX(df_path='5_lo_chg_mpInfo_nonmetals_test2.csv', 
                          model_path="../../Allegro-OAM-L-0.1.nequip.zip", 
                          device=device)

# we outputted "Eijs, EijFeats, icohps, icobis" 
# because if others want to plot the data differently from how we plot.
# Howerver, our object can plot and will reproduce our work's plots
Eijs, EijFeats, icohps, icobis, dists, chgs = alleg_icoxx.get_Eijs_EijFeats_icoxxs()
PCAEijFeats_var=alleg_icoxx.PCA_Xij()
print('PCA EijFeats variation:', PCAEijFeats_var)
print('number of edges', Eijs.shape)
# plot Eij vs. d and get Pearson and dcor correlations
Pcorr_chg, dcor_chg=alleg_icoxx.plot_Eij_chg(save_plot = True, Eii_icoxx_plot_name = 'Eij_chg.jpg', dpi = 800)
print('Eij and chg Pearson and dcor Correlations:',Pcorr_chg, dcor_chg)





exit()
# plot Eij vs. ICOHP and get Pearson and dcor correlations
PCorr_ICOHP, dcor_ICOHP = alleg_icoxx.plot_Eij_icoxx(icohp=True, save_plot=True, Eii_icoxx_plot_name = 'Eij_ICOHP.jpg', dpi = 800)
print('Eij and ICOHP Pearson and dcor Correlations:', PCorr_ICOHP, dcor_ICOHP)
# plot Eij vs. ICOBI and get Pearson and dcor correlations
PCorr_ICOBI, dcor_ICOBI=alleg_icoxx.plot_Eij_icoxx(icohp=False, save_plot=True, Eii_icoxx_plot_name = 'Eij_ICOBI.jpg', dpi = 800)
print('Eij and ICOBI Pearson and dcor Correlations:', PCorr_ICOBI, dcor_ICOBI)



# plot PCA(Eij Features) color-coded with bond length (d). Also, get dcor between PCA and d
XijD_dcor_mean_mat, XijD_dcor_std_mat = alleg_icoxx.plot_PCAEij_d_and_dcor( 
                                                                        num_dcor_samples=50000, # had memory issues processing everything here so got dcor
                                                                        save_plot=True, 
                                                                        PCAEiiFeats_dist_plot_name = 'PCAEijFeatureSpace_dist.jpg', # PCAEijFeatureSpace_ICOHP PCAEijFeatureSpace_ICOBI
                                                                        dpi=800)
print('\n\nd, Xij dcor (mean, std):'); print(XijD_dcor_mean_mat); print(XijD_dcor_std_mat);
XijD_dcor_mean_mat.to_json("XijD_dcor_mean_mat.json", orient="records", indent=4)
XijD_dcor_std_mat.to_json("XijD_dcor_std_mat.json", orient="records", indent=4)

# plot PCA(Eij Features) color-coded with ICOHP. Also, get dcor between PCA and ICOHP
XijICOHP_dcor_mean, XijICOHP_dcor_std = alleg_icoxx.plot_PCAEij_icoxx_and_dcor(
                                                                            num_dcor_samples=50000,
                                                                            icohp=True, 
                                                                            save_plot=True, 
                                                                            PCAEiiFeats_icoxx_plot_name = 'PCAEijFeatureSpace_ICOHP.jpg', # PCAEijFeatureSpace_ICOHP PCAEijFeatureSpace_ICOBI
                                                                            dpi=800)
print('\n\nICOHP, Xij dcor (mean, std):'); print(XijICOHP_dcor_mean); print(XijICOHP_dcor_std); # EijICOHP_dcor)
XijICOHP_dcor_mean.to_json("XijICOHP_dcor_mean.json", orient="records", indent=4)
XijICOHP_dcor_std.to_json("XijICOHP_dcor_std.json", orient="records", indent=4)

# plot PCA(Eij Features) color-coded with ICOBI. Also, get dcor between PCA and ICOBI
XijICOBI_dcor_mean, XijICOBI_dcor_std = alleg_icoxx.plot_PCAEij_icoxx_and_dcor(
                                                                            num_dcor_samples=50000,
                                                                            icohp=False, 
                                                                            save_plot=True, 
                                                                            PCAEiiFeats_icoxx_plot_name = 'PCAEijFeatureSpace_ICOBI.jpg', # PCAEijFeatureSpace_ICOHP PCAEijFeatureSpace_ICOBI
                                                                            dpi=800)
print('\n\nICOBI, Xij dcor (mean, std):'); print(XijICOBI_dcor_mean); print(XijICOBI_dcor_std);  #EijICOBI_dcor)
XijICOBI_dcor_mean.to_json("XijICOBI_dcor_mean.json", orient="records", indent=4)
XijICOBI_dcor_std.to_json("XijICOBI_dcor_std.json", orient="records", indent=4)



