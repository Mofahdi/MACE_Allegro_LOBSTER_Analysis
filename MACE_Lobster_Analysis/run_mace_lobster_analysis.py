import torch
from mace_lobster_node_feature_analysis import MACE_LOBSTER_NodeFeatureAnalysis

if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
    
    analyzer = MACE_LOBSTER_NodeFeatureAnalysis(
                                                df_path='nonmetals_mp_info.csv', 
                                                model_path='./mace-mpa-0-medium.model', 
                                                device=device,
                                                )
    analyzer.process_df_and_model()
    analyzer.extract_features()
    print('\n\nLinear Probing Loewdin charge, Mulliken charge, sum_icohp, and sum_icobi ....')
    print('Loewdin charge:')
    analyzer.linProb(feature_type = 'chg_loew', dpi=800)  # 'chg_loew'  'chg_mull'  'sum_icohp'  'sum_icobi'
    print('\nMulliken charge:')
    analyzer.linProb(feature_type = 'chg_mull', dpi=800)
    print('\nsum_icohp:')
    analyzer.linProb(feature_type = 'sum_icohp', dpi=800)
    print('\nsum_icobi charge:')
    analyzer.linProb(feature_type = 'sum_icobi', dpi=800)


    print('\n\ncomputing dCor from sampled data ....')
    #exit()
    analyzer.compute_distance_correlation(sample_size=300, num_bootstrap=50)

    print('\nPlotting PCA of MACE node features color-coded with Loewdin and Mulliken charges....')
    print('\ncomputing dCor from sampled data between PCA of MACE nodes features and charges....')
    analyzer.plot_pca_charge_dcor_relationship(
                                            plot_name_loewdin = 'PCAMACE_2D_LoewdinChg.jpg', 
                                            plot_name_mulliken = 'PCAMACE_2D_MullChg.jpg',
                                            sample_size=300, 
                                            num_bootstrap=50,
                                            )

