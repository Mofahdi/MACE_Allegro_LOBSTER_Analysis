import torch
from mace_lobster_node_feature_analysis import MACE_LOBSTER_NodeFeatureExtractor

if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
    
    analyzer = MACE_LOBSTER_NodeFeatureExtractor(
                                                df_path='5_lo_chg_mpInfo_nonmetals.csv', 
                                                model_path='./mace-mpa-0-medium.model', 
                                                device=device,
                                                )
    analyzer.process_df_and_model()
    analyzer.extract_features()
    analyzer.compute_distance_correlation(sample_size=400, num_bootstrap=100)
    analyzer.plot_pca_charge_distributions()
