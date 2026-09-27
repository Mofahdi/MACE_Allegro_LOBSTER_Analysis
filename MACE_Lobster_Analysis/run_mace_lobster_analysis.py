import torch
import argparse
import matplotlib.pyplot as plt
from mace_lobster_node_feature_analysis import MACE_LOBSTER_NodeFeatureAnalysis

parser = argparse.ArgumentParser(description="Run the MACE/Lobster analysis workflow.")
parser.add_argument("--num-dcor-samples", type=int, default=50, help="Number of bootstrap samples for distance-correlation estimation.")
parser.add_argument("--dcor-sample-size", type=int, default=300, help="Sample size used within each distance-correlation bootstrap iteration.")
parser.add_argument("--dpi", type=int, default=800, help="Figure resolution in dots per inch for saved plots.")
args = parser.parse_args()

# good numbers are: 20 then 50,000, then 800
num_dcor_samples = args.num_dcor_samples
dcor_sample_size = args.dcor_sample_size
dpi = args.dpi


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
analyzer.plot_Pearson_feats()

print('\n\nLinear Probing Loewdin charge, Mulliken charge, sum_icohp, and sum_icobi ....')
print('Loewdin charge:')
analyzer.linProb(feature_type = 'chg_loew', dpi=dpi) # 'chg_loew'  'chg_mull'  'sum_icohp'  'sum_icobi'  'mean_icohp'  'mean_icobi'
print('\nMulliken charge:')
analyzer.linProb(feature_type = 'chg_mull', dpi=dpi)

print('\nsum_icohp:')
analyzer.linProb(feature_type = 'sum_icohp', dpi=dpi)
print('\nsum_icobi:')
analyzer.linProb(feature_type = 'sum_icobi', dpi=dpi)

print('\nmean_icohp:')
analyzer.linProb(feature_type = 'mean_icohp', dpi=dpi)
print('\nmean_icobi:')
analyzer.linProb(feature_type = 'mean_icobi', dpi=dpi)



print('\n\ncomputing dCor from sampled data ....')
#exit()
analyzer.compute_distance_correlation(sample_size=dcor_sample_size, num_bootstrap=num_dcor_samples)

print('\nPlotting PCA of MACE node features color-coded with Loewdin and Mulliken charges....')
print('\ncomputing dCor from sampled data between PCA of MACE nodes features and charges....')
analyzer.plot_pca_charge_and_dcor(
                                    plot_name_loewdin = 'PCAMACE_2D_LoewdinChg.jpg', 
                                    plot_name_mulliken = 'PCAMACE_2D_MullChg.jpg',
                                    sample_size=dcor_sample_size, 
                                    num_bootstrap=num_dcor_samples,
                                    )



