import torch
import argparse
from mace_lobster_node_feature_analysis import MACE_LOBSTER_NodeFeatureAnalysis

parser = argparse.ArgumentParser(description="Run the MACE/Lobster analysis workflow.")
parser.add_argument("--num-dcor-samples", type=int, default=50, help="Number of bootstrap samples for distance-correlation estimation.")
parser.add_argument("--dcor-sample-size", type=int, default=300, help="Sample size used within each distance-correlation bootstrap iteration.")
parser.add_argument("--extension", type=str, default='.png', help="File extension for saved figures, including the leading dot (e.g., .png or .pdf).")
parser.add_argument("--dpi", type=int, default=800, help="Figure resolution in dots per inch for saved plots.")
args = parser.parse_args()

# good numbers are: 20 then 50,000, then 800
num_dcor_samples = args.num_dcor_samples
dcor_sample_size = args.dcor_sample_size
extension = args.extension
dpi = args.dpi


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device = 'cuda'
else: 
    device = 'cpu'

analyzer = MACE_LOBSTER_NodeFeatureAnalysis(
                                            df_path='mp_data_paths.csv', 
                                            model_path='./mace-mpa-0-medium.model', 
                                            device=device,
                                            )
analyzer.process_df_and_model()
analyzer.extract_features()
analyzer.plot_pearson_heatmap(extension=extension, dpi=dpi)


print('\n\nLinear Probing Loewdin charge, Mulliken charge, sum_icohp, and sum_icobi ....')
print('Loewdin charge:')
analyzer.fit_linProb(feature_type = 'chg_loew', extension=extension, dpi=dpi) # 'chg_loew'  'chg_mull'  'sum_icohp'  'sum_icobi'  'mean_icohp'  'mean_icobi'
print('\nMulliken charge:')
analyzer.fit_linProb(feature_type = 'chg_mull', extension=extension, dpi=dpi)

print('\nsum_icohp:')
analyzer.fit_linProb(feature_type = 'sum_icohp', extension=extension, dpi=dpi)
print('\nsum_icobi:')
analyzer.fit_linProb(feature_type = 'sum_icobi', extension=extension, dpi=dpi)



print('\n\ncomputing dCor from sampled data ....')
analyzer.compute_distance_correlation(sample_size=dcor_sample_size, num_bootstrap=num_dcor_samples, extension=extension, dpi=dpi)



print('\nPlotting PCA of MACE node embeddings color-coded with Loewdin and Mulliken charges....')
print('\ncomputing dCor from sampled data between PCA of MACE nodes embeddings and charges....')
analyzer.plot_pc_space_and_charge_dcor(
                                    sample_size=dcor_sample_size, 
                                    num_bootstrap=num_dcor_samples,
                                    dpi=dpi,
                                    )

print('\nPlotting charge vs. PC1 and PC2 of MACE node embeddings....')
analyzer.plot_pc1_pc2_vs_charge(extension=extension, dpi=dpi)

print('\nPlotting sumICOXX vs. PC2 of MACE node embeddings....')
analyzer.plot_pc2_vs_sum_icoxx(extension=extension, dpi=dpi)
#analyzer.plot_PC2MACE_sumICOXX(icoxx_type='icohp', dpi=dpi)






