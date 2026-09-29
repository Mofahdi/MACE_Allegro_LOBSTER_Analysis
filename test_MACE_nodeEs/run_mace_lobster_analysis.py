import torch
import dcor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from mace_lobster_node_feature_analysis import MACE_LOBSTER_NodeFeatureAnalysis

if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
    
    analyzer = MACE_LOBSTER_NodeFeatureAnalysis(
                                                df_path='nonmetals_mp_info_test3.csv', 
                                                model_path='./mace-mpa-0-medium.model', 
                                                device=device,
                                                )
    analyzer.process_df_and_model()
    analyzer.extract_features()
    mace_scaler = StandardScaler(copy=False)
    mace_node_feats_scaled = mace_scaler.fit_transform(analyzer.mace_node_feats)
    pca = PCA(n_components=2,) #svd_solver="randomized")
    mace_node_feats_pca = pca.fit_transform(mace_node_feats_scaled)

    lobster_df=analyzer.df_lobster
    
    new_df = analyzer.df_lobster
    new_df.insert(0, 'PC2_MACE', mace_node_feats_pca[:, 1])
    new_df.insert(0, 'PC1_MACE', mace_node_feats_pca[:, 0])
    corr = new_df.corr(method="pearson")
    #print(corr)

    # charge_loew   charge_mull

    plt.figure(figsize=(4.4,3.5))
    plt.scatter(mace_node_feats_pca[:, 0], new_df["charge_loew"], c='blue', alpha=0.1)
    plt.xlabel(r'PC1 (MACE Node Embeddings)')
    plt.ylabel(r'Loewdin Charge (e)',) #fontsize=12
    ax=plt.gca()
    ax = plt.gca()
    r = new_df["charge_loew"].corr(new_df["PC1_MACE"])
    plt.plot([], [], ' ', label=f'$r$ = {r:.3f}')
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
    plt.savefig('loew_mace.jpg', dpi=300)
    plt.close()


    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Pearson r',}, #'fontsize': 9},
        annot_kws={'size': 9}
    )

    # Colorbar font sizes
    cbar = ax.collections[0].colorbar
    cbar.set_label('Pearson r', fontsize=12)  # colorbar label
    cbar.ax.tick_params(labelsize=11)         # colorbar tick numbers

    #plt.title('Pearson Correlation Matrix', fontsize=16, pad=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('PearsonCorr.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    exit()

    print('\n\nLinear Probing Loewdin charge, Mulliken charge, sum_icohp, and sum_icobi ....')
    print('Loewdin charge:')
    analyzer.linProb(feature_type = 'chg_loew', dpi=300) # 'chg_loew'  'chg_mull'  'sum_icohp'  'sum_icobi'
    print('\nMulliken charge:')
    analyzer.linProb(feature_type = 'chg_mull', dpi=300)
    print('\nsum_icohp:')
    analyzer.linProb(feature_type = 'sum_icohp', dpi=300)
    print('\nsum_icobi charge:')
    analyzer.linProb(feature_type = 'sum_icobi', dpi=300)






    print('\nmean ICOHP:')
    # Fit a node-level linear model from MACE descriptors to Sum ICOHP.
    regression_df = pd.DataFrame(analyzer.mace_node_feats)
    regression_df['target'] = np.asarray(
        lobster_df['site_icohp_mean']
    ).reshape(-1)
    regression_df = regression_df.replace([np.inf, -np.inf], np.nan)
    nodes_before_filter = len(regression_df)
    regression_df = regression_df.dropna(axis=0, how='any').reset_index(drop=True)
    #print('\n\nLinear Regression mean ICOHPs...')
    print('Removed nodes with non-finite features or target:', nodes_before_filter - len(regression_df), 'out of',nodes_before_filter)
    clean_mace_node_feats = regression_df.drop(
        columns='target'
    ).to_numpy(dtype=np.float32)
    clean_sumICOHPs = regression_df['target'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        clean_mace_node_feats,
        clean_sumICOHPs,
        test_size=0.2,
        random_state=42,
    )
    #linear_model = make_pipeline(StandardScaler(), LinearRegression())
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)
    r2=r2_score(y_test, y_pred)
    print('Linear regression R2:', r2)
    #print('Linear regression RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('Linear regression MAE:', mean_absolute_error(y_test, y_pred))
    joblib.dump(linear_model, "linear_regression_model_meanICOHP.pkl")

    plt.figure(figsize=(5, 4))
    plt.scatter(y_test, y_pred, c='b', alpha=0.2)
    plot_limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(plot_limits, plot_limits, 'k--', linewidth=1)
    plt.xlabel('Actual Mean ICOHP (eV)')
    plt.ylabel('Predicted Mean ICOHP (eV)')
    ax = plt.gca()
    ax.text(
        0.05, 0.95,
        f"R$^2$ = {r2:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig('linear_regression_mace_feats_vs_meanICOHPs.jpg', dpi=700)
    plt.close()








    print('\nmean ICOBI:')
    regression_df = pd.DataFrame(analyzer.mace_node_feats)
    regression_df['target'] = np.asarray(
        lobster_df['site_icobi_mean']
    ).reshape(-1)
    regression_df = regression_df.replace([np.inf, -np.inf], np.nan)
    nodes_before_filter = len(regression_df)
    regression_df = regression_df.dropna(axis=0, how='any').reset_index(drop=True)
    #print('\n\nLinear Regression mean ICOBIs...')
    print('Removed nodes with non-finite features or target:', nodes_before_filter - len(regression_df), 'out of',nodes_before_filter)
    clean_mace_node_feats = regression_df.drop(
        columns='target'
    ).to_numpy(dtype=np.float32)
    clean_sumICOHPs = regression_df['target'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        clean_mace_node_feats,
        clean_sumICOHPs,
        test_size=0.2,
        random_state=42,
    )
    #linear_model = make_pipeline(StandardScaler(), LinearRegression())
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)
    r2=r2_score(y_test, y_pred)
    print('Linear regression R2:', r2)
    #print('Linear regression RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('Linear regression MAE:', mean_absolute_error(y_test, y_pred))
    joblib.dump(linear_model, "linear_regression_model_meanICOBI.pkl")

    plt.figure(figsize=(5, 4))
    plt.scatter(y_test, y_pred, c='b', alpha=0.2)
    plot_limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(plot_limits, plot_limits, 'k--', linewidth=1)
    plt.xlabel('Actual Mean ICOBI')
    plt.ylabel('Predicted Mean ICOBI')
    ax = plt.gca()
    ax.text(
        0.05, 0.95,
        f"R$^2$ = {r2:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig('linear_regression_mace_feats_vs_meanICOBIs.jpg', dpi=700)
    plt.close()




    ############## max ICOBI
    print('\nmax ICOBI:')
    regression_df = pd.DataFrame(analyzer.mace_node_feats)
    regression_df['target'] = np.asarray(
        lobster_df['site_icobi_max']
    ).reshape(-1)
    regression_df = regression_df.replace([np.inf, -np.inf], np.nan)
    nodes_before_filter = len(regression_df)
    regression_df = regression_df.dropna(axis=0, how='any').reset_index(drop=True)
    #print('\n\nLinear Regression max ICOBIs...')
    print('Removed nodes with non-finite features or target:', nodes_before_filter - len(regression_df), 'out of',nodes_before_filter)
    clean_mace_node_feats = regression_df.drop(
        columns='target'
    ).to_numpy(dtype=np.float32)
    clean_sumICOHPs = regression_df['target'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        clean_mace_node_feats,
        clean_sumICOHPs,
        test_size=0.2,
        random_state=42,
    )
    #linear_model = make_pipeline(StandardScaler(), LinearRegression())
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)
    r2=r2_score(y_test, y_pred)
    print('Linear regression R2:', r2)
    #print('Linear regression RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('Linear regression MAE:', mean_absolute_error(y_test, y_pred))
    joblib.dump(linear_model, "linear_regression_model_maxICOBI.pkl")

    plt.figure(figsize=(5, 4))
    plt.scatter(y_test, y_pred, c='b', alpha=0.2)
    plot_limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(plot_limits, plot_limits, 'k--', linewidth=1)
    plt.xlabel('Actual max ICOBI')
    plt.ylabel('Predicted max ICOBI')
    ax = plt.gca()
    ax.text(
        0.05, 0.95,
        f"R$^2$ = {r2:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig('linear_regression_mace_feats_vs_maxICOBIs.jpg', dpi=700)
    plt.close()





    ############## min ICOHP
    print('\nmin ICOHP:')
    regression_df = pd.DataFrame(analyzer.mace_node_feats)
    regression_df['target'] = np.asarray(
        lobster_df['site_icohp_min']
    ).reshape(-1)
    regression_df = regression_df.replace([np.inf, -np.inf], np.nan)
    nodes_before_filter = len(regression_df)
    regression_df = regression_df.dropna(axis=0, how='any').reset_index(drop=True)
    #print('\n\nLinear Regression max ICOBIs...')
    print('Removed nodes with non-finite features or target:', nodes_before_filter - len(regression_df), 'out of',nodes_before_filter)
    clean_mace_node_feats = regression_df.drop(
        columns='target'
    ).to_numpy(dtype=np.float32)
    clean_sumICOHPs = regression_df['target'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        clean_mace_node_feats,
        clean_sumICOHPs,
        test_size=0.2,
        random_state=42,
    )
    #linear_model = make_pipeline(StandardScaler(), LinearRegression())
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)
    r2=r2_score(y_test, y_pred)
    print('Linear regression R2:', r2)
    #print('Linear regression RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('Linear regression MAE:', mean_absolute_error(y_test, y_pred))
    joblib.dump(linear_model, "linear_regression_model_minICOHP.pkl")

    plt.figure(figsize=(5, 4))
    plt.scatter(y_test, y_pred, c='b', alpha=0.2)
    plot_limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(plot_limits, plot_limits, 'k--', linewidth=1)
    plt.xlabel('Actual min ICOHP')
    plt.ylabel('Predicted min ICOHP')
    ax = plt.gca()
    ax.text(
        0.05, 0.95,
        f"R$^2$ = {r2:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig('linear_regression_mace_feats_vs_minICOHPs.jpg', dpi=700)
    plt.close()





    exit()
    pca = PCA(n_components=2,) #svd_solver="randomized")
    mace_node_feats_pca = pca.fit_transform(mace_node_feats_scaled)
    print('\nvariances in PCA of MACE node embeddings are:', pca.explained_variance_ratio_)
    sum_icohp = analyzer.df_lobster['site_icohp_sum']
    mean_icohp = analyzer.df_lobster['site_icohp_mean']
    sum_icobi = analyzer.df_lobster['site_icobi_sum']
    mean_icobi = analyzer.df_lobster['site_icobi_mean']
    lowedin = analyzer.df_lobster['charge_mull']
    mulliken = analyzer.df_lobster['charge_loew']

    dcor_sumICOHP = dcor.distance_correlation(mace_node_feats_pca, sum_icohp)
    dcor_meanICOHP = dcor.distance_correlation(mace_node_feats_pca, mean_icohp)
    print(dcor_sumICOHP, dcor_meanICOHP) #0.3246871 0.33323073
    dcor_sumICOBI = dcor.distance_correlation(mace_node_feats_pca, sum_icobi)
    dcor_meanICOBI = dcor.distance_correlation(mace_node_feats_pca, mean_icobi)
    print(dcor_sumICOBI, dcor_meanICOBI) #0.50115037 0.42442834
    dcor_lowedin = dcor.distance_correlation(mace_node_feats_pca, lowedin)
    dcor_mull = dcor.distance_correlation(mace_node_feats_pca, mulliken)
    print(dcor_lowedin, dcor_mull)

    
    
    exit()
    print('\n\nLinear Probing Loewdin charge, Mulliken charge, sum_icohp, and sum_icobi ....')
    print('Loewdin charge:')
    analyzer.linProb(feature_type = 'chg_loew', dpi=300) # 'chg_loew'  'chg_mull'  'sum_icohp'  'sum_icobi'
    print('\nMulliken charge:')
    analyzer.linProb(feature_type = 'chg_mull', dpi=300)
    print('\nsum_icohp:')
    analyzer.linProb(feature_type = 'sum_icohp', dpi=300)
    print('\nsum_icobi charge:')
    analyzer.linProb(feature_type = 'sum_icobi', dpi=300)


    print('\n\ncomputing dCor from sampled data ....')
    #exit()
    analyzer.compute_distance_correlation(sample_size=300, num_bootstrap=40)

    print('\nPlotting PCA of MACE node features color-coded with Loewdin and Mulliken charges....')
    print('\ncomputing dCor from sampled data between PCA of MACE nodes features and charges....')
    analyzer.plot_pca_charge_dcor_relationship(
                                            plot_name_loewdin = 'PCAMACE_2D_LoewdinChg.jpg', 
                                            plot_name_mulliken = 'PCAMACE_2D_MullChg.jpg',
                                            sample_size=300, 
                                            num_bootstrap=40,
                                            )

