"""
Functions for evaluating distance correlation between feature sets using bootstrap resampling.
"""

import itertools
import numpy as np
import pandas as pd
import dcor


def evaluate_distance_correlation_matrix_bootstrap(
    sets: dict[str, np.ndarray],
    sample_size: int,
    num_bootstrap: int = 200,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise distance correlation (and p-values) using bootstrap resampling
    to estimate mean and standard deviation of dCor.

    Parameters
    ----------
    sets : dict
        Dictionary where keys are names (str) and values are numpy arrays (samples × features).
        All arrays must have the same number of samples (rows).
    num_resamples : int, optional
        Number of random permutations for the independence test (p-values).
    num_bootstrap : int, optional
        Number of bootstrap samples for mean/std estimation.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    mean_mat : pd.DataFrame
        Mean distance correlation matrix across bootstrap samples.
    std_mat : pd.DataFrame
        Standard deviation of distance correlations across bootstrap samples.
    """
    rng = np.random.default_rng(random_state)
    names = list(sets.keys())
    n_samples = list(sets.values())[0].shape[0]

    mean_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    std_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    #mean_pvals = pd.DataFrame(index=names, columns=names, dtype=float)

    for i, j in itertools.combinations_with_replacement(names, 2):
        dcor_values = []
       # pval_values = []
        if i == j:
            mean_mat.loc[i, j] = 1.0
            std_mat.loc[i, j] = 0.0
            continue

        for _ in range(num_bootstrap):
            # Bootstrap resample indices with replacement
            idx = rng.integers(0, n_samples, sample_size)
            X_i = sets[i][idx]
            X_j = sets[j][idx]
            #X_i = np.asarray(sets[i], dtype=np.float64)
            #X_j = np.asarray(sets[j], dtype=np.float64)

            # Compute distance correlation and permutation p-value
            #val = dcor.distance_correlation(X_i, X_j)
            # if i == j:
            #     val = dcor.distance_correlation(X_i, X_j)
            # else:
            val = dcor.distance_correlation(
                np.asarray(X_i, dtype=np.float64),
                np.asarray(X_j, dtype=np.float64),
            )
            #test = dcor.independence.distance_covariance_test(
             #   X_i, X_j, num_resamples=num_resamples, random_state=random_state
            #)

            dcor_values.append(val)
            #pval_values.append(test.p_value)
        
        #print(dcor_values)
        mean_val = np.mean(dcor_values)
        std_val = np.std(dcor_values)
        #mean_pval = np.mean(pval_values)

        mean_mat.loc[i, j] = mean_mat.loc[j, i] = mean_val
        std_mat.loc[i, j] = std_mat.loc[j, i] = std_val
        #mean_pvals.loc[i, j] = mean_pvals.loc[j, i] = mean_pval

    return mean_mat, std_mat #mean_pvals
