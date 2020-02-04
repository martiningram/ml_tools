import numpy as np


def remove_correlated_variables(covariate_df, corr_threshold=0.95):

    # Returns names of columns to keep so that those with an absolute
    # correlation above the threshold are dropped.
    covariate_names = covariate_df.columns

    corrs = np.corrcoef(covariate_df.values.T)
    corrs = np.tril(corrs, -1)

    abs_corr = np.abs(corrs)
    highly_correlated = np.where(abs_corr > 0.95)
    to_drop = highly_correlated[0]

    to_keep = np.setdiff1d(np.arange(len(covariate_names)), to_drop)
    to_keep_names = np.array(covariate_names)[to_keep]

    return to_keep_names
