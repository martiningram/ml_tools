import autograd.numpy as anp
import autograd.scipy.linalg as aspl


def solve_via_cholesky(k_chol, y):
    """Solves a positive definite linear system via a Cholesky decomposition.

    Args:
        k_chol: The Cholesky factor of the matrix to solve. A lower triangular
            matrix, perhaps more commonly known as L.
        y: The vector to solve.
    """

    # Solve Ls = y
    s = aspl.solve_triangular(k_chol, y, lower=True)

    # Solve Lt b = s
    b = aspl.solve_triangular(k_chol.T, s)

    return b


def fit_gp_regression(X, y, X_predict, kernel_fun, obs_var):

    k_xstar_x = kernel_fun(X_predict, X)
    k_xx = kernel_fun(X, X)
    obs_mat = anp.diag(obs_var * anp.ones(X.shape[0]))
    k_xstar_xstar = kernel_fun(X_predict, X_predict)

    k_chol = anp.linalg.cholesky(k_xx + obs_mat)

    pred_mean = k_xstar_x @ solve_via_cholesky(k_chol, y)
    pred_cov = k_xstar_xstar - k_xstar_x @ solve_via_cholesky(
        k_chol, k_xstar_x.T)

    return pred_mean, pred_cov


def gp_regression_marginal_likelihood(X, y, kernel_fun, obs_var):
    # I am omitting the n term. Should I include it? It's constant.

    obs_var_full = anp.diag(obs_var * anp.ones(X.shape[0]))
    k_xx = kernel_fun(X, X)

    k_chol = anp.linalg.cholesky(k_xx + obs_var_full)

    det_term = -0.5 * anp.linalg.slogdet(k_xx + obs_var_full)[1]
    data_term = -0.5 * y @ solve_via_cholesky(k_chol, y)

    return det_term + data_term
