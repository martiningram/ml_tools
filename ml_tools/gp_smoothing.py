import autograd.numpy as np
from .autograd_kernels import ard_rbf_kernel_efficient
from scipy.optimize import minimize
from autograd import value_and_grad
from functools import partial
import autograd.scipy.linalg as spl
from autograd.scipy.stats import gamma


def solve_via_cholesky(k_chol, y):
    """Solves a positive definite linear system via a Cholesky decomposition.

    Args:
        k_chol: The Cholesky factor of the matrix to solve. A lower triangular
            matrix, perhaps more commonly known as L.
        y: The vector to solve.
    """

    # Solve Ls = y
    s = spl.solve_triangular(k_chol, y, lower=True)

    # Solve Lt b = s
    b = spl.solve_triangular(k_chol.T, s)

    return b


def fit_gp_regression(X, y, X_predict, kernel_fun, obs_var):

    k_xstar_x = kernel_fun(X_predict, X)
    k_xx = kernel_fun(X, X)
    obs_mat = np.diag(obs_var * np.ones(X.shape[0]))
    k_xstar_xstar = kernel_fun(X_predict, X_predict)

    k_chol = np.linalg.cholesky(k_xx + obs_mat)

    pred_mean = k_xstar_x @ solve_via_cholesky(k_chol, y)
    pred_cov = k_xstar_xstar - k_xstar_x @ solve_via_cholesky(
        k_chol, k_xstar_x.T)

    return pred_mean, pred_cov


def gp_regression_marginal_likelihood(X, y, kernel_fun, obs_var,
                                      solve_fun=np.linalg.solve):
    # I am omitting the n term. Should I include it? It's constant.

    obs_var_full = np.diag(obs_var * np.ones(X.shape[0]))
    k_xx = kernel_fun(X, X)

    k_chol = np.linalg.cholesky(k_xx + obs_var_full)

    det_term = -0.5 * np.linalg.slogdet(k_xx + obs_var_full)[1]
    data_term = -0.5 * y @ solve_via_cholesky(k_chol, y)

    return det_term + data_term


def map_smooth_data_1d(X, y, X_pred, kernel_fun=ard_rbf_kernel_efficient,
                       mean_centre_y=True, prior_k=3, prior_theta=1/3):

    y_mean = y.mean()

    if mean_centre_y:

        y = y - y_mean

    def to_optimize(theta):

        alpha, lscale, obs_var = theta**2

        cur_k = partial(kernel_fun, lengthscales=np.array([lscale]),
                        alpha=alpha)

        marg_lik = gp_regression_marginal_likelihood(X, y, cur_k, obs_var)

        prior_contrib = (prior_k - 1) * np.log(lscale) - lscale / prior_theta

        return -marg_lik - prior_contrib

    to_opt_with_grad = value_and_grad(to_optimize)

    opt_result = minimize(to_opt_with_grad, [1., 1., 1.], jac=True,
                          method='BFGS', tol=1e-3)

    assert opt_result.success

    alpha, lscale, obs_var = opt_result.x**2

    final_k = partial(kernel_fun, lengthscales=np.array([lscale]),
                      alpha=alpha)

    # Predict
    pred_mean, pred_cov = fit_gp_regression(X, y, X_pred, final_k, obs_var)

    if mean_centre_y:

        pred_mean += y_mean

    return pred_mean, np.diag(pred_cov)
