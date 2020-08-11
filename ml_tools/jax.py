import jax.numpy as np
from jax import lax, hessian, jacobian
from typing import Callable, Tuple
from scipy.optimize import minimize
from jax.ops.scatter import index_update
from jax.scipy.special import expit
import numpy as onp


# def hessian(fun, argnum=0):
#     return jit(jacfwd(jacrev(fun, argnum), argnum))


def multivariate_normal_inv_normaliser(cov_inv):

    n = cov_inv.shape[0]
    chol = np.linalg.cholesky(cov_inv)
    logdet = 2 * np.sum(np.log(np.diag(chol)))
    det_term = 0.5 * (logdet - n * np.log(2 * np.pi))
    return det_term


def multivariate_normal_zero_mean_from_inv(x, cov_inv):

    det_term = multivariate_normal_inv_normaliser(cov_inv)
    logpdf = det_term - 0.5 * x @ cov_inv @ x
    return logpdf


def multivariate_normal_logpdf(x, cov):

    sign, logdet = np.linalg.slogdet(np.pi * 2 * cov)
    logdet = sign * logdet
    det_term = -0.5 * logdet

    # TODO: Could be improved by using some kind of Cholesky here rather than
    # inverse -- or even a solve.
    total_prior = det_term - 0.5 * x @ np.linalg.inv(cov) @ x

    return total_prior


def newton_optimize(start_f, fun, jac, hess, solve_fun=np.linalg.solve, tolerance=1e-5):

    # TODO: Consider adding a maxiter
    def body(val):

        f, difference = val

        cur_hess = hess(f)
        cur_jac = jac(f)
        sol = solve_fun(cur_hess, cur_jac)
        new_f = f - sol

        difference = np.linalg.norm(f - new_f)

        return (new_f, difference)

    init_val = (start_f, 1.0)

    result = lax.while_loop(lambda data: data[1] > tolerance, body, init_val)

    return result[0]


def fit_laplace_approximation(
    neg_log_posterior_fun: Callable[[np.ndarray], float],
    start_val: np.ndarray,
    optimization_method: str = "Newton-CG",
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Fits a Laplace approximation to the posterior.
    Args:
        neg_log_posterior_fun: Returns the [unnormalized] negative log
            posterior density for a vector of parameters.
        start_val: The starting point for finding the mode.
        optimization_method: The method to use to find the mode. This will be
            fed to scipy.optimize.minimize, so it has to be one of its
            supported methods. Defaults to "Newton-CG".
    Returns:
        A tuple containing three entries; mean, covariance and a boolean flag
        indicating whether the optimization succeeded.
    """

    jac = jacobian(neg_log_posterior_fun)
    hess = hessian(neg_log_posterior_fun)

    result = minimize(
        neg_log_posterior_fun, start_val, jac=jac, hess=hess, method=optimization_method
    )

    covariance_approx = np.linalg.inv(hess(result.x))
    mean_approx = result.x

    return mean_approx, covariance_approx, result.success


def lo_tri_from_elements(elements, n):

    L = np.zeros((n, n))
    indices = np.tril_indices(L.shape[0])
    L = index_update(L, indices, elements)

    return L


def weighted_sum(mean, cov, weights):
    """
    Computes mean and variance of a weighted sum of the mvn r.v.
    Args:
        mean (np.array): The mean of the MVN.
        cov (np.array): The covariance of the MVN.
        weights (np.array): A vector of weights to give the elements.
    Returns:
        Tuple[float, float]: The mean and variance of the weighted sum.
    """

    mean_summed_theta = np.dot(mean, weights)

    outer_x = np.outer(weights, weights)
    multiplied = cov * outer_x
    weighted_sum = np.sum(multiplied)

    return mean_summed_theta, weighted_sum


def logistic_normal_integral_approx(mu, var):
    """
    Approximates the logistic normal integral, E[logit^{-1}(X)], where
    X ~ N(mu, var).
    """

    gamma = np.sqrt(1 + (np.pi * (var / 8)))

    return expit(mu / gamma)


def pos_def_mat_from_tri_elts(elts, mat_size, jitter=1e-6):

    cov_mat = lo_tri_from_elements(elts, mat_size)
    cov_mat = cov_mat @ cov_mat.T

    cov_mat = cov_mat + np.eye(mat_size) * jitter

    return cov_mat


def vector_from_pos_def_mat(pos_def_mat, jitter=0):

    # Subtract jitter
    pos_def_mat -= np.eye(pos_def_mat.shape[0]) * jitter
    L = np.linalg.cholesky(pos_def_mat)
    elts = np.tril_indices_from(L)

    return L[elts]


def convert_decorator(fun, verbose=True):
    def result(x):

        value, grad = fun(x)

        if verbose:
            print(value, np.linalg.norm(grad))

        return (
            onp.array(value).astype(onp.float64),
            onp.array(grad).astype(onp.float64),
        )

    return result
