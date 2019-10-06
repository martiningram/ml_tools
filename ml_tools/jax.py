import jax.numpy as np
from jax import lax, hessian, jacobian
from typing import Callable, Tuple
from scipy.optimize import minimize
from jax.ops.scatter import index_update


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


def newton_optimize(start_f, fun, jac, hess, solve_fun=np.linalg.solve,
                    tolerance=1e-5):

    # TODO: Consider adding a maxiter
    def body(val):

        f, difference = val

        cur_hess = hess(f)
        cur_jac = jac(f)
        sol = solve_fun(cur_hess, cur_jac)
        new_f = f - sol

        difference = np.linalg.norm(f - new_f)

        return (new_f, difference)

    init_val = (start_f, 1.)

    result = lax.while_loop(lambda data: data[1] > tolerance, body, init_val)

    return result[0]


def fit_laplace_approximation(neg_log_posterior_fun: Callable[[np.ndarray],
                                                              float],
                              start_val: np.ndarray,
                              optimization_method: str = 'Newton-CG') \
        -> Tuple[np.ndarray, np.ndarray, bool]:
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

    result = minimize(neg_log_posterior_fun, start_val, jac=jac, hess=hess,
                      method=optimization_method)

    covariance_approx = np.linalg.inv(hess(result.x))
    mean_approx = result.x

    return mean_approx, covariance_approx, result.success


def lo_tri_from_elements(elements, n):

    L = np.zeros((n, n))
    indices = np.tril_indices(L.shape[0])
    L = index_update(L, indices, elements)

    return L
