import jax.numpy as np
from jax import jit, jacfwd, jacrev, lax


def hessian(fun, argnum=0):
    return jit(jacfwd(jacrev(fun, argnum), argnum))


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
