import jax.numpy as np
from jax import jit, jacfwd, jacrev


def hessian(fun, argnum=0):
    return jit(jacfwd(jacrev(fun, argnum), argnum))


def multivariate_normal_zero_mean_from_inv(x, cov_inv):

    n = cov_inv.shape[0]

    chol = np.linalg.cholesky(cov_inv)
    logdet = 2 * np.sum(np.log(np.diag(chol)))

    # sign, logdet = np.linalg.slogdet(cov_inv)
    # logdet = sign * logdet
    det_term = 0.5 * (logdet - n * np.log(2 * np.pi))

    logpdf = det_term - 0.5 * x @ cov_inv @ x
    return logpdf
