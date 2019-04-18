import autograd.numpy as np
from autograd import make_jvp


def forward_grad_vector(fun, arg_no, n_derivs, *args):
    # Example call:
    # forward_grad_vector(
    # ard_rbf_kernel_efficient, 3, lscales.shape[0], X, X, var, lscales)
    # TODO: Maybe make syntax agree even more with the current autograd.

    grad = make_jvp(fun, arg_no)
    all_indices = np.eye(n_derivs)

    all_grads = list()

    for cur_index in all_indices:

        all_grads.append(grad(*args)(cur_index)[1])

    return np.stack(all_grads, -1)


def multivariate_normal_zero_mean_from_inv(x, cov_inv):

    n = cov_inv.shape[0]
    sign, logdet = np.linalg.slogdet(cov_inv)
    logdet = sign * logdet
    det_term = 0.5 * (logdet - n * np.log(2 * np.pi))

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


def logdet_via_cholesky(mat):

    chol = np.linalg.cholesky(mat)
    logdet = 2 * np.sum(np.log(np.diag(chol)))
    return logdet
