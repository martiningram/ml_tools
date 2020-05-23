import numpy as np
from ml_tools.kernels import ard_rbf_kernel_efficient, additive_rbf_kernel


def generate_data(n_1=10, n_2=10, n_c=4):

    n_1 = 10
    n_2 = 10
    n_c = 4
    lscales = np.random.uniform(2, 4, size=n_c)

    x1 = np.random.randn(n_1, n_c)
    x2 = np.random.randn(n_2, n_c)

    return n_1, n_2, n_c, lscales, x1, x2


def test_additive_rbf_kernel_against_ard_rbf():

    n_1, n_2, n_c, lscales, x1, x2 = generate_data()

    summed_ard = np.zeros((n_1, n_2))

    alpha = 4.

    for cur_c in range(n_c):

        cur_alpha = np.sqrt(alpha**2 / n_c)

        cur_x1 = x1[:, [cur_c]]
        cur_x2 = x2[:, [cur_c]]
        cur_lscale = lscales[[cur_c]]

        ard_result = ard_rbf_kernel_efficient(cur_x1, cur_x2, cur_alpha,
                                              cur_lscale, jitter=0.)

        summed_ard += ard_result

    new_version = additive_rbf_kernel(x1, x2, lscales, alpha, jitter=0.)

    np.allclose(new_version, summed_ard)


def test_additive_rbf_kernel_diag_only():

    n_1, n_2, n_c, lscales, x1, x2 = generate_data()

    alpha = 2.

    new_version_diag = additive_rbf_kernel(x1, x2, lscales, alpha, jitter=0.,
                                           diag_only=True)

    new_version = additive_rbf_kernel(x1, x2, lscales, alpha, jitter=0.)

    np.allclose(np.diag(new_version), new_version_diag)
