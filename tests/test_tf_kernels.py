import numpy as np
import tensorflow as tf
from ml_tools.tf_kernels import (
    ard_rbf_kernel, ard_rbf_kernel_old, compute_diag_weighted_square_distance,
    compute_weighted_square_distances, additive_rbf_kernel) # NOQA
import ml_tools.kernels as np_kernels

jitter = 1e-5


def get_fake_data():

    n_cov = 4
    n_data_1 = 100
    n_data_2 = 50

    x1 = tf.constant(np.random.randn(n_data_1, n_cov))
    x2 = tf.constant(np.random.randn(n_data_2, n_cov))
    rho = tf.constant(np.random.uniform(2, 4, size=n_cov))
    alpha = tf.constant(2., dtype=tf.float64)

    return x1, x2, rho, alpha


def test_new_ard_kernel_against_old():

    x1, x2, rho, alpha = get_fake_data()

    old_result = ard_rbf_kernel_old(x1, x2, rho, alpha, jitter=jitter)
    new_result = ard_rbf_kernel(x1, x2, rho, alpha, jitter=jitter)

    assert np.allclose(old_result.numpy(), new_result.numpy())


def test_diag_weighted_square_distance():

    x1, x2, rho, _ = get_fake_data()

    diag_square_dist = compute_diag_weighted_square_distance(x1, x2, rho)
    full_square_dist = compute_weighted_square_distances(x1, x2, rho)

    diag_full = tf.linalg.diag_part(full_square_dist)

    assert np.allclose(diag_square_dist.numpy(), diag_full.numpy())


def test_ard_kernel_diag_only():

    x1, x2, rho, alpha = get_fake_data()

    full_result = ard_rbf_kernel(x1, x2, rho, alpha, jitter=jitter,
                                 diag_only=False)

    diag_result = ard_rbf_kernel(x1, x2, rho, alpha, jitter=jitter,
                                 diag_only=True)

    diag_full = tf.linalg.diag_part(full_result)

    assert np.allclose(diag_full.numpy(), diag_result.numpy())


def test_additive_rbf_kernel_against_numpy():

    x1, x2, rho, alpha = get_fake_data()

    np_result = np_kernels.additive_rbf_kernel(
        x1.numpy(), x2.numpy(), rho.numpy(), alpha.numpy(), jitter=jitter)

    tf_result = additive_rbf_kernel(x1, x2, rho, alpha, jitter=jitter)

    assert np.allclose(tf_result.numpy(), np_result)
