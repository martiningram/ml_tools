import numpy as np
import jax.numpy as jnp
from ml_tools.lin_alg import (
    log_determinant_diag_plus_low_rank_eigendecomp,
    triple_matmul_with_diagonal_mat,
)


def test_triple_matmul_with_diagonal_mat():

    A = np.random.randn(3, 5)
    B_elts = np.random.randn(5)
    C = np.random.randn(5, 7)

    full_res = A @ jnp.diag(B_elts) @ C
    other_res = triple_matmul_with_diagonal_mat(A, B_elts, C)

    assert np.allclose(full_res, other_res)


def test_log_determinant_diag_plus_low_rank_eigendecomp():

    A_elts = np.random.uniform(1, 5, size=5)
    U = np.random.randn(5, 3)
    W_elts = np.random.uniform(1, 5, size=3)

    full_mat = np.diag(A_elts) + U @ np.diag(W_elts) @ U.T
    full_sign, full_logdet = jnp.linalg.slogdet(full_mat)
    full_logdet = full_sign * full_logdet

    other_res = log_determinant_diag_plus_low_rank_eigendecomp(A_elts, U, W_elts)

    assert np.allclose(full_logdet, other_res)
