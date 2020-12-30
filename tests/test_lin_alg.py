import numpy as np
import jax.numpy as jnp
from ml_tools.lin_alg import (
    log_determinant_diag_plus_low_rank_eigendecomp,
    triple_matmul_with_diagonal_mat,
    generate_random_pos_def,
)
import ml_tools.lin_alg as lin_alg
from scipy.sparse.linalg import cg as scipy_cg
from quspin.tools.lanczos import lanczos_full


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


# def test_cg():
#
#     N = 100
#
#     A = generate_random_pos_def(N) + np.eye(100)
#     b = np.ones(N)
#
#     matvec_fun = lambda x: A @ x
#
#     custom_res, custom_info = lin_alg.cg(matvec_fun, b, tol=1e-12)
#     scipy_res, scipy_info = scipy_cg(A, b, tol=1e-8)
#
#     assert np.allclose(custom_res, scipy_res)


def test_lanczos():

    from collections import namedtuple

    N = 100
    A = generate_random_pos_def(N)
    b = np.random.randn(N)

    m = 30

    # Wrap for the quspin library
    Matvec = namedtuple("Matvec", "dot,dtype")
    wrapped_matvec = Matvec(dot=lambda x: A @ x, dtype=b.dtype)

    E, V, Q_T = lanczos_full(wrapped_matvec, b, m)

    # Compute with custom routine
    lanczos_res = lin_alg.lanczos(lambda x: A @ x, b, m=m)

    assert np.allclose(E, lanczos_res["evs"])
    assert np.allclose(np.abs(V), np.abs(lanczos_res["evecs"]))
    assert np.allclose(Q_T, lanczos_res["Q"])
