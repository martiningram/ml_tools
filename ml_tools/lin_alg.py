import autograd.numpy as np
import scipy.sparse.linalg as spl
from scipy.linalg import cho_solve, cho_factor, solve_triangular
import jax.numpy as jnp


def cholesky_inverse(matrix):

    return cho_solve(cho_factor(matrix, lower=True), np.eye(matrix.shape[0]))


def compute_sparse_inverse(sparse_matrix):

    lu = spl.splu(sparse_matrix)
    inverse = lu.solve(np.eye(sparse_matrix.shape[0]))

    return inverse


def pos_def_mat_from_vector(vec, target_size, jitter=0):

    L = np.zeros((target_size, target_size))
    L[np.tril_indices(target_size)] = vec

    return np.matmul(L, L.T) + np.eye(target_size) * jitter


def vector_from_pos_def_mat(pos_def_mat, jitter=0):

    # Subtract jitter
    pos_def_mat -= np.eye(pos_def_mat.shape[0]) * jitter
    L = np.linalg.cholesky(pos_def_mat)
    elts = np.tril_indices_from(L)

    return L[elts]


def num_triangular_elts(mat_size, include_diagonal=True):

    if include_diagonal:
        return int(mat_size * (mat_size + 1) / 2)
    else:
        return int(mat_size * (mat_size - 1) / 2)


def solve_via_cholesky(k_chol, y):
    """Solves a positive definite linear system via a Cholesky decomposition.

    Args:
        k_chol: The Cholesky factor of the matrix to solve. A lower triangular
            matrix, perhaps more commonly known as L.
        y: The vector to solve.
    """

    # Solve Ls = y
    s = solve_triangular(k_chol, y, lower=True)

    # Solve Lt b = s
    b = solve_triangular(k_chol.T, s)

    return b


def generate_random_pos_def(n, jitter=10 ** -4):

    elements = np.random.randn(n, n)
    cov = elements @ elements.T + np.eye(n) * jitter

    return cov


def triple_matmul_with_diagonal_mat(A, B_elts, C):
    # B is diagonal.
    return jnp.einsum("ik,k,kj->ij", A, B_elts, C)


def log_determinant_diag_plus_low_rank_eigendecomp(A_elts, U, W_elts):
    # Computes the log of the determinant
    # |A + U W U^T|,
    # where A is diagonal, W is diagonal, and W's dimension is smaller than A's dimension.

    A_logdet = jnp.sum(jnp.log(A_elts))
    W_logdet = jnp.sum(jnp.log(W_elts))

    A_inv_elts = 1 / A_elts
    W_inv_elts = 1 / W_elts

    second_term_in_slogdet = triple_matmul_with_diagonal_mat(U.T, A_inv_elts, U)

    sign, magnitude = jnp.linalg.slogdet(jnp.diag(W_inv_elts) + second_term_in_slogdet)
    first_term = sign * magnitude

    return first_term + W_logdet + A_logdet
