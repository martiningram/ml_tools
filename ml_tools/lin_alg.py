import autograd.numpy as np
import scipy.sparse.linalg as spl
from scipy.linalg import cho_solve, cho_factor, solve_triangular


def cholesky_inverse(matrix):

    return cho_solve(cho_factor(matrix, lower=True), np.eye(matrix.shape[0]))


def compute_sparse_inverse(sparse_matrix):

    lu = spl.splu(sparse_matrix)
    inverse = lu.solve(np.eye(sparse_matrix.shape[0]))

    return inverse


def pos_def_mat_from_vector(vec, target_size):
    # I can't differentiate this with autograd. Sad days. Maybe there's some
    # kind of workaround?

    L = np.zeros((target_size, target_size))
    L[np.tril_indices(target_size)] = vec

    return np.matmul(L, L.T)


def pos_def_mat_from_mat(mat):

    return np.matmul(mat, mat.T) + np.eye(mat.shape[0]) * 1e-6


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
