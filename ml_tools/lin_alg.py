import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.linalg import cho_solve, cho_factor

def cholesky_inverse(matrix):

    return cho_solve(cho_factor(matrix, lower=True), np.eye(matrix.shape[0]))


def compute_sparse_inverse(sparse_matrix):

    lu = spl.splu(sparse_matrix)
    inverse = lu.solve(np.eye(sparse_matrix.shape[0]))

    return inverse
