import numpy as np


def matrix_argsort(mat):
    # Returns indices of the sorted elements in mat, starting with the smallest
    # Return shape is N x 2, where N is the total number of elements in mat

    return np.dstack(np.unravel_index(np.argsort(mat.ravel()), mat.shape))[0]
