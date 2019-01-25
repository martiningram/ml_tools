import autograd.numpy as np


def ard_rbf_kernel_efficient(x1, x2, alpha, rho, jitter=1e-5):

    z1 = x1 / np.expand_dims(rho, axis=0)
    z2 = x2 / np.expand_dims(rho, axis=0)

    # Matrix part
    cross_contrib = -2 * np.matmul(z1, z2.T)

    # Other bits
    z1_sq = np.sum(z1**2, axis=1)
    z2_sq = np.sum(z2**2, axis=1)

    # Sum it all up
    combined = (np.expand_dims(z1_sq, axis=1) + cross_contrib +
                np.expand_dims(z2_sq, axis=0))

    kernel = alpha**2 * np.exp(-0.5 * combined)

    # Add the jitter
    diag_indices = np.diag_indices(np.min(kernel.shape[:2]))
    to_add = np.zeros_like(kernel)
    to_add[diag_indices] += jitter

    kernel = kernel + to_add

    return kernel
