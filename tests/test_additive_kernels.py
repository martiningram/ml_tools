import numpy as np
from ml_tools.kernels import ard_rbf_kernel_efficient
from ml_tools.additive_kernels import newton_girard_combination


def test_newton_girard_combination():

    X = np.random.randn(20, 4)
    N = 4

    kerns = np.array(
        [
            ard_rbf_kernel_efficient(
                cur_x.reshape(-1, 1),
                cur_x.reshape(-1, 1),
                1.0,
                np.random.uniform(2.0, 4.0, size=(1,)),
            )
            for cur_x in X.T
        ]
    )

    es = newton_girard_combination(kerns, N)

    zs = kerns

    e1 = es[0]

    e2 = 0.5 * (e1 * np.sum(zs, axis=0) - np.sum(zs ** 2, axis=0))
    e3 = (
        1.0
        / 3.0
        * (
            e2 * np.sum(zs, axis=0)
            - e1 * np.sum(zs ** 2, axis=0)
            + np.sum(zs ** 3, axis=0)
        )
    )

    assert np.allclose(e3, es[2])
