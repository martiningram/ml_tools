# Test against library implementation
import numpy as np
from numpy_ml.neural_nets.optimizers import Adam
from ml_tools.adam import initialise_state, adam_step


def test_against_numpy_ml():

    lr = 0.001
    decay1 = 0.9
    decay2 = 0.999
    eps = 1e-7

    n_params = 100
    n_steps = 100

    adam_np = Adam(lr, decay1, decay2, eps)
    adam_state = initialise_state(n_params)

    theta_lib = np.random.randn(n_params)
    theta_cust = theta_lib.copy()

    for _ in range(n_steps):

        # Make up a bogus gradient
        cur_grad = np.random.randn(n_params)

        theta_cust, adam_state = adam_step(
            adam_state,
            theta_cust,
            cur_grad,
            step_size_fun=lambda _: lr,
            beta_1=decay1,
            beta_2=decay2,
            eps=eps,
        )

        theta_lib = adam_np.update(theta_lib, cur_grad, "bla")

    assert np.allclose(theta_lib, theta_cust)
