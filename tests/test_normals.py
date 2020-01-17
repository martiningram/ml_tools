import numpy as np
from ml_tools.normals import (moments_of_linear_combination_rvs,
                              generate_random_pos_def)


def test_moments_of_linear_combination_rvs():

    np.random.seed(2)

    n_test = 10

    means_1 = np.random.randn(n_test)
    cov_1 = generate_random_pos_def(n_test)

    means_2 = np.random.randn(n_test)
    cov_2 = generate_random_pos_def(n_test)

    sum_mean, sum_var = moments_of_linear_combination_rvs(
        means_1, cov_1, means_2, cov_2)

    # Compare against simulation
    n_sim = int(1e7)

    draws_1 = np.random.multivariate_normal(means_1, cov_1, size=n_sim)
    draws_2 = np.random.multivariate_normal(means_2, cov_2, size=n_sim)

    sim_results = np.sum(draws_1 * draws_2, axis=1)

    sim_mean, sim_var = np.mean(sim_results), np.std(sim_results)**2

    # TODO: This is hardly a stringent test, and I set the tolerance more or
    # less arbitrarily. I'm not entirely sure what else to do.
    assert np.isclose(sum_mean, sim_mean, rtol=1e-2)
    assert np.isclose(sum_var, sim_var, rtol=1e-2)
