import numpy as np
from ml_tools.normals import (moments_of_linear_combination_rvs,
                              generate_random_pos_def,
                              moments_of_linear_combination_rvs_batch,
                              moments_of_linear_combination_rvs_selected)


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


def test_moments_of_linear_combination_rvs_batch():

    n = 10
    n_l = 4
    n_out = 6

    means_1 = np.random.randn(n, n_l)
    means_2 = np.random.randn(n_out, n_l)

    cov_1 = np.stack([generate_random_pos_def(n_l) for _ in range(n)])
    cov_2 = np.stack([generate_random_pos_def(n_l) for _ in range(n_out)])

    batch_means, batch_vars = moments_of_linear_combination_rvs_batch(
        means_1, cov_1, means_2, cov_2)

    # Do a loop version:
    vars = np.zeros((n, n_out))
    means = np.zeros((n, n_out))

    for i in range(n):
        for j in range(n_out):

            cur_mean_1 = means_1[i]
            cur_mean_2 = means_2[j]

            cur_cov_1 = cov_1[i]
            cur_cov_2 = cov_2[j]

            cur_sum_mean, cur_sum_cov = \
                moments_of_linear_combination_rvs(cur_mean_1, cur_cov_1,
                                                  cur_mean_2, cur_cov_2)

            means[i, j] = cur_sum_mean
            vars[i, j] = cur_sum_cov

    assert np.allclose(batch_means, means)
    assert np.allclose(batch_vars, vars)


def test_moments_of_linear_combination_rvs_selected():

    n = 10
    n_l = 4
    n_out = 6

    to_select = np.random.choice(n_out, size=n, replace=True)

    means_1 = np.random.randn(n, n_l)
    means_2 = np.random.randn(n_out, n_l)

    cov_1 = np.stack([generate_random_pos_def(n_l) for _ in range(n)])
    cov_2 = np.stack([generate_random_pos_def(n_l) for _ in range(n_out)])

    # First compute all of them
    batch_means, batch_vars = moments_of_linear_combination_rvs_batch(
        means_1, cov_1, means_2, cov_2)

    # This produces:
    # [n x n_out]
    # for both of them. I'm only interested in the "to_select" column
    # of this.
    relevant_means_2 = means_2[to_select]
    relevant_covs_2 = cov_2[to_select]

    means_subset, vars_subset = moments_of_linear_combination_rvs_selected(
        means_1, cov_1, relevant_means_2, relevant_covs_2)

    assert np.allclose(means_subset, batch_means[np.arange(n), to_select])
    assert np.allclose(vars_subset, batch_vars[np.arange(n), to_select])
