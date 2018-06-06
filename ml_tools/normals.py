import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import inv


class MultivariateNormal(object):

    def __init__(self, m, v_inv):
        """
        Initialises a Multivariate normal.
        Args:
            m (np.array): The mean vector.
            v_inv (np.array): The inverse of the covariance matrix.
        """

        # Do some size checking
        assert(v_inv.shape[0] == m.shape[0] and v_inv.shape[1] == m.shape[0])

        self.m = m
        self.v_inv = v_inv

    def multiply(self, m2, v_inv2):
        """
        Multiplies the multivariate normal with another multivariate normal,
        and returns the result.
        """

        m1, v_inv1 = self.m, self.v_inv
        summed_inv = v_inv1 + v_inv2
        summed = inv(summed_inv)
        pt1 = np.matmul(np.matmul(summed, v_inv1), m1)
        pt2 = np.matmul(np.matmul(summed, v_inv2), m2)
        new_m = pt1 + pt2
        new_v_inv = summed_inv
        return MultivariateNormal(new_m, new_v_inv)

    def divide(self, m2, v_inv2):
        """
        Divides the multivariate normal by another multivariate normal, and
        returns the result.
        """

        m1, v_inv1 = self.m, self.v_inv
        subtracted_inv = v_inv1 - v_inv2
        subtracted = inv(subtracted_inv)
        pt1 = np.matmul(np.matmul(subtracted, v_inv1), m1)
        pt2 = np.matmul(np.matmul(subtracted, v_inv2), m2)
        new_m = pt1 - pt2
        new_v_inv = subtracted_inv
        return MultivariateNormal(new_m, new_v_inv)

    def weighted_sum(self, weights):
        """
        Computes mean and variance of a weighted sum of the mvn r.v.
        Args:
            weights (np.array): A vector of weights to give the elements.
        Returns:
            Tuple[float, float]: The mean and variance of the weighted sum.
        """

        mean_summed_theta = np.dot(self.m, weights)

        cur_cov = inv(self.v_inv)
        outer_x = np.outer(weights, weights)
        multiplied = cur_cov * outer_x
        weighted_sum = np.sum(multiplied)

        return mean_summed_theta, weighted_sum

    def __str__(self):

        return ('Normal distribution with mean {} and precision'
                ' {}.'.format(self.m, self.v_inv))

    def get_marginal_var(self):
        """
        Returns the marginal variance of the variables.
        """

        cov_matrix = inv(self.v_inv)
        marginals = np.diagonal(cov_matrix)
        return marginals

    def summarise(self):
        """
        Returns marginal summaries [intervals] of the variables as a pandas
        DataFrame.
        """

        marginals = self.get_marginal_var()

        results = list()

        for cur_m, cur_var in zip(self.m, marginals):

            cur_std = np.sqrt(cur_var)
            cur_norm = norm(cur_m, cur_std)

            quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
            values = cur_norm.ppf(quantiles)

            results.append(values)

        return pd.DataFrame(results, columns=[str(np.round(x, 2)) for x in
                                              np.array(quantiles)*100])


class DiagonalNormal(object):

    def __init__(self, m, v):
        """Instantiates a new diagonal multivariate normal.
        Args:
            m (np.array): The mean vector.
            v (np.array): The vector of variances.
        """

        assert(m.shape[0] == v.shape[0])
        assert(np.prod(v.shape) == m.shape[0])

        self.m = m
        self.v = v

    def multiply(self, m2, v2):
        m1, v1 = self.m, self.v
        new_v = 1. / (1. / v1 + 1. / v2)
        new_m = new_v * (m1 / v1 + m2 / v2)
        return DiagonalNormal(new_m, new_v)

    def divide(self, m2, v2):
        m1, v1 = self.m, self.v
        new_v = 1. / (1. / v1 - 1. / v2)
        new_m = new_v * (m1 / v1 - m2 / v2)
        return DiagonalNormal(new_m, new_v)

    def __str__(self):
        return 'Normal distribution with mean {} and variance {}.'.format(
            self.m, self.v)

    def plot(self, ax=None):

        if ax is None:
            f, ax = plt.subplots(1, 1)

        # Plot marginals
        for i in range(self.m.shape[0]):

            cur_m = self.m[i]
            cur_std = np.sqrt(self.v[i])

            lower = cur_m - 4 * cur_std
            upper = cur_m + 4 * cur_std

            cur_pts = np.linspace(lower, upper, 100)

            ax.plot(cur_pts, norm.pdf(cur_pts, cur_m, cur_std))

        return ax
