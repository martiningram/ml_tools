import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.linalg import inv, cho_factor, cho_solve
from scipy.special import expit, logsumexp


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


def covar_to_corr(covar_mat):
    # Turns a covariance matrix into a correlation matrix

    marg_var = np.diag(covar_mat)
    marg_sd = np.sqrt(marg_var)
    inv_marg = np.diag(1. / marg_sd)

    return np.dot(np.dot(inv_marg, covar_mat), inv_marg)


def conditional_mean_and_cov(mu_x, mu_y, y, A, C, B):
    """
    Calculates the conditional mean and covariance of x given y.

    Here, x and y are defined as in Quinonero & Candela's Sparse GP paper
    (Appendix):

    [x, y] ~ N([mu_x, mu_y], [[A, C], [C^T, B]])

    and we calculate

    x | y.
    """

    # Calculate the conditional mean
    difference = y - mu_y
    b_chol = cho_factor(B)
    conditional_mean = mu_x + np.matmul(C, cho_solve(b_chol, difference))
    conditional_cov = A - np.matmul(C, cho_solve(b_chol, C.T))

    return {'mean': conditional_mean, 'cov': conditional_cov}


def conjugate_update_univariate(prior_mu, prior_var, lik_mu, lik_var):
    """
    Calculates the posterior distribution of a random variable theta for which

    theta ~ N(prior_mu, prior_var)
    y|theta ~ N(lik_mu, lik_var)

    In this case, theta|y is Normal.

    Args:
        prior_mu: Mean of the normal prior on theta.
        prior_var: Variance of the normal prior on theta.
        lik_mu: Mean of the likelihood given theta.
        lik_var: Variance of the likelihood given theta.

    Returns:
        Tuple[float,float]: Mean and variance of the posterior distribution of
        theta.
    """

    prior_prec = 1. / prior_var
    lik_prec = 1. / lik_var

    new_prec = (prior_prec + lik_prec)

    # TODO: Check this!
    bracket_term = prior_mu * prior_prec + lik_mu * lik_prec

    new_mean = (1. / new_prec) * bracket_term
    new_var = 1. / new_prec

    return new_mean, new_var


def linear_regression_online_update(m_km1, P_km1, H, m_obs, var_obs):
    # m_km1: Prior mean
    # P_km1: Prior cov
    # H: Link from latent to observed
    # m_obs: Mean of observation
    # var_obs: Variance of observation
    # Returns mean, cov, and _negative_ log marg lik.

    # We need to work with matrices here or the maths will be wrong
    assert all([len(x.shape) == 2 for x in [m_km1, H]]), \
        'm_km1 and H must have two dimensions each!'

    v_k = m_obs - H @ m_km1

    S_k = H @ P_km1 @ H.T + var_obs
    K_k = P_km1 @ H.T * (1 / S_k)
    m_k = m_km1 + K_k * v_k
    P_k = P_km1 - (K_k * S_k) @ K_k.T

    # Calculate the log marginal likelihood here too
    sign, logdet = np.linalg.slogdet(2 * np.pi * S_k)
    logdet = sign * logdet

    # Second part
    quadratic_term = 0.5 * v_k.T @ np.linalg.solve(S_k, v_k)
    energy_contrib = 0.5 * logdet + quadratic_term

    return m_k, P_k, np.squeeze(energy_contrib)


def weighted_sum(mean, cov, weights):
    """
    Computes mean and variance of a weighted sum of the mvn r.v.
    Args:
        mean (np.array): The mean of the MVN.
        cov (np.array): The covariance of the MVN.
        weights (np.array): A vector of weights to give the elements.
    Returns:
        Tuple[float, float]: The mean and variance of the weighted sum.
    """

    mean_summed_theta = np.dot(mean, weights)

    outer_x = np.outer(weights, weights)
    multiplied = cov * outer_x
    weighted_sum = np.sum(multiplied)

    return mean_summed_theta, weighted_sum


def logistic_normal_integral_approx(mu, var):
    """
    Approximates the logistic normal integral, E[logit^{-1}(X)], where
    X ~ N(mu, var).
    """

    gamma = np.sqrt(1 + (np.pi * (var / 8)))

    return expit(mu / gamma)


def corr_to_covar(variances, corr_mat):

    diag_cov = np.diag(variances)

    return np.sqrt(diag_cov) @ corr_mat @ np.sqrt(diag_cov)


def mvn_kl(mu_0, sigma_0, mu_1, sigma_1):

    logdet_sigma_1 = np.linalg.slogdet(sigma_1)[1]
    logdet_sigma_0 = np.linalg.slogdet(sigma_0)[1]
    term_1 = 0.5 * (logdet_sigma_1 - logdet_sigma_0)

    # I wonder if there's a more efficient way?
    mu_outer = np.outer(mu_0 - mu_1, mu_0 - mu_1)
    inside_term = mu_outer + sigma_0 - sigma_1
    solved = np.linalg.solve(sigma_1, inside_term)
    term_2 = 0.5 * np.trace(solved)

    return term_1 + term_2


def calculate_log_joint_bernoulli_likelihood(
        latent_prob_samples: np.ndarray, outcomes: np.ndarray,
        link: str = 'probit') -> float:
    # latent_prob_samples is n_samples x n_outcomes array of probabilities on
    # the probit scale
    # outcomes is (n_outcomes,) array of binary outcomes (1 and 0)
    assert(latent_prob_samples.shape[1] == outcomes.shape[0])

    # Make sure broadcasting is unambiguous
    assert(latent_prob_samples.shape[0] != outcomes.shape[0])

    n_samples = latent_prob_samples.shape[0]

    # Get log likelihood for each draw

    assert link in ['logit', 'probit'], \
        'Only logit and probit links supported!'

    if link == 'probit':
        individual_liks = np.sum(
            outcomes * norm.logcdf(latent_prob_samples)
            + (1 - outcomes) * norm.logcdf(-latent_prob_samples),
            axis=1)
    else:
        individual_liks = np.sum(
            outcomes * np.log(expit(latent_prob_samples))
            + (1 - outcomes) * np.log(1 - expit(latent_prob_samples)),
            axis=1)

    # Compute the Monte Carlo expectation
    return logsumexp(individual_liks - np.log(n_samples))
