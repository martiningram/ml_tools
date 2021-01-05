from numpyro.infer import MCMC, NUTS
from jax import jit
import jax.numpy as jnp
import jax
from ml_tools.flattening import flatten_and_summarise, reconstruct
import numpy as np
from jax import vmap, jit
import arviz as az

# TODO: Would be better not to have a dependency on jax_advi here!
from jax_advi.constraints import apply_constraints
from jax_advi.advi import _calculate_log_posterior
from functools import partial


def initialise_from_shapes(param_shape_dict, sd=0.1, n_chains=4):

    # Make placeholder
    init_theta = {x: np.empty(y) for x, y in param_shape_dict.items()}

    flat_placeholder, summary = flatten_and_summarise(**init_theta)

    # Make flat draws
    flat_init = np.random.randn(n_chains, flat_placeholder.shape[0])

    return flat_init, summary


def sample_numpyro_nuts(
    log_posterior_fun,
    flat_init_params,
    parameter_summary,
    constrain_fun_dict={},
    target_accept=0.8,
    draws=1000,
    tune=1000,
    chains=4,
    progress_bar=True,
    random_seed=10,
    chain_method="parallel",
):
    # Strongly inspired by:
    # https://github.com/pymc-devs/pymc3/blob/master/pymc3/sampling_jax.py#L116
    def _sample(current_state, seed):

        step_size = jnp.ones_like(flat_init_params)

        nuts_kernel = NUTS(
            potential_fn=lambda x: -log_posterior_fun(x),
            target_accept_prob=target_accept,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
        )

        numpyro = MCMC(
            nuts_kernel,
            num_warmup=tune,
            num_samples=draws,
            num_chains=chains,
            postprocess_fn=None,
            progress_bar=progress_bar,
            chain_method=chain_method,
        )

        numpyro.run(seed, init_params=current_state)
        samples = numpyro.get_samples(group_by_chain=True)
        return samples

    seed = jax.random.PRNGKey(random_seed)
    samples = _sample(flat_init_params, seed)

    # Presumably, the output shape should be:
    # n_chains, n_samples, n_params?
    # Yes.
    # Reshape this into a dict
    def reshape_single_chain(theta):
        fun_to_map = lambda x: apply_constraints(
            reconstruct(x, parameter_summary, jnp.reshape), constrain_fun_dict
        )[0]
        return vmap(fun_to_map)(theta)

    samples = vmap(reshape_single_chain)(samples)

    return az.from_dict(posterior=samples)


def sample_nuts(
    parameter_shape_dict,
    log_prior_fun,
    log_lik_fun,
    constrain_fun_dict,
    target_accept=0.8,
    draws=1000,
    tune=1000,
    chains=4,
    progress_bar=True,
    random_seed=10,
    chain_method="vectorized",
):

    flat_theta, summary = initialise_from_shapes(parameter_shape_dict, n_chains=chains)

    log_post_fun = jit(
        partial(
            _calculate_log_posterior,
            log_lik_fun=lik_fun,
            log_prior_fun=calculate_prior,
            constrain_fun_dict=constrain_fun_dict,
            summary=summary,
        )
    )

    sampling_result = sample_numpyro_nuts(
        log_post_fun,
        flat_theta,
        summary,
        theta_constraints,
        chains=chains,
        draws=draws,
        tune=tune,
        chain_method=chain_method,
    )

    return sampling_result
