from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from jaxhmc.mcmc import HMCConfig, hmc, mh
from jaxhmc.potentials import Potential


@pytest.fixture
def potential():
    class Gaussian(Potential):
        # Standard Gaussian
        def __init__(self, dim):
            super().__init__(dim)

        def __call__(self, x):
            return jnp.sum(x**2)

    return Gaussian(dim=2)


@pytest.fixture
def vpotential(potential: Potential):
    return jax.vmap(nnx.grad(potential), in_axes=0)


@pytest.fixture
def key():
    return jax.random.key(42)


def test_mh_return_shape(potential: Potential, vpotential: nnx.Module, key: jax.Array):
    batch_size = 8
    q = jax.random.uniform(key, (batch_size, potential.dim))
    precm = jnp.eye(potential.dim)

    _, (p_next, q_next) = mh(
        (q, key),
        None,
        potential,
        vpotential,
        precm,
        step_size=0.2,
        steps=40,
        batch_size=batch_size,
    )

    assert p_next.shape == q_next.shape


def test_scan_mh_return_trajectory(potential: Potential, vpotential: nnx.Module, key: jax.Array):
    batch_size = 8
    q = jax.random.uniform(key, (batch_size, potential.dim))
    precm = jnp.eye(potential.dim)
    chain_length = 16

    _, (p_next, q_next) = jax.lax.scan(
        partial(
            mh,
            potential=potential,
            potential_grad=vpotential,
            precm=precm,
            step_size=0.2,
            steps=40,
            batch_size=batch_size,
        ),
        init=(q, key),
        length=chain_length,
    )

    assert p_next.shape == q_next.shape
    assert p_next.shape == (chain_length, batch_size, potential.dim)


def test_hmc_return_batched_chains(potential: Potential, key: jax.Array):
    batch_size = 8
    chain_length = 16
    q = jax.random.uniform(key, (batch_size, potential.dim))
    precm = jnp.eye(potential.dim)

    p, q = hmc(
        potential,
        q,
        HMCConfig(
            initial_step_size=0.2,
            max_path_len=2,
            warmup_steps=10,
            iterations=chain_length,
            initial_precm=precm,
            key=key,
        ),
    )

    assert p.shape == q.shape
    assert p.shape == (chain_length, batch_size, potential.dim)
