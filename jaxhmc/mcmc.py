from functools import partial

import flax.nnx as nnx
import flax.struct as struct
import jax
import jax.numpy as jnp

from jaxhmc.integrators import leapfrog
from jaxhmc.potentials import Potential


def sample_gaussian_from_precision(batch_size: int, dim: int, precm_L: jax.Array, key: jax.Array):
    # Then we sample from a standard normal distribution and transform it
    key, subkey = jax.random.split(key, 2)
    z = jax.random.normal(subkey, shape=(batch_size, dim))

    # We have x = L^{-T}z, so L^{T}x = z
    x = jnp.linalg.solve(precm_L.T[None, ...], z[..., None])
    x = x.squeeze(-1)
    return x, key


@struct.dataclass
class HMCConfig:
    initial_step_size: int = struct.field(pytree_node=False)  # We update with dual averaging
    max_path_len: int = struct.field(pytree_node=False)  # We use jittering

    warmup_steps: int = struct.field(pytree_node=False)
    iterations: int = struct.field(pytree_node=False)

    initial_precm: jax.Array
    key: jax.Array


def mh(
    carry: tuple[jax.Array, jax.Array, jax.Array],
    _,
    potential: Potential,
    potential_grad: nnx.Module,
    precm: jax.Array,
    precm_L: jax.Array,
    step_size: int,
    steps: int,
    batch_size: int,
):
    q, key = carry

    p, key = sample_gaussian_from_precision(
        batch_size,
        potential.dim,
        precm_L,
        key,
    )

    # We first simulate the dynamics
    p_new, q_new = leapfrog(
        p,
        q,
        precm,
        potential_grad,
        steps=steps,
        step_size=step_size,
    )

    # Compute the Hamiltonian
    H = potential(q) + jnp.einsum("bi, ij, bj->b", p, precm, p)
    H_new = potential(q_new) + jnp.einsum("bi, ij, bj->b", p_new, precm, p_new)

    # Compute the MH correction step
    alpha = jnp.minimum(1.0, jnp.exp(H - H_new))
    key, subkey = jax.random.split(key, 2)
    b = jax.random.bernoulli(subkey, alpha)
    b = b[..., None]

    # We only make the move if b = 1
    p_next = b * p_new + (1 - b) * p
    q_next = b * q_new + (1 - b) * q
    return (q_next, key), (p_next, q_next)


def hmc(potential: Potential, initial_position: jax.Array, config: HMCConfig):
    # We could sample all momentum variables from the outset - not a concern
    # unless we are operating at a huge dimensional space with multiple chains
    # This should be assessed.

    # Each step consists of (i) simulating the Hamiltonian dynamics
    # and (ii) Metropolis correction. We also update the
    # step size with Robbins-Monro's dual averaging algorithm.

    pot_grad_vmap = jax.vmap(jax.grad(potential), in_axes=0)
    pot_vmap = jax.vmap(potential, in_axes=0)

    step_size = config.initial_step_size

    # We first compute the Cholesky decomposition of the precision matrix
    precm_L = jnp.linalg.cholesky(config.initial_precm)

    _, (p, q) = jax.lax.scan(
        f=partial(
            mh,
            potential=pot_vmap,
            potential_grad=pot_grad_vmap,
            step_size=step_size,
            precm=config.initial_precm,
            precm_L=precm_L,
            steps=config.max_path_len // step_size,
            batch_size=initial_position.shape[0],
        ),
        init=(initial_position, config.key),
        length=config.iterations,
    )

    return p, q
