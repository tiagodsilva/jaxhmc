from functools import partial

import flax.nnx as nnx
import flax.struct as struct
import jax
import jax.numpy as jnp

from jaxhmc.integrators import leapfrog
from jaxhmc.potentials import Potential
from jaxhmc.tuning import NesterovConfig, NesterovState, nesterov_dual_averaging


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

    iterations: int = struct.field(pytree_node=False)

    initial_precm: jax.Array
    key: jax.Array


def mh(
    carry: tuple[jax.Array, jax.Array, NesterovState],
    _,
    potential: Potential,
    potential_grad: nnx.Module,
    precm: jax.Array,
    precm_L: jax.Array,
    steps: int,
    batch_size: int,
    nesterov_config: NesterovConfig,
):
    q, key, nesterov_state = carry

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
        step_size=nesterov_state.step_size,
    )

    # Compute the Hamiltonian
    H = potential(q) + jnp.einsum("bi, ij, bj->b", p, precm, p)
    H_new = potential(q_new) + jnp.einsum("bi, ij, bj->b", p_new, precm, p_new)

    # Compute the MH correction step
    alpha = jnp.minimum(1.0, jnp.exp(H - H_new))
    alpha = jnp.where(jnp.isnan(alpha), 0.0, alpha)

    key, subkey = jax.random.split(key, 2)
    b = jax.random.bernoulli(subkey, alpha)
    b = b[..., None]

    # We only make the move if b = 1
    p_next = b * p_new + (1 - b) * p
    q_next = b * q_new + (1 - b) * q

    nesterov_state = nesterov_dual_averaging(
        nesterov_state,
        jnp.mean(alpha),
        nesterov_config,
    )

    return (q_next, key, nesterov_state), (p_next, q_next)


def hmc(
    potential: Potential,
    initial_position: jax.Array,
    config: HMCConfig,
    tuning_steps: int = 1000,
):
    pot_grad_vmap = jax.vmap(jax.grad(potential), in_axes=0)
    pot_vmap = jax.vmap(potential, in_axes=0)

    step_size = config.initial_step_size

    # We first compute the Cholesky decomposition of the precision matrix
    precm_L = jnp.linalg.cholesky(config.initial_precm)

    nesterov_state = NesterovState(step_size=config.initial_step_size)
    nesterov_config = NesterovConfig(
        mu=jnp.log(10 * config.initial_step_size),
        tuning_steps=tuning_steps,
    )

    _, (p, q) = jax.lax.scan(
        f=partial(
            mh,
            potential=pot_vmap,
            potential_grad=pot_grad_vmap,
            precm=config.initial_precm,
            precm_L=precm_L,
            steps=config.max_path_len // step_size,
            batch_size=initial_position.shape[0],
            nesterov_config=nesterov_config,
        ),
        init=(initial_position, config.key, nesterov_state),
        length=config.iterations + tuning_steps,
    )

    return p[tuning_steps:], q[tuning_steps:]
