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

    tuning_steps: int = struct.field(pytree_node=False, default=1000)


def mh_step(
    carry: tuple[jax.Array, jax.Array, NesterovState],
    _,
    pot_vmap: callable,
    pot_grad_vmap: callable,
    precm: jax.Array,
    precm_L: jax.Array,
    steps: int,
    batch_size: int,
    nesterov_config: NesterovConfig,
    tuning: bool,
):
    q, key, nesterov_state = carry

    p, key = sample_gaussian_from_precision(
        batch_size,
        q.shape[1],
        precm_L,
        key,
    )

    # We first simulate the dynamics
    p_new, q_new = leapfrog(
        p,
        q,
        precm,
        pot_grad_vmap,
        steps=steps,
        step_size=nesterov_state.step_size,
    )

    # Compute the Hamiltonian
    def hamiltonian(p, q):
        return pot_vmap(q) + 0.5 * jnp.einsum("bi, ij, bj->b", p, precm, p)

    H = hamiltonian(p, q)
    H_new = hamiltonian(p_new, q_new)

    # Compute the MH correction step
    alpha = jnp.minimum(1.0, jnp.exp(H - H_new))
    alpha = jnp.where(jnp.isnan(alpha), 0.0, alpha)

    key, subkey = jax.random.split(key, 2)
    b = jax.random.bernoulli(subkey, alpha)
    b = b[..., None]

    # We only make the move if b = 1
    p_next = jnp.where(b == 1, p_new, q)
    q_next = jnp.where(b == 1, q_new, q)

    nesterov_state = jax.lax.cond(
        tuning,
        lambda: nesterov_dual_averaging(
            nesterov_state,
            jnp.mean(alpha),
            nesterov_config,
        ),
        lambda: nesterov_state,
    )

    return (q_next, key, nesterov_state), (p_next, q_next)


def run_chain(
    pot_vmap: callable,
    pot_grad_vmap: callable,
    config: HMCConfig,
    precm_L: jax.Array,
    initial_position: jax.Array,
    nesterov_config: NesterovConfig,
    nesterov_state: NesterovState,
    steps: int,
    key: jax.Array,
    length: int,
    tuning: bool,
):
    return jax.lax.scan(
        f=partial(
            mh_step,
            pot_vmap=pot_vmap,
            pot_grad_vmap=pot_grad_vmap,
            precm=config.initial_precm,
            precm_L=precm_L,
            steps=steps,
            batch_size=initial_position.shape[0],
            nesterov_config=nesterov_config,
            tuning=tuning,
        ),
        init=(initial_position, key, nesterov_state),
        length=length,
    )


@jax.jit
def hmc(potential: Potential, initial_position: jax.Array, config: HMCConfig):
    pot_grad_vmap = jax.vmap(jax.grad(potential), in_axes=0)
    pot_vmap = jax.vmap(potential, in_axes=0)

    # We first compute the Cholesky decomposition of the precision matrix
    precm_L = jnp.linalg.cholesky(config.initial_precm)

    nesterov_state = NesterovState(step_size=config.initial_step_size)
    nesterov_config = NesterovConfig(
        mu=jnp.log(10 * config.initial_step_size),
    )

    # First step: Tuning.
    # We put a large limit to the number of steps, and mask indices exceeding the dynamic step size.
    steps = jnp.floor(config.max_path_len / config.initial_step_size).astype(jnp.int32)
    (q, key, nesterov_state), _ = run_chain(
        pot_vmap=pot_vmap,
        pot_grad_vmap=pot_grad_vmap,
        steps=steps,
        initial_position=initial_position,
        key=config.key,
        precm_L=precm_L,
        config=config,
        nesterov_state=nesterov_state,
        nesterov_config=nesterov_config,
        length=config.tuning_steps,
        tuning=True,
    )

    nesterov_state = nesterov_state.replace(
        step_size=jnp.exp(nesterov_state.running_avg),
    )

    # Second step: Sampling.
    # We fix the step size with the value encountered above.

    steps = jnp.floor(config.max_path_len / nesterov_state.step_size).astype(int)

    _, (p, q) = run_chain(
        pot_vmap=pot_vmap,
        pot_grad_vmap=pot_grad_vmap,
        precm_L=precm_L,
        steps=steps,
        key=key,
        initial_position=q,
        config=config,
        nesterov_config=nesterov_config,
        nesterov_state=nesterov_state,
        length=config.iterations,
        tuning=False,
    )

    return p, q
