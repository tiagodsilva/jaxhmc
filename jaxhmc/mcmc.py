from functools import partial

import flax.struct as struct
import jax
import jax.numpy as jnp

from jaxhmc.integrators import leapfrog
from jaxhmc.potentials import Potential
from jaxhmc.tuning import (
    NesterovConfig,
    NesterovState,
    WelfordState,
    nesterov_dual_averaging,
    welford_step,
)


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
    initial_step_size: float = struct.field(pytree_node=False)  # We update with dual averaging
    max_path_len: int = struct.field(pytree_node=False)  # We use jittering

    iterations: int = struct.field(pytree_node=False)

    initial_precm: jax.Array
    key: jax.Array

    fast_tuning_steps: int = struct.field(pytree_node=False, default=125)
    slow_tuning_phases: int = struct.field(pytree_node=False, default=8)
    slow_tuning_initial_length: int = struct.field(pytree_node=False, default=50)


def hmc_step(
    q: jax.Array,
    key: jax.Array,
    # For online tuning
    nesterov_state: NesterovState,
    welford_state: WelfordState,
    *,
    pot_vmap: callable,
    pot_grad_vmap: callable,
    precm: jax.Array,
    precm_L: jax.Array,
    steps: int,
    max_steps: int,
    # For tuning (static)
    nesterov_config: NesterovConfig,
    step_size_tuning: bool,
    momentum_tuning: bool,
):
    B, d = q.shape

    p, key = sample_gaussian_from_precision(B, d, precm_L, key)

    # We first simulate the dynamics
    p_new, q_new = leapfrog(
        p,
        q,
        precm,
        pot_grad_vmap,
        steps=steps,
        max_steps=max_steps,
        step_size=nesterov_state.step_size,
    )

    # Compute the Hamiltonian
    def hamiltonian(p_st, q_st):
        return pot_vmap(q_st) + 0.5 * jnp.einsum("bi, ij, bj->b", p_st, precm, p_st)

    H = hamiltonian(p, q)
    H_new = hamiltonian(p_new, q_new)

    # Compute the MH correction step
    alpha = jnp.minimum(1.0, jnp.exp(H - H_new))
    alpha = jnp.where(jnp.isnan(alpha), 0.0, alpha)

    key, subkey = jax.random.split(key, 2)
    b = jax.random.bernoulli(subkey, alpha)
    b = b[..., None]

    # We only make the move if b = 1
    p_next = jnp.where(b == 1, p_new, p)
    q_next = jnp.where(b == 1, q_new, q)

    if step_size_tuning:
        nesterov_state = nesterov_dual_averaging(
            nesterov_state,
            jnp.mean(alpha),
            nesterov_config,
        )

    if momentum_tuning:
        welford_state = welford_step(welford_state, q_next)

    return (q_next, key, nesterov_state, welford_state), (p_next, q_next)


def hmc_scan_step(
    carry: tuple[jax.Array, jax.Array, NesterovState, WelfordState],
    _,
    **kwargs,
):
    q, key, nesterov_state, welford_state = carry
    return hmc_step(q, key, nesterov_state, welford_state, **kwargs)


def hmc_fori_step(
    _,
    carry: tuple[jax.Array, jax.Array, NesterovState, WelfordState],
    **kwargs,
):
    q, key, nesterov_state, welford_state = carry
    out, _ = hmc_step(q, key, nesterov_state, welford_state, **kwargs)
    return out


def run_hmc_chain(
    key: jax.Array,
    initial_position: jax.Array,
    nesterov_state: NesterovState,
    welford_state: WelfordState,
    length: int,
    **kwargs,
):
    return jax.lax.scan(
        f=partial(
            hmc_scan_step,
            **kwargs,
        ),
        init=(initial_position, key, nesterov_state, welford_state),
        length=length,
    )


def update_welford_state(
    i: int,
    carry: tuple[jax.Array, WelfordState, jax.Array, jax.Array, jax.Array],
    initial_chain_length: int,
    nesterov_state: NesterovState,
    **kwargs,
):
    q, key, welford_state, precm, precm_L = carry
    new_q, key, nesterov_state, new_ws = jax.lax.fori_loop(
        lower=0,
        upper=initial_chain_length * (2**i),
        body_fun=partial(
            hmc_fori_step,
            step_size_tuning=True,
            momentum_tuning=True,
            precm=precm,
            precm_L=precm_L,
            **kwargs,
        ),
        init_val=(q, key, nesterov_state, welford_state),
    )

    precm = jnp.linalg.inv(new_ws.C / new_ws.size)
    precm_L = jnp.linalg.cholesky(precm)
    new_ws = new_ws.replace(size=0)

    return new_q, key, new_ws, precm, precm_L


def hmc_tune(
    pot_vmap: callable,
    pot_grad_vmap: callable,
    precm: jax.Array,
    precm_L: jax.Array,
    steps: int,
    max_steps: int,
    initial_position: jax.Array,
    nesterov_state: NesterovState,
    nesterov_config: NesterovConfig,
    welford_state: WelfordState,
    config: HMCConfig,
):
    kwargs = {
        "pot_vmap": pot_vmap,
        "pot_grad_vmap": pot_grad_vmap,
        "steps": steps,
        "max_steps": max_steps,
        "nesterov_config": nesterov_config,
    }

    # We first tune the step size (fast tuning).
    (q, key, nesterov_state, welford_state), _ = run_hmc_chain(
        **kwargs,
        length=config.fast_tuning_steps,
        key=config.key,
        welford_state=welford_state,
        nesterov_state=nesterov_state,
        initial_position=initial_position,
        precm=precm,
        precm_L=precm_L,
        step_size_tuning=True,
        momentum_tuning=False,
    )

    # We then tune the momentum distribution (slow, step-wise tuning).
    # For this, we follow Stan's implementation and run our chain on
    # increasing times, specifically, k * 2 ** i for 1 <= i <= T
    # k and T are user-specified parameters (fixed at k = 50, T = 8).
    # The i-th interval uses the momentum matrix M^{(i - 1)} obtained
    # from the prior interval.

    (q, key, welford_state, precm, precm_L) = jax.lax.fori_loop(
        lower=0,
        upper=config.slow_tuning_phases,
        # carry = p, q, nesterov_state, welford_state
        body_fun=partial(
            update_welford_state,
            initial_chain_length=config.slow_tuning_initial_length,
            nesterov_state=nesterov_state,
            **kwargs,
        ),
        init_val=(q, key, welford_state, precm, precm_L),
    )

    nesterov_state = nesterov_state.replace(t=1)

    # We further refine the step size based on the updated momentum.
    (q, key, nesterov_state, _), _ = run_hmc_chain(
        **kwargs,
        length=config.fast_tuning_steps,
        key=key,
        welford_state=welford_state,
        nesterov_state=nesterov_state,
        initial_position=q,
        precm=precm,
        precm_L=precm_L,
        step_size_tuning=True,
        momentum_tuning=False,
    )

    return q, key, nesterov_state, precm, precm_L


def init_states(config: HMCConfig, q: jax.Array, precm: jax.Array):
    nesterov_state = NesterovState(
        step_size=config.initial_step_size,
        log_running_avg=jnp.log(config.initial_step_size),
    )
    nesterov_config = NesterovConfig(
        mu=jnp.log(10 * config.initial_step_size),
        gamma=1,  # Equivalent to 1 on Hoffman's works
    )

    welford_state = WelfordState(
        C=jnp.linalg.inv(precm),
        mu=jnp.zeros((q.shape[1],)),
        # This should be actually null, as we need a pair of points for
        # computing the covariance, however, letting s = 1 allows for a cleaner implementation
        size=0,
    )

    return nesterov_state, nesterov_config, welford_state


@partial(jax.jit, static_argnames=("potential",))
def hmc(potential: Potential, initial_position: jax.Array, config: HMCConfig):
    pot_grad_vmap = jax.vmap(jax.grad(potential), in_axes=0)
    pot_vmap = jax.vmap(potential, in_axes=0)

    # We first compute the Cholesky decomposition of the precision matrix
    precm_L = jnp.linalg.cholesky(config.initial_precm)

    nesterov_state, nesterov_config, welford_state = init_states(config, initial_position, config.initial_precm)

    # First step: Tuning.
    # We put a large limit to the number of steps, and mask indices exceeding the dynamic step size.
    max_steps = jnp.floor(config.max_path_len / jnp.exp(nesterov_config.log_min_step_size)).astype(jnp.int32)
    steps = jnp.floor(config.max_path_len / config.initial_step_size).astype(jnp.int32)

    nesterov_state = nesterov_state.replace(
        step_size=jnp.exp(nesterov_state.log_running_avg),
    )

    q, key, nesterov_state, precm, precm_L = hmc_tune(
        pot_vmap=pot_vmap,
        pot_grad_vmap=pot_grad_vmap,
        steps=steps,
        precm=config.initial_precm,
        precm_L=precm_L,
        max_steps=max_steps,
        initial_position=initial_position,
        nesterov_state=nesterov_state,
        nesterov_config=nesterov_config,
        welford_state=welford_state,
        config=config,
    )

    # Second step: Sampling.
    # We fix the step size with the value encountered above.

    steps = jnp.floor(config.max_path_len / nesterov_state.step_size).astype(int)

    _, (p, q) = run_hmc_chain(
        pot_vmap=pot_vmap,
        pot_grad_vmap=pot_grad_vmap,
        precm=precm,
        precm_L=precm_L,
        steps=steps,
        max_steps=steps,
        key=key,
        initial_position=q,
        nesterov_config=nesterov_config,
        nesterov_state=nesterov_state,
        welford_state=welford_state,
        length=config.iterations,
        step_size_tuning=False,
        momentum_tuning=False,
    )

    return p, q


@struct.dataclass
class RandomWalkConfig:
    key: jax.Array

    iterations: int = struct.field(pytree_node=False)
    tuning_steps: int = struct.field(pytree_node=False, default=1000)

    initial_step_size: float = 0.1


def random_walk_step(
    carry: tuple[jax.Array, jax.Array, NesterovState],
    _,
    pot_vmap: Potential,
    batch_size: int,
    nesterov_config: NesterovConfig,
    tuning: bool = False,
):
    position, key, nesterov_state = carry

    key, subkey = jax.random.split(key, 2)
    noise = jax.random.normal(subkey, position.shape)
    new_position = position + nesterov_state.step_size * noise

    # Since it is symmetric, the MH probability is the ratio of the potential energies
    alpha = jnp.minimum(1.0, jnp.exp(pot_vmap(position) - pot_vmap(new_position)))
    key, subkey = jax.random.split(key, 2)
    b = jax.random.bernoulli(subkey, p=alpha, shape=(batch_size,))[..., None]

    new_position = jnp.where(b == 1, new_position, position)

    # Update the step size via Nesterov dual averaging only if we are in the tuning phase
    if tuning:
        nesterov_state = nesterov_dual_averaging(
            nesterov_state,
            jnp.mean(alpha),
            nesterov_config,
        )

    return (new_position, key, nesterov_state), new_position


@partial(jax.jit, static_argnames=("potential",))
def random_walk(potential: Potential, initial_position: jax.Array, config: RandomWalkConfig):
    nesterov_state = NesterovState(
        step_size=config.initial_step_size,
        log_running_avg=jnp.log(config.initial_step_size),
    )
    nesterov_config = NesterovConfig(
        mu=jnp.log(10 * config.initial_step_size),
    )

    pot_vmap = jax.vmap(potential, in_axes=0)
    batch_size = initial_position.shape[0]

    # First step: Tuning.
    # We put a large limit to the number of steps, and mask indices exceeding the dynamic step size.
    (position, key, nesterov_state), _ = jax.lax.scan(
        f=partial(
            random_walk_step,
            pot_vmap=pot_vmap,
            batch_size=batch_size,
            nesterov_config=nesterov_config,
            tuning=True,
        ),
        init=(initial_position, config.key, nesterov_state),
        length=config.tuning_steps,
    )

    nesterov_state = nesterov_state.replace(
        step_size=jnp.exp(nesterov_state.log_running_avg),
    )

    # Second step: Sampling.
    # We fix the step size with the value encountered above.

    _, samples = jax.lax.scan(
        f=partial(
            random_walk_step,
            pot_vmap=pot_vmap,
            batch_size=batch_size,
            nesterov_config=nesterov_config,
            tuning=False,
        ),
        init=(position, key, nesterov_state),
        length=config.iterations,
    )

    return samples
