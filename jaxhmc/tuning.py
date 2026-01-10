import flax.struct as struct
import jax
import jax.numpy as jnp


@struct.dataclass
class NesterovState:
    running_avg: jax.Array = 0.0
    error: jax.Array = 0.0
    step_size: jax.Array = 1.0
    t: jax.Array = 1


@struct.dataclass
class NesterovConfig:
    mu: jax.Array

    goal: float = struct.field(pytree_node=False, default=0.8)
    gamma: float = struct.field(pytree_node=False, default=0.05)
    to: float = struct.field(pytree_node=False, default=10)
    kappa: float = struct.field(pytree_node=False, default=0.75)

    log_min_step_size: float = struct.field(pytree_node=False, default_factory=lambda: jnp.log(1e-2))
    log_max_step_size: float = struct.field(pytree_node=False, default_factory=lambda: jnp.log(1e1))


def _step(carry: NesterovState, pa: float, config: NesterovConfig):
    error = carry.error + config.goal - pa
    log_step = config.mu - config.gamma * error / jnp.sqrt(carry.t)
    log_step = jnp.clip(
        log_step,
        min=config.log_min_step_size,
        max=config.log_max_step_size,
    )

    eta = (carry.t + config.to) ** (-config.kappa)  # This avoids attributing excessive weight to initial iterations
    running_avg = (1 - eta) * carry.running_avg + eta * log_step

    jax.debug.print("{}", jnp.exp(log_step))
    return NesterovState(
        running_avg=running_avg,
        error=error,
        step_size=jnp.exp(log_step),
        t=carry.t + 1,
    )


def nesterov_dual_averaging(
    carry: NesterovState,
    pa: jax.Array,
    config: NesterovConfig,
):
    return _step(carry, pa, config)


@struct.dataclass
class WelfordState:
    L: jax.Array
    C: jax.Array
    mu: jax.Array
    size: jax.Array


def cholesky_step(k: int, carry: tuple[jax.Array, jax.Array]):
    L, v = carry
    _, d = L.shape[0], L.shape[1]

    L_kk = L[:, k, k]
    v_k = v[:, k]
    r = jnp.sqrt(L_kk**2 + v_k**2)
    L = L.at[:, k, k].set(r)
    c = (r / L_kk)[:, None]
    s = (v_k / L_kk)[:, None]

    mask = jnp.arange(d) > k
    L_col = L[:, :, k]

    L_col = jnp.where(mask[None, :], (L_col + s * v) / c, L_col)
    L = L.at[:, :, k].set(L_col)

    v = jnp.where(mask[None, :], c * v - s * L[:, :, k], v)

    return L, v


def welford_step(welford_state: WelfordState, q: jax.Array):
    B, d = q.shape
    L = welford_state.L
    C = welford_state.C
    mu = welford_state.mu
    n = welford_state.size

    n_new = n + 1
    delta = q - mu[None, :]
    mu_new = mu + delta.mean(axis=0) / n_new

    v = delta * jnp.sqrt(n / n_new)
    L_batched = jnp.broadcast_to(L, (B, d, d))
    L_new, _ = jax.lax.fori_loop(0, d, cholesky_step, (L_batched, v))

    L_new = L_new.mean(axis=0)
    mu_new = mu_new.mean(axis=0)

    L_new = jnp.where(n == 0, L, L_new)
    mu_new = jnp.where(n == 0, q.mean(axis=0), mu_new)
    n_new = jnp.where(n == 0, 1, n_new)

    return welford_state.replace(
        L=L_new,
        C=C,
        mu=mu_new,
        size=n_new,
    )
