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

    log_min_step_size: float = struct.field(pytree_node=False, default_factory=lambda: jnp.log(1e-3))
    log_max_step_size: float = struct.field(pytree_node=False, default_factory=lambda: jnp.log(1e1))


def _step(carry: NesterovState, pa: float, config: NesterovConfig):
    error = carry.error + config.goal - pa
    log_step = config.mu - error / (jnp.sqrt(carry.t) * config.gamma)
    log_step = jnp.clip(
        log_step,
        min=config.log_min_step_size,
        max=config.log_max_step_size,
    )

    eta = (carry.t + config.to) ** (-config.kappa)  # This avoids attributing excessive weight to initial iterations
    running_avg = (1 - eta) * carry.running_avg + eta * log_step

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

    size: jax.Array

    mu: jax.Array


def cholesky_step(k: int, carry: tuple[jax.Array, jax.Array]):
    # See https://christian-igel.github.io/paper/AMERCMAUfES.pdf
    L, v = carry
    _, d = v.shape

    # Update the diagonal elements
    L_kk = L[:, k, k]
    v_k = v[:, k]
    r = jnp.sqrt(L_kk**2 + v_k**2)
    L = L.at[:, k, k].set(r)

    c = (r / L_kk)[:, None]
    s = (v_k / L_kk)[:, None]

    # Update the Cholesky decomposition
    mask = jnp.tril(jnp.ones((d, d)), k=1)
    m_kk = mask[None, k, :]

    L = L.at[:, :, k].set(m_kk * (L[:, :, k] + s * v[:, :]) / c + (1 - m_kk) * L[:, :, k])
    v = m_kk * (c * v - s * L[:, :, k]) + (1 - m_kk) * v

    return L, v


def welford_step(welford_state: WelfordState, q: jax.Array):
    # At each step, we compute the within-chain covariance, and average their values
    # to update L
    B, d = q.shape
    L = welford_state.L
    n = welford_state.size + 1
    mu = welford_state.mu
    delta = q - mu[None, ...]  # (B, d)

    # Update the average
    mu = mu + delta / n
    u = q - mu

    # Update the cholesky factorization
    v = u / jnp.sqrt(n)  # (B, d)
    L = jnp.broadcast_to(L, (B, d, d))  # (B, d, d)
    L, _ = jax.lax.fori_loop(lower=0, upper=d, body_fun=cholesky_step, init_val=(L, v))
    L = L.mean(axis=0)  # (d, d)
    mu = mu.mean(axis=0)  # (d,)

    return welford_state.replace(
        L=L,
        mu=mu,
        size=n,
    )
