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
