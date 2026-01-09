import flax.struct as struct
import jax
import jax.numpy as jnp


@struct.dataclass
class NesterovState:
    running_avg: jax.Array = 0.0
    error: jax.Array = 0.0
    step_size: jax.Array = 0.0
    t: jax.Array = 1


@struct.dataclass
class NesterovConfig:
    mu: jax.Array
    tuning_steps: int = struct.field(pytree_node=False)

    goal: float = struct.field(pytree_node=False, default=0.8)
    gamma: float = struct.field(pytree_node=False, default=0.05)
    to: float = struct.field(pytree_node=False, default=10)
    kappa: float = struct.field(pytree_node=False, default=0.75)

    min_step_size: float = struct.field(pytree_node=False, default=1e-3)


def _step(carry: NesterovState, pa: float, config: NesterovConfig):
    error = carry.error + config.goal - pa
    log_step = config.mu - error / (jnp.sqrt(carry.t) * config.gamma)
    eta = (carry.t + config.to) ** -config.kappa  # This avoids attributing excessive weight to initial iterations
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
    should_tune = carry.t < config.tuning_steps
    return jax.lax.cond(
        should_tune,
        lambda: _step(carry, pa, config),
        lambda: carry.replace(step_size=jnp.exp(carry.running_avg), t=carry.t + 1),
    )
