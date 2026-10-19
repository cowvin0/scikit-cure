import jax  # type: ignore[import-not-found]

jax.config.update(
    "jax_enable_x64",
    True,
)

import jax.numpy as jnp
import numpy as np

from jax import lax


def pgamma_shape_derivative(
    x,
    shape,
):
    return float(
        lax.igamma_grad_a(
            jnp.asarray(
                shape,
                dtype=jnp.float64,
            ),
            jnp.asarray(
                x,
                dtype=jnp.float64,
            ),
        )
    )


def pgamma_shape_derivative_vec(
    u,
    shape,
):
    u = jnp.asarray(
        u,
        dtype=jnp.float64,
    )

    result = lax.igamma_grad_a(
        jnp.asarray(
            shape,
            dtype=jnp.float64,
        ),
        u,
    )

    return np.asarray(
        result,
        dtype=float,
    )
