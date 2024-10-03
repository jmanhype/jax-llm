import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from functools import partial

def fractal_mandelbrot(c, max_iter=100):
    """Generate a Mandelbrot set fractal."""
    def body_fun(carry, _):
        z, n = carry
        z = z**2 + c
        n = n + 1
        return (z, n), None

    init = (jnp.zeros_like(c), jnp.zeros(c.shape, dtype=jnp.int32))
    (z, n), _ = jax.lax.scan(body_fun, init, None, length=max_iter)
    return n * (jnp.abs(z) <= 2)

@partial(jit, static_argnums=(0,))
def create_fractal(size, scale=3.0, center=(-0.65, 0.0)):
    """Create a Mandelbrot fractal image."""
    y, x = jnp.mgrid[-1:1:size*1j, -1:1:size*1j]
    c = scale * (x + 1j*y) + complex(*center)
    fractal = vmap(vmap(fractal_mandelbrot))(c)
    return fractal / jnp.max(fractal)

key = random.PRNGKey(0)
size = 1000
fractal = create_fractal(size)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.imshow(fractal, cmap='hot', extent=[-2, 1, -1.5, 1.5])
plt.colorbar()
plt.title("Mandelbrot Set Fractal")
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.tight_layout()
plt.show()

print("Fractal generation complete. Shape:", fractal.shape)
