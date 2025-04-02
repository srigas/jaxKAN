import jax.numpy as jnp

# Dictionary of Chebyshev polynomials up to degree 10
Cb = {
    0: lambda x: jnp.ones_like(x),
    1: lambda x: x,
    2: lambda x: 2 * x**2 - 1,
    3: lambda x: 4 * x**3 - 3 * x,
    4: lambda x: 8 * x**4 - 8 * x**2 + 1,
    5: lambda x: 16 * x**5 - 20 * x**3 + 5 * x,
    6: lambda x: 32 * x**6 - 48 * x**4 + 18 * x**2 - 1,
    7: lambda x: 64 * x**7 - 112 * x**5 + 56 * x**3 - 7 * x,
    8: lambda x: 128 * x**8 - 256 * x**6 + 160 * x**4 - 32 * x**2 + 1,
    9: lambda x: 256 * x**9 - 576 * x**7 + 432 * x**5 - 120 * x**3 + 9 * x,
    10: lambda x: 512 * x**10 - 1280 * x**8 + 1120 * x**6 - 400 * x**4 + 50 * x**2 - 1,
}