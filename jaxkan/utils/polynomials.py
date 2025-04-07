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

# Dictionary of Legendre polynomials up to degree 10
Lg = {
    0: lambda x: jnp.ones_like(x),
    1: lambda x: x,
    2: lambda x: (1/2) * (3 * x**2 - 1),
    3: lambda x: (1/2) * (5 * x**3 - 3 * x),
    4: lambda x: (1/8) * (35 * x**4 - 30 * x**2 + 3),
    5: lambda x: (1/8) * (63 * x**5 - 70 * x**3 + 15 * x),
    6: lambda x: (1/16) * (231 * x**6 - 315 * x**4 + 105 * x**2 - 5),
    7: lambda x: (1/16) * (429 * x**7 - 693 * x**5 + 315 * x**3 - 35*x),
    8: lambda x: (1/128) * (6435 * x**8 - 12012 * x**6 + 6930 * x**4 - 1260 * x**2 + 35),
    9: lambda x: (1/128) * (12155 * x**9 - 25740 * x**7 + 18018 * x**5 - 4620 * x**3 + 315 * x),
    10: lambda x: (1/256) * (46189 * x**10 - 109395 * x**8 + 90090 * x**6 - 30030 * x**4 + 3465 * x**2 - 63),
}
