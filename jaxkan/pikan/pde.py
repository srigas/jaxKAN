import jax
import jax.numpy as jnp


def get_ac_res(D=1e-4, c=5.0):
    """
    Returns the Allen-Cahn equation residual function (2D).
    
    Args:
        D (float):
            Diffusion coefficient. Default is 1e-4.
        c (float):
            Reaction coefficient. Default is 5.0.
    
    Returns:
        ac_res (function):
            Residual function for the Allen-Cahn equation.
    
    Example:
        >>> # Use default parameters
        >>> ac_res = get_ac_res()
        >>> 
        >>> # Use custom parameters
        >>> ac_res = get_ac_res(D=0.001, c=10.0)
    """
    D = jnp.array(D, dtype=jnp.float32)
    c = jnp.array(c, dtype=jnp.float32)

    def ac_res(model, collocs):
        def u_fn(t, x):
            return model(jnp.array([[t, x]]))[0, 0]

        u_t_fn = jax.grad(u_fn, argnums=0)
        u_x_fn = jax.grad(u_fn, argnums=1)
        u_xx_fn = jax.grad(u_x_fn, argnums=1)

        def point_res(t, x):
            u = u_fn(t, x)
            return u_t_fn(t, x) - D * u_xx_fn(t, x) - c * (u - u**3)

        return jax.vmap(point_res, in_axes=(0, 0))(collocs[:, 0], collocs[:, 1]).reshape(-1, 1)

    return ac_res


def get_diffusion_res(D=0.25):
    """
    Returns the diffusion equation residual function (2D).
    
    Args:
        D (float):
            Diffusion coefficient. Default is 0.25.
    
    Returns:
        diffusion_res (function):
            Residual function for the diffusion equation.
    
    Example:
        >>> # Use default parameters
        >>> diffusion_res = get_diffusion_res()
        >>> 
        >>> # Use custom parameters
        >>> diffusion_res = get_diffusion_res(D=0.5)
    """
    D = jnp.array(D, dtype=jnp.float32)

    def diffusion_res(model, collocs):
        def u_fn(t, x):
            return model(jnp.array([[t, x]]))[0, 0]

        u_t_fn = jax.grad(u_fn, argnums=0)
        u_x_fn = jax.grad(u_fn, argnums=1)
        u_xx_fn = jax.grad(u_x_fn, argnums=1)

        def point_res(t, x):
            return u_t_fn(t, x) - D * u_xx_fn(t, x)

        return jax.vmap(point_res, in_axes=(0, 0))(collocs[:, 0], collocs[:, 1]).reshape(-1, 1)

    return diffusion_res


def get_burgers_res(nu=0.01/jnp.pi):
    """
    Returns the Burgers equation residual function (2D).
    
    Args:
        nu (float):
            Viscosity coefficient. Default is 0.01/π.
    
    Returns:
        burgers_res (function):
            Residual function for the Burgers equation.
    
    Example:
        >>> # Use default parameters
        >>> burgers_res = get_burgers_res()
        >>> 
        >>> # Use custom parameters
        >>> burgers_res = get_burgers_res(nu=0.001)
    """
    nu = jnp.array(nu, dtype=jnp.float32)

    def burgers_res(model, collocs):
        def u_fn(t, x):
            return model(jnp.array([[t, x]]))[0, 0]

        u_t_fn = jax.grad(u_fn, argnums=0)
        u_x_fn = jax.grad(u_fn, argnums=1)
        u_xx_fn = jax.grad(u_x_fn, argnums=1)

        def point_res(t, x):
            u = u_fn(t, x)
            return u_t_fn(t, x) + u * u_x_fn(t, x) - nu * u_xx_fn(t, x)

        return jax.vmap(point_res, in_axes=(0, 0))(collocs[:, 0], collocs[:, 1]).reshape(-1, 1)

    return burgers_res


def get_kdv_res(eta=1.0, mu=0.022):
    """
    Returns the Korteweg-de Vries equation residual function (2D).
    
    Args:
        eta (float):
            Nonlinearity coefficient. Default is 1.0.
        mu (float):
            Dispersion coefficient. Default is 0.022.
    
    Returns:
        kdv_res (function):
            Residual function for the Korteweg-de Vries equation.
    
    Example:
        >>> # Use default parameters
        >>> kdv_res = get_kdv_res()
        >>> 
        >>> # Use custom parameters
        >>> kdv_res = get_kdv_res(eta=2.0, mu=0.01)
    """
    eta = jnp.array(eta, dtype=jnp.float32)
    mu = jnp.array(mu, dtype=jnp.float32)

    def kdv_res(model, collocs):
        def u_fn(t, x):
            return model(jnp.array([[t, x]]))[0, 0]

        u_t_fn = jax.grad(u_fn, argnums=0)
        u_x_fn = jax.grad(u_fn, argnums=1)
        u_xx_fn = jax.grad(u_x_fn, argnums=1)
        u_xxx_fn = jax.grad(u_xx_fn, argnums=1)

        def point_res(t, x):
            u = u_fn(t, x)
            return u_t_fn(t, x) + eta * u * u_x_fn(t, x) + (mu**2) * u_xxx_fn(t, x)

        return jax.vmap(point_res, in_axes=(0, 0))(collocs[:, 0], collocs[:, 1]).reshape(-1, 1)

    return kdv_res


def get_sg_res(D=1.0):
    """
    Returns the sine-Gordon equation residual function (2D).
    
    Args:
        D (float):
            Wave speed coefficient. Default is 1.0.
    
    Returns:
        sg_res (function):
            Residual function for the sine-Gordon equation.
    
    Example:
        >>> # Use default parameters
        >>> sg_res = get_sg_res()
        >>> 
        >>> # Use custom parameters
        >>> sg_res = get_sg_res(D=2.0)
    """
    D = jnp.array(D, dtype=jnp.float32)

    def sg_res(model, collocs):
        def u_fn(t, x):
            return model(jnp.array([[t, x]]))[0, 0]

        u_t_fn = jax.grad(u_fn, argnums=0)
        u_tt_fn = jax.grad(u_t_fn, argnums=0)
        u_x_fn = jax.grad(u_fn, argnums=1)
        u_xx_fn = jax.grad(u_x_fn, argnums=1)

        def point_res(t, x):
            return u_tt_fn(t, x) - D * u_xx_fn(t, x) + jnp.sin(u_fn(t, x))

        return jax.vmap(point_res, in_axes=(0, 0))(collocs[:, 0], collocs[:, 1]).reshape(-1, 1)

    return sg_res


def get_advection_res(c=20.0):
    """
    Returns the advection equation residual function (2D).
    
    Args:
        c (float):
            Wave speed coefficient. Default is 20.0.
    
    Returns:
        advection_res (function):
            Residual function for the advection equation.
    
    Example:
        >>> # Use default parameters
        >>> advection_res = get_advection_res()
        >>> 
        >>> # Use custom parameters
        >>> advection_res = get_advection_res(c=10.0)
    """
    c = jnp.array(c, dtype=jnp.float32)

    def advection_res(model, collocs):
        def u_fn(t, x):
            return model(jnp.array([[t, x]]))[0, 0]

        u_t_fn = jax.grad(u_fn, argnums=0)
        u_x_fn = jax.grad(u_fn, argnums=1)

        def point_res(t, x):
            return u_t_fn(t, x) + c * u_x_fn(t, x)

        return jax.vmap(point_res, in_axes=(0, 0))(collocs[:, 0], collocs[:, 1]).reshape(-1, 1)

    return advection_res


def get_helmholtz_res(a1=1.0, a2=4.0, k=1.0):
    """
    Returns the Helmholtz equation residual function (2D).
    
    Args:
        a1 (float):
            First frequency parameter for source term. Default is 1.0.
        a2 (float):
            Second frequency parameter for source term. Default is 4.0.
        k (float):
            Wave number. Default is 1.0.
    
    Returns:
        helmholtz_res (function):
            Residual function for the Helmholtz equation.
    
    Example:
        >>> # Use default parameters
        >>> helmholtz_res = get_helmholtz_res()
        >>> 
        >>> # Use custom parameters
        >>> helmholtz_res = get_helmholtz_res(a1=2.0, a2=3.0, k=2.0)
    """
    a1 = jnp.array(a1, dtype=jnp.float32)
    a2 = jnp.array(a2, dtype=jnp.float32)
    k = jnp.array(k, dtype=jnp.float32)

    def helmholtz_res(model, collocs):
        factor = k**2 - (jnp.pi**2) * (a1**2 + a2**2)

        def u_fn(x, y):
            return model(jnp.array([[x, y]]))[0, 0]

        u_x_fn = jax.grad(u_fn, argnums=0)
        u_xx_fn = jax.grad(u_x_fn, argnums=0)
        u_y_fn = jax.grad(u_fn, argnums=1)
        u_yy_fn = jax.grad(u_y_fn, argnums=1)

        def point_res(x, y):
            f = factor * jnp.sin(jnp.pi * a1 * x) * jnp.sin(jnp.pi * a2 * y)
            return u_xx_fn(x, y) + u_yy_fn(x, y) + (k**2) * u_fn(x, y) - f

        return jax.vmap(point_res, in_axes=(0, 0))(collocs[:, 0], collocs[:, 1]).reshape(-1, 1)

    return helmholtz_res


def get_poisson_res(a1=4.0, a2=4.0):
    """
    Returns the Poisson equation residual function (2D).
    
    Args:
        a1 (float):
            First frequency parameter for source term. Default is 4.0.
        a2 (float):
            Second frequency parameter for source term. Default is 4.0.
    
    Returns:
        poisson_res (function):
            Residual function for the Poisson equation.
    
    Example:
        >>> # Use default parameters
        >>> poisson_res = get_poisson_res()
        >>> 
        >>> # Use custom parameters
        >>> poisson_res = get_poisson_res(a1=2.0, a2=3.0)
    """
    a1 = jnp.array(a1, dtype=jnp.float32)
    a2 = jnp.array(a2, dtype=jnp.float32)

    def poisson_res(model, collocs):
        factor = -(jnp.pi**2) * (a1**2 + a2**2)

        def u_fn(x, y):
            return model(jnp.array([[x, y]]))[0, 0]

        u_x_fn = jax.grad(u_fn, argnums=0)
        u_xx_fn = jax.grad(u_x_fn, argnums=0)
        u_y_fn = jax.grad(u_fn, argnums=1)
        u_yy_fn = jax.grad(u_y_fn, argnums=1)

        def point_res(x, y):
            f = factor * jnp.sin(jnp.pi * a1 * x) * jnp.sin(jnp.pi * a2 * y)
            return u_xx_fn(x, y) + u_yy_fn(x, y) - f

        return jax.vmap(point_res, in_axes=(0, 0))(collocs[:, 0], collocs[:, 1]).reshape(-1, 1)

    return poisson_res


def get_wave_res(c=4.0):
    """
    Returns the wave equation residual function (2D).
    
    Args:
        c (float):
            Wave speed coefficient. Default is 4.0.
    
    Returns:
        wave_res (function):
            Residual function for the wave equation.
    
    Example:
        >>> # Use default parameters
        >>> wave_res = get_wave_res()
        >>> 
        >>> # Use custom parameters
        >>> wave_res = get_wave_res(c=2.0)
    """
    c = jnp.array(c, dtype=jnp.float32)

    def wave_res(model, collocs):
        def u_fn(t, x):
            return model(jnp.array([[t, x]]))[0, 0]

        u_t_fn = jax.grad(u_fn, argnums=0)
        u_tt_fn = jax.grad(u_t_fn, argnums=0)
        u_x_fn = jax.grad(u_fn, argnums=1)
        u_xx_fn = jax.grad(u_x_fn, argnums=1)

        def point_res(t, x):
            return u_tt_fn(t, x) - c * u_xx_fn(t, x)

        return jax.vmap(point_res, in_axes=(0, 0))(collocs[:, 0], collocs[:, 1]).reshape(-1, 1)

    return wave_res


def get_ks_res(alpha=100/16, beta=100/(16**2), gamma=100/(16**4)):
    """
    Returns the Kuramoto-Sivashinsky equation residual function (2D).
    
    Args:
        alpha (float):
            Nonlinearity coefficient. Default is 100/16.
        beta (float):
            Second-order dispersion coefficient. Default is 100/(16²).
        gamma (float):
            Fourth-order dispersion coefficient. Default is 100/(16⁴).
    
    Returns:
        ks_res (function):
            Residual function for the Kuramoto-Sivashinsky equation.
    
    Example:
        >>> # Use default parameters
        >>> ks_res = get_ks_res()
        >>> 
        >>> # Use custom parameters
        >>> ks_res = get_ks_res(alpha=6.25, beta=0.390625, gamma=0.0009765625)
    """
    alpha = jnp.array(alpha, dtype=jnp.float32)
    beta = jnp.array(beta, dtype=jnp.float32)
    gamma = jnp.array(gamma, dtype=jnp.float32)

    def ks_res(model, collocs):
        def u_fn(t, x):
            return model(jnp.array([[t, x]]))[0, 0]

        u_t_fn = jax.grad(u_fn, argnums=0)
        u_x_fn = jax.grad(u_fn, argnums=1)
        u_xx_fn = jax.grad(u_x_fn, argnums=1)
        u_xxx_fn = jax.grad(u_xx_fn, argnums=1)
        u_xxxx_fn = jax.grad(u_xxx_fn, argnums=1)

        def point_res(t, x):
            u = u_fn(t, x)
            return u_t_fn(t, x) + alpha * u * u_x_fn(t, x) + beta * u_xx_fn(t, x) + gamma * u_xxxx_fn(t, x)

        return jax.vmap(point_res, in_axes=(0, 0))(collocs[:, 0], collocs[:, 1]).reshape(-1, 1)

    return ks_res