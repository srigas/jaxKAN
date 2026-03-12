import jax.numpy as jnp

from flax import nnx

from typing import Union


class DenseLayer(nnx.Module):
    """
    Dense layer with random weight factorization (RWF) for use in MLP architectures.
    
    Note: This is not a KAN layer, but a standard MLP building block used in advanced
    KAN architectures like KKAN (see jaxkan.models module).

    Attributes:
        g (nnx.Param):
            Scale factor vector of shape (n_out,) from the RWF reparameterization.
        v (nnx.Param):
            Direction matrix of shape (n_in, n_out) from the RWF reparameterization.
        b (nnx.Param or None):
            Bias vector of shape (n_out,), or None if add_bias is False.
        activation (callable or None):
            Activation function applied after the linear transformation, or None.
    """
    
    def __init__(self, n_in: int, n_out: int, activation = None,
                 RWF: dict = {"mean": 1.0, "std": 0.1},
                 add_bias: bool = True, seed: int = 42):
        """
        Initializes a Dense layer with RWF.

        Args:
            n_in (int):
                Number of input features.
            n_out (int):
                Number of output features.
            activation (callable, optional):
                Activation function applied after the linear transformation.
                Defaults to None.
            RWF (dict, optional):
                Dictionary with keys ``'mean'`` and ``'std'`` controlling the
                log-normal scale of the RWF reparameterization.
                Defaults to ``{"mean": 1.0, "std": 0.1}``.
            add_bias (bool, optional):
                Whether to include a learnable bias term. Defaults to True.
            seed (int, optional):
                Random seed for parameter initialization. Defaults to 42.

        Example:
            >>> layer = DenseLayer(n_in=64, n_out=32, add_bias=True, seed=42)
        """
        # Setup nnx rngs
        rngs = nnx.Rngs(seed)
        
        # Initialize kernel via RWF - shape (n_in, n_out)
        mu, sigma = RWF["mean"], RWF["std"]

        # Glorot Initialization
        stddev = jnp.sqrt(2.0/(n_in + n_out))

        # Weight matrix with shape (n_in, n_out)
        w = nnx.initializers.normal(stddev=stddev)(
                rngs.params(), (n_in, n_out), jnp.float32
            )

        # Reparameterization towards g, v
        g = nnx.initializers.normal(stddev=sigma)(
                rngs.params(), (n_out,), jnp.float32
            )
        g += mu
        g = jnp.exp(g) # shape (n_out,)
        v = w/g # shape (n_in, n_out)

        self.g = nnx.Param(g)
        self.v = nnx.Param(v)

        # Initialize bias - shape (n_out,)
        if add_bias:
            self.b = nnx.Param(jnp.zeros((n_out,)))
        else:
            self.b = None

        self.activation = activation
        

    def __call__(self, x):
        """
        Applies the dense layer to the input.

        Args:
            x (jnp.ndarray):
                Input array of shape (batch, n_in).

        Returns:
            jnp.ndarray:
                Output array of shape (batch, n_out).

        Example:
            >>> layer = DenseLayer(n_in=4, n_out=2)
            >>> x = jnp.ones((3, 4))
            >>> y = layer(x)  # shape: (3, 2)
        """
        # Reconstruct kernel
        g, v = self.g[...], self.v[...]
        kernel = g * v

        # Apply kernel and bias
        y = jnp.dot(x, kernel)
        
        if self.b is not None:
            y = y + self.b[...]

        if self.activation is not None:
            y = self.activation(y)
        
        return y
