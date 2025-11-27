import jax.numpy as jnp

from flax import nnx

from typing import Union


class Dense(nnx.Module):
    """
    Weight-normalized Dense layer for use in MLP architectures.
    
    This layer implements weight normalization as described in:
    "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks"
    by Salimans & Kingma (arXiv:1602.07868)
    
    Note: This is not a KAN layer, but a standard MLP building block used in advanced
    KAN architectures like KKAN (see jaxkan.models module).

    Attributes:
        rngs (nnx.Rngs):
            Random number generator state.
        W (nnx.Param):
            Weight matrix.
        g (nnx.Param):
            Scale parameter for weight normalization.
        b (Union[nnx.Param, None]):
            Bias parameter if add_bias is True, else None.
    """
    
    def __init__(self, n_in: int, n_out: int, init_scheme: str = 'glorot',
                 add_bias: bool = True, seed: int = 42):
        """
        Initializes a Dense layer with weight normalization.

        Args:
            n_in (int):
                Number of input features.
            n_out (int):
                Number of output features.
            init_scheme (str):
                Initialization scheme for weight matrix W. Options:
                - 'glorot' or 'xavier': Glorot/Xavier normal initialization (default)
                - 'glorot_uniform': Glorot/Xavier uniform initialization
                - 'he' or 'kaiming': He/Kaiming normal initialization
                - 'he_uniform': He/Kaiming uniform initialization
                - 'lecun': LeCun normal initialization
                - 'normal': Standard normal initialization
                - 'uniform': Uniform initialization in [-1, 1]
            add_bias (bool):
                Whether to include a bias term.
            seed (int):
                Random seed for initialization.
                
        Example:
            >>> layer = Dense(n_in=64, n_out=32, init_scheme='glorot', add_bias=True, seed=42)
        """
        # Setup nnx rngs
        self.rngs = nnx.Rngs(seed)
        
        # Get the initializer based on init_scheme
        initializer = self._get_initializer(init_scheme.lower())
        
        # Initialize weight matrix W
        # Shape: (n_in, n_out)
        self.W = nnx.Param(initializer(
            self.rngs.params(), (n_in, n_out), jnp.float32))
        
        # Initialize scale parameter g (one per output feature)
        # Shape: (n_out,)
        self.g = nnx.Param(jnp.ones((n_out,)))
        
        # Initialize bias parameter b
        # Shape: (n_out,)
        if add_bias:
            self.b = nnx.Param(jnp.zeros((n_out,)))
        else:
            self.b = None

    def _get_initializer(self, init_scheme: str):
        """
        Returns the appropriate initializer based on the scheme name.

        Args:
            init_scheme (str):
                Name of the initialization scheme.

        Returns:
            initializer:
                An nnx initializer function.
        """
        init_map = {
            'glorot': nnx.initializers.glorot_normal(),
            'xavier': nnx.initializers.glorot_normal(),
            'glorot_uniform': nnx.initializers.glorot_uniform(),
            'xavier_uniform': nnx.initializers.glorot_uniform(),
            'he': nnx.initializers.he_normal(),
            'kaiming': nnx.initializers.he_normal(),
            'he_uniform': nnx.initializers.he_uniform(),
            'kaiming_uniform': nnx.initializers.he_uniform(),
            'lecun': nnx.initializers.lecun_normal(),
            'lecun_uniform': nnx.initializers.lecun_uniform(),
            'normal': nnx.initializers.normal(stddev=1.0),
            'uniform': nnx.initializers.uniform(scale=1.0),
        }
        
        if init_scheme not in init_map:
            raise ValueError(f"Unknown init_scheme: {init_scheme}. "
                           f"Available options: {list(init_map.keys())}")
        
        return init_map[init_scheme]

    def __call__(self, x):
        """
        Forward pass with weight normalization.
        
        Computes: y = g * (x @ V) + b, where V = W / ||W||_2 (column-wise)

        Args:
            x (jnp.array):
                Input tensor, shape (batch, n_in).

        Returns:
            y (jnp.array):
                Output tensor, shape (batch, n_out).
                
        Example:
            >>> layer = Dense(n_in=64, n_out=32, seed=42)
            >>> x = jax.random.uniform(jax.random.key(0), (100, 64))
            >>> y = layer(x)  # shape: (100, 32)
        """
        # Weight normalization: V = W / ||W||_2 (column-wise)
        W_norm = jnp.linalg.norm(self.W, axis=0, keepdims=True)
        V = self.W / (W_norm + 1e-8)
        
        # Compute output: y = g * (x @ V) + b
        y = self.g * jnp.dot(x, V)
        
        if self.b is not None:
            y = y + self.b
        
        return y
