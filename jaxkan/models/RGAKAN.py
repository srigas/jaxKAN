import jax.numpy as jnp

from flax import nnx

from ..layers.Sine import SineLayer
from ..layers.Chebyshev import ChebyshevLayer as Layer
from .utils import PeriodEmbedder, RFFEmbedder

from typing import Union


class RGABlock(nnx.Module):
    """
    Residual-Gated Adaptive Block for RGAKAN architecture.

    Attributes:
        InputLayer (Layer):
            First Chebyshev layer in the block.
        OutputLayer (Layer):
            Second Chebyshev layer in the block.
        alpha (nnx.Param):
            Trainable residual connection weight for the output.
        beta (nnx.Param):
            Trainable residual connection weight for the hidden state.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int, D: int = 5, flavor: str = 'exact',
                 init_scheme: Union[dict, None] = None, alpha: float = 0.0, beta: float = 1.0, 
                 seed: int = 42):
        """
        Initializes an RGABlock.

        Args:
            n_in (int):
                Input dimension.
            n_out (int):
                Output dimension.
            n_hidden (int):
                Hidden layer dimension.
            D (int):
                Degree of Chebyshev polynomials.
            flavor (str):
                Type of Chebyshev layer ('exact' or other variants).
            init_scheme (dict, optional):
                Initialization scheme for layer weights.
            alpha (float):
                Initial value for output residual connection weight.
            beta (float):
                Initial value for hidden residual connection weight.
            seed (int):
                Random seed for reproducible initialization.
                
        Example:
            >>> block = RGABlock(n_in=64, n_out=64, n_hidden=64, D=5, flavor='exact', seed=42)
        """
        # Define the 2 layers
        self.InputLayer = Layer(n_in = n_in, n_out = n_hidden, D = D, flavor = flavor, 
                                residual = None, external_weights = False, init_scheme = init_scheme, 
                                add_bias = True, seed = seed)
        self.OutputLayer = Layer(n_in = n_hidden, n_out = n_out, D = D, flavor = flavor, 
                                 residual = None, external_weights = False, init_scheme = init_scheme, 
                                 add_bias = True, seed = seed)

        # Define alpha, beta
        self.alpha = nnx.Param(jnp.array(alpha, dtype=jnp.float32))
        self.beta = nnx.Param(jnp.array(beta, dtype=jnp.float32))


    def __call__(self, x, u, v):
        """
        Forward pass through the RGA block.

        Args:
            x (jnp.array):
                Input array, shape (batch, n_in).
            u (jnp.array):
                First gating signal, shape (batch, n_hidden).
            v (jnp.array):
                Second gating signal, shape (batch, n_hidden).

        Returns:
            x (jnp.array):
                Output after applying gated attention and residual connections, shape (batch, n_out).
                
        Example:
            >>> block = RGABlock(n_in=64, n_out=64, n_hidden=64, D=5, seed=42)
            >>> x = jnp.ones((32, 64))
            >>> u = jnp.ones((32, 64))
            >>> v = jnp.zeros((32, 64))
            >>> output = block(x, u, v)  # Shape: (32, 64)
        """
        identity = x

        x = self.InputLayer(x)
        x = x * u + (1 - x) * v

        b = self.beta
        x = b * x + (1 - b) * identity

        x = self.OutputLayer(x)
        x = x * u + (1 - x) * v

        a = self.alpha
        x = a * x + (1 - a) * identity

        return x


class RGAKAN(nnx.Module):
    """
    Residual-Gated Adaptive Kolmogorov-Arnold Network (RGAKAN).
    See paper "Towards Deep Physics-Informed Kolmogorov-Arnold Networks".

    Attributes:
        pi_init (bool):
            Whether physics-informed initialization is enabled.
        n_hidden (int):
            Hidden layer dimension.
        D (int):
            Degree of Chebyshev polynomials.
        PE (Union[PeriodEmbedder, None]):
            Periodic embedder if period_axes is provided.
        FE (Union[RFFEmbedder, None]):
            Random Fourier Features embedder if rff_std is provided.
        SineBasis (Union[SineLayer, None]):
            Sine basis layer if sine_D is provided.
        U (Layer):
            First gating network.
        V (Layer):
            Second gating network.
        blocks (nnx.List):
            List of RGABlock instances.
        OutBasis (Union[nnx.Param, None]):
            Physics-informed output coefficients if pi_init is True.
        OutLayer (Union[Layer, None]):
            Standard output layer if pi_init is False.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int, num_blocks: int,
                 flavor: str = 'exact', D: int = 5, init_scheme: Union[dict, None] = None,
                 alpha: float = 0.0, beta: float = 1.0, ref: Union[None, dict] = None,
                 period_axes: Union[None, dict] = None, rff_std: Union[None, float] = None,
                 sine_D: Union[None, int] = None, seed: int = 42):
        """
        Initializes an RGAKAN model.

        Args:
            n_in (int):
                Input dimension (before any embeddings).
            n_out (int):
                Output dimension.
            n_hidden (int):
                Hidden layer dimension.
            num_blocks (int):
                Number of RGA blocks to stack.
            flavor (str):
                Type of Chebyshev layer ('exact' or other variants).
            D (int):
                Degree of Chebyshev polynomials.
            init_scheme (dict, optional):
                Initialization scheme for layer weights.
            alpha (float):
                Initial value for output residual connection weights in blocks.
            beta (float):
                Initial value for hidden residual connection weights in blocks.
            ref (dict, optional):
                Reference data for physics-informed initialization. Must contain 't', 'x', and 'usol'.
            period_axes (dict, optional):
                Dictionary for periodic embedding: {axis: (period, trainable)}.
            rff_std (float, optional):
                Standard deviation for Random Fourier Features embedding.
            sine_D (int, optional):
                Degree for sine basis layer.
            seed (int):
                Random seed for reproducible initialization.
                
        Example:
            >>> # Standard RGAKAN
            >>> model = RGAKAN(n_in=2, n_out=1, n_hidden=64, num_blocks=4, D=5, seed=42)
            >>>
            >>> # RGAKAN with periodic embedding
            >>> period_axes = {0: (2.0 * jnp.pi, False)}
            >>> model = RGAKAN(n_in=2, n_out=1, n_hidden=64, num_blocks=4, 
            ...                period_axes=period_axes, seed=42)
        """
        self.pi_init = True if ref is not None else False
        self.n_hidden = n_hidden
        self.D = D

        # Check for periodic embeddings
        if period_axes:
            self.PE = PeriodEmbedder(period_axes)
            n_in += len(period_axes.keys()) # input dimension has now changed
        else:
            self.PE = None
        
        # Check for RFF
        if rff_std:
            self.FE = RFFEmbedder(std = rff_std, n_in = n_in, embed_dim = n_hidden)
            n_in = n_hidden # input dimension has now changed
        else:
            self.FE = None
        
        # Check for Sine-Basis Layer
        if sine_D:
            self.SineBasis = SineLayer(n_in = n_in, n_out = n_hidden, D = sine_D, residual = None, 
                                       external_weights = False, init_scheme = init_scheme, 
                                       add_bias = True, seed = seed)
        else:
            self.SineBasis = None

        # Define gates
        self.U = Layer(n_in = n_hidden, n_out = n_hidden, D = D, flavor = flavor, 
                       residual = None, external_weights = False, init_scheme = init_scheme, 
                       add_bias = True, seed = seed)
        self.V = Layer(n_in = n_hidden, n_out = n_hidden, D = D, flavor = flavor, 
                       residual = None, external_weights = False, init_scheme = init_scheme, 
                       add_bias = True, seed = seed)

        # Define blocks
        self.blocks = nnx.List([])
        for i in range(num_blocks):
        
            self.blocks.append(
                RGABlock(
                    n_in = n_hidden, n_out = n_hidden, n_hidden = n_hidden, D = D, flavor = flavor, 
                    init_scheme = init_scheme, alpha = alpha, beta = beta, seed = seed
                )
            )

        # Check for physics-informed initialization
        if self.pi_init:
            C = self._pi_init(ref)
            self.OutBasis = nnx.Param(jnp.array(C))
        else:
            self.OutLayer = Layer(n_in = n_hidden, n_out = n_out, flavor = flavor, 
                                  residual = None, external_weights = False, init_scheme = init_scheme, 
                                  add_bias = True, seed = seed)
        
    def _pi_init(self, ref):
        """
        Performs physics-informed initialization for the output layer.

        Args:
            ref (dict):
                Reference data dictionary containing:
                - 't' (jnp.array): Temporal coordinates.
                - 'x' (jnp.array): Spatial coordinates.
                - 'usol' (jnp.array): Solution array where usol[0, :] is the initial condition.

        Returns:
            C (jnp.array):
                Output basis coefficients, shape (1, n_hidden, D).
                
        Example:
            >>> ref = {'t': t_array, 'x': x_array, 'usol': u_solution}
            >>> C = model._pi_init(ref)
        """
        # Get collocation points for the spatiotemporal domain to impose initial condition
        t = ref['t'].flatten()[::10] # Downsampled temporal - shape (Nt, )
        x = ref['x'].flatten() # spatial - shape (Nx, )
        tt, xx = jnp.meshgrid(t, x, indexing="ij")

        # collocation inputs - shape (batch, 2), batch = Nt*Nx
        inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])

        # Get Y for inputs
        u_0 = ref['usol'][0, :] # initial condition - shape (Nx, )
        Y = jnp.tile(u_0.flatten(), (t.shape[0], 1)) # shape (Nt, Nx)
        Y = Y.flatten().reshape(-1, 1) # shape (batch, 1)
        
        # Get Φ - essentially do a full forward pass up until the final layer
        if self.PE:
            inputs = self.PE(inputs)

        if self.FE:
            inputs = self.FE(inputs)

        if self.SineBasis:
            inputs = self.SineBasis(inputs)

        u, v = self.U(inputs), self.V(inputs)

        for block in self.blocks:
            inputs = block(inputs, u, v)

        Phi = self.U.basis(inputs) # (batch, n_hidden, D)

        # Reshape to (batch, n_hidden * D)
        Phi_flat = Phi.reshape(Phi.shape[0], -1)

        # Solve least squares to get C as shape (n_hidden * D, 1)
        result, residuals, rank, s = jnp.linalg.lstsq(
                    Phi_flat, Y, rcond=None
                )

        # result.T is shaped (1, n_hidden * D), so we reshape to (1, n_hidden, D)
        C = result.T.reshape(1, self.n_hidden, self.D)

        return C

    
    def __call__(self, x):
        """
        Forward pass through the RGAKAN model.

        Args:
            x (jnp.array):
                Input array, shape (batch, n_in).

        Returns:
            y (jnp.array):
                Model output, shape (batch, n_out).
                
        Example:
            >>> model = RGAKAN(n_in=2, n_out=1, n_hidden=64, num_blocks=4, seed=42)
            >>> x = jnp.ones((32, 2))
            >>> y = model(x)  # Shape: (32, 1)
        """
        # Apply embedders
        if self.PE:
            x = self.PE(x)

        if self.FE:
            x = self.FE(x)

        if self.SineBasis:
            x = self.SineBasis(x)

        # Get u and v
        u = self.U(x)
        v = self.V(x)

        # Pass through blocks
        for block in self.blocks:
            x = block(x, u, v)

        # If the last layer is physics-informed
        if self.pi_init:
            C = self.OutBasis.value # (1, n_hidden, D)
            # use u (or v) as a helper to apply basis on x - helper
            B = self.U.basis(x) # (batch, n_hidden, D)
            y = jnp.einsum('bhk, ohk -> bo', B, C) # (batch, 1)
        else:
            y = self.OutLayer(x)

        return y
