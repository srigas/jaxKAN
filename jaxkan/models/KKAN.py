import jax.numpy as jnp

from flax import nnx

from typing import Union, List

from ..layers import get_layer
from ..layers.Dense import Dense
from ..layers.Chebyshev import Cb
from .utils import get_activation


class ChebyshevEmbedding(nnx.Module):
    """
    Chebyshev polynomial embedding layer with trainable coefficients.
    
    For an input x, computes: [c_0 * T_0(x), c_1 * T_1(x), ..., c_{D_e} * T_{D_e}(x)]
    where T_n are Chebyshev polynomials of the first kind and c_n are trainable parameters.

    Attributes:
        D_e (int):
            Degree of Chebyshev polynomial expansion.
        use_exact (bool):
            Whether to use exact Chebyshev polynomials from Cb dictionary.
        C (nnx.Param):
            Trainable coefficients for the Chebyshev polynomials.
    """
    
    def __init__(self, D_e: int):
        """
        Initializes a ChebyshevEmbedding layer.

        Args:
            D_e (int):
                Degree of Chebyshev polynomial expansion.
                
        Example:
            >>> embedding = ChebyshevEmbedding(D_e=5)
        """
        self.D_e = D_e
        
        # Check if D_e exceeds the maximum degree in the Cb dictionary
        self.use_exact = D_e <= max(Cb.keys())
        
        # Initialize trainable coefficients C_n
        # Motivated by code accompanying original paper: c_i = 1/(i+1) for i = 0, 1, ..., D_e
        C_init = jnp.array([1.0 / (i + 1) for i in range(D_e + 1)], dtype=jnp.float32)
        self.C = nnx.Param(C_init)


    def __call__(self, x):
        """
        Applies Chebyshev embedding to input.

        Args:
            x (jnp.array):
                Input tensor, shape (batch, n_features) or (batch,).

        Returns:
            embedded (jnp.array):
                Chebyshev embedded tensor, shape (batch, n_features * (D_e + 1)).
                
        Example:
            >>> embedding = ChebyshevEmbedding(D_e=5)
            >>> x = jax.random.uniform(jax.random.key(0), (100, 1))
            >>> y = embedding(x)  # shape: (100, 6)
        """
        # Handle 1D input (batch,) -> (batch, 1)
        if x.ndim == 1:
            x = x[:, None]
        
        batch = x.shape[0]
        n_features = x.shape[1]
        
        if self.use_exact:
            # Use pre-defined Chebyshev polynomials from Cb dictionary
            # Compute Chebyshev polynomials T_0(x) to T_{D_e}(x)
            # Shape: (batch, n_features, D_e + 1)
            cheb = jnp.stack([Cb[i](x) for i in range(self.D_e + 1)], axis=-1)
        else:
            # Use recursive formula for higher degrees
            # T_0(x) = 1, T_1(x) = x, T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
            cheb = jnp.ones((batch, n_features, self.D_e + 1))
            cheb = cheb.at[:, :, 1].set(x)
            for K in range(2, self.D_e + 1):
                cheb = cheb.at[:, :, K].set(2 * x * cheb[:, :, K - 1] - cheb[:, :, K - 2])
        
        # Apply trainable coefficients: c_n * T_n(x)
        # C shape: (D_e + 1,), broadcast to (batch, n_features, D_e + 1)
        weighted_cheb = cheb * self.C
        
        # Flatten the last two dimensions
        # Shape: (batch, n_features * (D_e + 1))
        embedded = weighted_cheb.reshape(batch, -1)
        
        return embedded


class InnerBlock(nnx.Module):
    """
    Inner Block for KKAN architecture.
    
    The Inner Block processes a single input dimension x_p through:
    1. Chebyshev embedding: x_p -> [c_0*T_0(x_p), ..., c_{D_e}*T_{D_e}(x_p)] -> (D_e+1)-dim
    2. Input Dense layer: (D_e+1)-dim -> H-dim
    3. L hidden Dense layers: H-dim -> H-dim (each followed by activation)
    4. Output Chebyshev embedding: H-dim -> H*(D_e+1)-dim (flattened)
    5. Final Dense layer: H*(D_e+1)-dim -> m-dim

    Attributes:
        activation (callable):
            Activation function.
        input_embedding (ChebyshevEmbedding):
            Chebyshev embedding layer for input.
        input_layer (Dense):
            Dense layer after input embedding.
        hidden_layers (nnx.List):
            List of hidden Dense layers.
        output_embedding (ChebyshevEmbedding):
            Chebyshev embedding layer for output.
        output_layer (Dense):
            Final Dense layer.
    """
    
    def __init__(self,
                 D_e: int = 7,
                 H: int = 32,
                 L: int = 4,
                 m: int = 32,
                 activation: str = 'tanh',
                 seed: int = 42
                ):
        """
        Initializes an InnerBlock.

        Args:
            D_e (int):
                Degree of Chebyshev polynomial expansion.
            H (int):
                Hidden dimension for MLP layers.
            L (int):
                Number of hidden layers.
            m (int):
                Output dimension.
            activation (str):
                Activation function.
            seed (int):
                Random seed.
                
        Example:
            >>> inner_block = InnerBlock(D_e=5, H=32, L=2, m=10, activation='tanh', seed=42)
        """
        self.activation = get_activation(activation)
        
        # Input Chebyshev embedding
        self.input_embedding = ChebyshevEmbedding(D_e=D_e)
        
        # Input Dense layer: (D_e + 1) -> H
        self.input_layer = Dense(n_in=D_e + 1, n_out=H, seed=seed)
        
        # Hidden Dense layers: H -> H
        self.hidden_layers = nnx.List([
            Dense(n_in=H, n_out=H, seed=seed + i + 1)
            for i in range(L)
        ])
        
        # Output Chebyshev embedding (operates on H-dimensional vector)
        self.output_embedding = ChebyshevEmbedding(D_e=D_e)
        
        # Final Dense layer: H * (D_e + 1) -> m
        self.output_layer = Dense(n_in=H * (D_e + 1), n_out=m, seed=seed)


    def __call__(self, x_p):
        """
        Forward pass through the inner block for a single input component.

        Args:
            x_p (jnp.array):
                Single input component, shape (batch, 1).

        Returns:
            out (jnp.array):
                Output tensor, shape (batch, m).
                
        Example:
            >>> inner_block = InnerBlock(D_e=5, H=32, L=2, m=10, seed=42)
            >>> x_p = jax.random.uniform(jax.random.key(0), (100, 1))
            >>> y = inner_block(x_p)  # shape: (100, 10)
        """
        # Step 1: Input Chebyshev embedding
        # x_p: (batch, 1) -> (batch, D_e + 1)
        h = self.input_embedding(x_p)
        
        # Step 2: Input Dense layer + activation
        # (batch, D_e + 1) -> (batch, H)
        h = self.input_layer(h)
        h = self.activation(h)
        
        # Step 3: L hidden Dense layers + activations
        # (batch, H) -> (batch, H)
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.activation(h)
        
        # Step 4: Output Chebyshev embedding
        # (batch, H) -> (batch, H * (D_e + 1))
        h = self.output_embedding(h)
        
        # Step 5: Final Dense layer (no activation)
        # (batch, H * (D_e + 1)) -> (batch, m)
        out = self.output_layer(h)
        
        return out


class OuterBlock(nnx.Module):
    """
    Outer Block for KKAN architecture.
    
    This is a wrapper around existing KAN layers from jaxkan.layers.
    It applies the selected KAN layer to map from m dimensions to n_out dimensions.

    Attributes:
        layer (nnx.Module):
            The underlying KAN layer instance.
    """
    
    def __init__(self,
                 m: int,
                 n_out: int,
                 layer_type: str = 'sine',
                 layer_params: Union[dict, None] = {'D': 7, 'init_scheme': {'type': 'glorot_fine'}},
                 seed: int = 42
                ):
        """
        Initializes an OuterBlock.

        Args:
            m (int):
                Input dimension.
            n_out (int):
                Output dimension.
            layer_type (str):
                Type of KAN layer ('chebyshev', 'legendre', 'rbf', 'sine', 'fourier', etc.).
            layer_params (dict, optional):
                Additional parameters for the KAN layer (e.g., D, flavor, kernel).
            seed (int):
                Random seed.
                
        Example:
            >>> outer_block = OuterBlock(m=10, n_out=1, layer_type='chebyshev', 
            ...                               layer_params={'D': 5}, seed=42)
        """
        
        # Get the layer class
        LayerClass = get_layer(layer_type)
        
        # Default layer parameters
        if layer_params is None:
            layer_params = {}
        
        # Create the KAN layer
        self.layer = LayerClass(
            n_in=m,
            n_out=n_out,
            seed=seed,
            **layer_params
        )


    def __call__(self, xi):
        """
        Forward pass through the outer block.

        Args:
            xi (jnp.array):
                Input from combination layer, shape (batch, m).

        Returns:
            y (jnp.array):
                Output tensor, shape (batch, n_out).
                
        Example:
            >>> outer_block = OuterBlock(m=10, n_out=1, layer_type='chebyshev', seed=42)
            >>> xi = jax.random.uniform(jax.random.key(0), (100, 10))
            >>> y = outer_block(xi)  # shape: (100, 1)
        """
        return self.layer(xi)


class KKAN(nnx.Module):
    """
    KKAN architecture based on:
    "KKANs: Kůrková-Kolmogorov-Arnold Networks and Their Learning Dynamics"
    by Juan Diego Toscano, Li-Lian Wang, and George Em Karniadakis

    Attributes:
        n_in (int):
            Input dimension (d).
        inner_blocks (nnx.List):
            List of InnerBlock modules, one for each input dimension.
        outer_block (OuterBlock):
            The outer block that produces the final output.
    """
    
    def __init__(self,
                 n_in: int,
                 n_out: int,
                 m: int = 32,
                 D_e: int = 7,
                 H: int = 32,
                 L: int = 4,
                 activation: str = 'tanh',
                 outer_layer_type: str = 'sine',
                 outer_layer_params: Union[dict, None] = {'D': 7, 'init_scheme': {'type': 'glorot_fine'}},
                 seed: int = 42):
        """
        Initializes a KKAN model.

        Args:
            n_in (int):
                Input dimension.
            n_out (int):
                Output dimension.
            m (int):
                Intermediate dimension.
            D_e (int):
                Degree of Chebyshev expansion in inner blocks.
            H (int):
                Hidden dimension for inner block MLP.
            L (int):
                Number of hidden layers in inner block MLP.
            activation (str):
                Activation function ('tanh', 'relu', 'silu', 'gelu').
            outer_layer_type (str):
                Type of KAN layer for outer block ('chebyshev', 'legendre', 'rbf', 'sine', etc.).
            outer_layer_params (dict, optional):
                Additional parameters for outer block layer (e.g., {'D': 5, 'flavor': 'exact'}).
            seed (int):
                Random seed.
                
        Example:
            >>> model = KKAN(n_in=2, n_out=1, m=10, D_e=5, H=32, L=2, 
            ...              outer_layer_type='chebyshev', seed=42)
        """
        self.n_in = n_in
        
        # Create d inner blocks (one for each input dimension)
        self.inner_blocks = nnx.List([
            InnerBlock(
                D_e=D_e,
                H=H,
                L=L,
                m=m,
                activation=activation,
                seed=seed + p
            )
            for p in range(n_in)
        ])
        
        # Outer block (KAN layer)
        self.outer_block = OuterBlock(
            m=m,
            n_out=n_out,
            layer_type=outer_layer_type,
            layer_params=outer_layer_params,
            seed=seed
        )

    def __call__(self, x):
        """
        Forward pass through the KKAN model.

        Args:
            x (jnp.array):
                Input tensor, shape (batch, n_in).

        Returns:
            y (jnp.array):
                Output tensor, shape (batch, n_out).
            
        Example:
            >>> model = KKAN(n_in=2, n_out=1, m=10, D_e=5, H=32, L=2, seed=42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = model(x_batch)
        """
        # Part 1: Inner Blocks
        # Process each input dimension through its corresponding inner block
        psi_outputs = []
        for p in range(self.n_in):
            x_p = x[:, p:p+1]  # (batch, 1)
            psi_p = self.inner_blocks[p](x_p)  # (batch, m)
            psi_outputs.append(psi_p)
        
        # Part 2: Combination Layer
        # Sum the outputs from all inner blocks
        # ξ_q = Σ_{p=1}^{d} Ψ_p,q(x_p)
        xi = jnp.stack(psi_outputs, axis=0).sum(axis=0)  # (batch, m)
        
        # Part 3: Outer Block
        # Apply KAN layer to produce final output
        y = self.outer_block(xi)  # (batch, n_out)
        
        return y
