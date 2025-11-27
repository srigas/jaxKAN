import jax.numpy as jnp

from flax import nnx


class ActLayer(nnx.Module):
    """
    ActLayer implementation based on:
    "Deep Learning Alternatives of the Kolmogorov Superposition Theorem"
    by Leonardo Ferreira Guilhoto and Paris Perdikaris (arXiv:2410.01990)
    
    Forward pass: ActLayer(x) = S(Λ ⊙ (β @ B(x)))
    
    Where:
    - B(x) is the basis expansion matrix with B(x)_{ij} = b_i(x_j), shape (N, d)
    - β ∈ R^{m × N} are the basis coefficients
    - Λ ∈ R^{m × d} are the mixing weights
    - S is the row-sum function
    - ⊙ is the Hadamard (element-wise) product
    
    The k-th output is: (ActLayer(x))_k = Σ_i λ_{ki} Σ_j β_{kj} b_j(x_i)

    Attributes:
        n_in (int):
            Number of layer's incoming nodes.
        n_out (int):
            Number of layer's outgoing nodes.
        N (int):
            Number of basis functions (paper recommends N=4).
        train_basis (bool):
            Whether the basis function parameters (omega and phase) are trainable.
        seed (int):
            Random key selection for initializations wherever necessary.
    """
    
    def __init__(self,
                 n_in: int = 3, n_out: int = 4, N: int = 5, train_basis: bool = True, seed: int = 42
                ):
        """
        Initializes an ActLayer instance.

        Args:
            n_in (int):
                Number of layer's incoming nodes.
            n_out (int):
                Number of layer's outgoing nodes.
            N (int):
                Number of basis functions (paper recommends N=4).
            train_basis (bool):
                Whether the basis function parameters (omega and phase) are trainable.
            seed (int):
                Random key selection for initializations wherever necessary.
                
        Example:
            >>> layer = ActLayer(n_in=2, n_out=5, N=4, train_basis=True, seed=42)
        """
        # Setup basic parameters
        self.n_in = n_in
        self.n_out = n_out
        self.N = N
        
        # Setup nnx rngs
        self.rngs = nnx.Rngs(seed)
        
        # Initialize betas - shape (n_out, N)
        # Paper: std = 1/sqrt(N) for balanced initialization
        std_beta = jnp.sqrt(1.0 / self.N)
        self.beta = nnx.Param(nnx.initializers.normal(stddev=std_beta)(
                        self.rngs.params(), (self.n_out, self.N), jnp.float32))

        # Initialize Lambdas - shape (n_out, n_in)
        # Paper: std = 1/sqrt(d) for balanced initialization
        std_lambda = jnp.sqrt(1.0 / self.n_in)
        self.Lambda = nnx.Param(nnx.initializers.normal(stddev=std_lambda)(
                        self.rngs.params(), (self.n_out, self.n_in), jnp.float32))

        # Initialize omegas (frequencies) - shape (N,)
        # Paper: ω_i ~ N(0, 1), initialized from standard normal
        omega_init = nnx.initializers.normal(stddev=1.0)(
                        self.rngs.params(), (self.N,), jnp.float32)
        
        # Initialize phases - shape (N,)
        # Paper: p_i initialized at 0
        phase_init = jnp.zeros((self.N,))
        
        # Set omega and phase as trainable (Param) or fixed (Variable) based on train_basis
        if train_basis:
            self.omega = nnx.Param(omega_init)
            self.phase = nnx.Param(phase_init)
        else:
            self.omega = omega_init
            self.phase = phase_init
        

    def basis(self, x):
        """
        Compute normalized sinusoidal basis functions.
        
        Paper Equation 11:
        b_i(t) = (sin(ω_i * t + p_i) - μ(ω_i, p_i)) / σ(ω_i, p_i)
        
        Where μ and σ are computed assuming x ~ N(0, 1):
        μ(ω, p) = exp(-ω²/2) * sin(p)
        σ(ω, p) = sqrt(1/2 - 1/2 * exp(-2ω²) * cos(2p) - μ²)
        
        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).
            
        Returns:
            B (jnp.array):
                Basis expansion matrix, shape (batch, N, n_in),
                where B[b, j, i] = b_j(x_{b,i}).
                
        Example:
            >>> layer = ActLayer(n_in=2, n_out=5, N=4, seed=42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> B = layer.basis(x_batch)
        """
        # x shape: (batch, n_in)
        # We need B(x)_{ji} = b_j(x_i), so output shape should be (batch, N, n_in)
        
        # Expand dimensions for broadcasting
        x_expanded = x[:, None, :]  # (batch, 1, n_in)
        omega = self.omega[None, :, None]  # (1, N, 1)
        phase = self.phase[None, :, None]  # (1, N, 1)
        
        # Compute sin(ω_j * x_i + p_j) for all combinations
        wx = omega * x_expanded  # (batch, N, n_in)
        s = jnp.sin(wx + phase)  # (batch, N, n_in)
        
        # Compute mean: μ(ω, p) = exp(-ω²/2) * sin(p)
        mu = jnp.exp(-0.5 * (omega ** 2)) * jnp.sin(phase)  # (1, N, 1)
        
        # Compute std: σ(ω, p) = sqrt(1/2 - 1/2 * exp(-2ω²) * cos(2p) - μ²)
        var = 0.5 * (1.0 - jnp.exp(-2.0 * (omega ** 2)) * jnp.cos(2.0 * phase)) - mu ** 2
        std = jnp.sqrt(jnp.maximum(var, 1e-8))  # (1, N, 1)
        
        # Normalize basis
        eps = 1e-8
        B = (s - mu) / (std + eps)  # (batch, N, n_in)
        
        return B


    def __call__(self, x):
        """
        The layer's forward pass.
        
        Paper Equation 6: ActLayer(x) = S(Λ ⊙ (β @ B(x)))
        Paper Equation 9: (ActLayer(x))_k = Σ_i λ_{ki} Σ_j β_{kj} b_j(x_i)
        
        Args:
            x (jnp.array):
                Inputs, shape (batch, n_in).
            
        Returns:
            y (jnp.array):
                Output of the forward pass, shape (batch, n_out).
                
        Example:
            >>> layer = ActLayer(n_in=2, n_out=5, N=4, seed=42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = layer(x_batch)
        """
        # Calculate basis: B shape (batch, N, n_in) where B[b,j,i] = b_j(x_{b,i})
        B = self.basis(x)
        
        # Get parameters
        beta = self.beta      # (n_out, N)
        Lambda = self.Lambda  # (n_out, n_in)
        
        # Compute inner function expansion: Φ(x) = β @ B(x)
        # Φ[b, k, i] = Σ_j β[k,j] * B[b,j,i]
        # This gives the inner functions φ_k(x_i) for each output k
        Phi = jnp.einsum('kj,bji->bki', beta, B)  # (batch, n_out, n_in)
        
        # Apply Lambda weights and sum: y_k = Σ_i λ_{ki} * Φ[k,i]
        # This is S(Λ ⊙ Φ) where S is row-sum
        y = jnp.einsum('bki,ki->bk', Phi, Lambda)  # (batch, n_out)
        
        return y


class ActNet(nnx.Module):
    """
    ActNet architecture based on:
    "Deep Learning Alternatives of the Kolmogorov Superposition Theorem"
    by Leonardo Ferreira Guilhoto and Paris Perdikaris (arXiv:2410.01990)

    Attributes:
        layer_dims (List[int]):
            Defines the network in terms of nodes. E.g. [2,5,1] is a network with 2 layers.
        N (int):
            Number of basis functions per ActLayer (paper recommends N=4).
        add_bias (bool):
            Whether to add learnable bias after each ActLayer.
        omega0 (float):
            Frequency multiplier for input (paper's Appendix D.1).
        use_projections (bool):
            Whether to use input/output linear projections.
        train_basis (bool):
            Whether the basis function parameters (omega and phase) are trainable.
        seed (int):
            Random key selection for initializations wherever necessary.
    """
    
    def __init__(self, 
                 layer_dims, 
                 N: int = 4, 
                 add_bias: bool = True,
                 omega0: float = 1.0,
                 use_projections: bool = False,
                 train_basis: bool = True,
                 seed: int = 42):
        """
        Initializes an ActNet model.

        Args:
            layer_dims (List[int]):
                Defines the network in terms of nodes. E.g. [2,5,1] is a network with 2 layers.
            N (int):
                Number of basis functions per ActLayer (paper recommends N=4).
            add_bias (bool):
                Whether to add learnable bias after each ActLayer.
            omega0 (float):
                Frequency multiplier for input (paper's Appendix D.1).
            use_projections (bool):
                Whether to use input/output linear projections.
            train_basis (bool):
                Whether the basis function parameters (omega and phase) are trainable.
            seed (int):
                Random key selection for initializations wherever necessary.
                
        Example:
            >>> model = ActNet(layer_dims=[2, 5, 1], N=4, add_bias=True, train_basis=True, seed=42)
        """
        # Setup nnx rngs
        self.rngs = nnx.Rngs(seed)
                
        self.add_bias = add_bias
        self.omega0 = omega0
        self.use_projections = use_projections
        
        input_dim = layer_dims[0]
        output_dim = layer_dims[-1]
        
        # If using projections, ActLayers operate on hidden dimensions only
        if use_projections and len(layer_dims) > 2:
            hidden_dim = layer_dims[1]
            
            # Input projection: input_dim -> hidden_dim
            self.input_proj = nnx.Linear(input_dim, hidden_dim, rngs=self.rngs)
            
            # ActLayers operate on hidden dimensions
            self.layers = nnx.List([
                ActLayer(
                    n_in=layer_dims[i],
                    n_out=layer_dims[i + 1],
                    N=N,
                    train_basis=train_basis,
                    seed=seed
                )
                for i in range(1, len(layer_dims) - 2)
            ])
            
            # Output projection: hidden_dim -> output_dim
            self.output_proj = nnx.Linear(layer_dims[-2], output_dim, rngs=self.rngs)
        else:
            # Standard mode: ActLayers for all layer transitions
            self.layers = nnx.List([
                ActLayer(
                    n_in=layer_dims[i],
                    n_out=layer_dims[i + 1],
                    N=N,
                    train_basis=train_basis,
                    seed=seed
                )
                for i in range(len(layer_dims) - 1)
            ])
    
        if self.add_bias:
            if use_projections and len(layer_dims) > 2:
                # Biases for hidden layers only
                self.biases = nnx.List([
                    nnx.Param(jnp.zeros((layer_dims[i+1],))) 
                    for i in range(1, len(layer_dims) - 2)
                ])
            else:
                self.biases = nnx.List([
                    nnx.Param(jnp.zeros((dim,))) for dim in layer_dims[1:]
                ])

    
    def __call__(self, x):
        """
        Equivalent to the network's forward pass.

        Args:
            x (jnp.array):
                Inputs for the first layer, shape (batch, layer_dims[0]).

        Returns:
            x (jnp.array):
                Network output, shape (batch, layer_dims[-1]).
            
        Example:
            >>> model = ActNet(layer_dims=[2, 5, 1], N=4, add_bias=True, seed=42)
            >>>
            >>> key = jax.random.key(42)
            >>> x_batch = jax.random.uniform(key, shape=(100, 2), minval=-1.0, maxval=1.0)
            >>>
            >>> output = model(x_batch)
        """
        # Apply omega0 frequency scaling (Appendix D.1)
        if self.omega0 != 1.0:
            x = self.omega0 * x
        
        # Input projection if enabled
        if self.use_projections and hasattr(self, 'input_proj'):
            x = self.input_proj(x)
        
        # Pass through each ActLayer
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.add_bias and i < len(self.biases):
                x += self.biases[i]
        
        # Output projection if enabled
        if self.use_projections and hasattr(self, 'output_proj'):
            x = self.output_proj(x)

        return x