from typing import List
from jax import numpy as jnp

from flax import linen as nn
from flax.linen import initializers
from flax.core import unfreeze

from .ReLUKANLayer import ReLUKANLayer


class ReLUKAN(nn.Module):
    """
    ReLUKAN class, corresponding to a network of ReLUKANLayers.

    Args:
    -----------
        layer_dims (List[int]): defines the network in terms of nodes. E.g. [4,5,1] is a network with 2 layers: one with n_in=4 and n_out=5 and one with n_in=5 and n_out = 1.
        add_bias (bool): boolean that controls wether bias terms are also included during the forward pass or not. Default: True
        p (int): input for ReLUKANLayer class - see ReLUKANLayer.py
        k (int): input for ReLUKANLayer class - see ReLUKANLayer.py
        const_R (float/bool): input for ReLUKANLayer class - see ReLUKANLayer.py
        const_res (float/bool): input for ReLUKANLayer class - see ReLUKANLayer.py
        residual (nn.Module): input for ReLUKANLayer class - see ReLUKANLayer.py
        noise_std (float): input for ReLUKANLayer class - see ReLUKANLayer.py
        grid_e (float): input for ReLUKANLayer class - see ReLUKANLayer.py
    """
    layer_dims: List[int]
    add_bias: bool = True
    
    k: int = 2
    p: int = 2
    
    const_R: float or bool = False
    const_res: float or bool = False
    residual: nn.Module = nn.swish
    
    noise_std: float = 0.1
    grid_e: float = 0.05

    
    def setup(self):
        """
        Registers and initializes all ReLUKANLayers of the architecture.
        Optionally includes a trainable bias for each ReLUKANLayer.
        """
        # Initialize ReLUKAN layers based on layer_dims            
        self.layers = [ReLUKANLayer(
                                n_in=self.layer_dims[i],
                                n_out=self.layer_dims[i + 1],
                                k=self.k,
                                p=self.p,
                                const_R=self.const_R,
                                const_res=self.const_res,
                                residual=self.residual,
                                noise_std=self.noise_std,
                                grid_e=self.grid_e
                            )
                            for i in range(len(self.layer_dims) - 1)
                      ]
        
        if self.add_bias:
            self.biases = [self.param(f'bias_{i}', initializers.zeros, (dim,)) for i, dim in enumerate(self.layer_dims[1:])]

    
    def update_grids(self, x, G_new):
        """
        Performs the grid update for each layer of the ReLUKAN architecture.

        Args:
        -----
            x (jnp.array): inputs for the first layer
                shape (batch, self.layers[0])
            G_new (int): Size of the new grid (in terms of intervals)

        """
        # Unfreeze is used to avoid in-place edits - we do not want to update the
        # parameters this way, we will simply pass the new object manually at the next calls
        updated_params = unfreeze(self.scope.variables()['params'])
        updated_state = unfreeze(self.scope.variables()['state'])

        # Loop over each layer
        for i, layer in enumerate(self.layers):
            # Extract the variables for the current layer
            layer_variables = {
                'params': updated_params[f'layers_{i}'],
                'state': updated_state[f'layers_{i}']
            }
            
            # Call the update_grid method on the current layer
            coeffs, updated_layer_state = layer.apply(layer_variables, x, G_new, method=layer.update_grid, mutable=['state'])
            
            # Update the state and parameters for the current layer
            updated_params[f'layers_{i}']['c_basis'] = coeffs
            updated_state[f'layers_{i}'] = updated_layer_state['state']

            # Forward pass with the updated parameters to get the input for the next layer
            layer_variables = {
                'params': updated_params[f'layers_{i}'],
                'state': updated_state[f'layers_{i}']
            }
            layer_output, _ = layer.apply(layer_variables, x)
            if self.add_bias:
                layer_output += self.biases[i]
            x = layer_output

        # Return a new variables dictionary to be passed to the model and optimizer
        return {'params': updated_params, 'state': updated_state}

    
    def __call__(self, x):
        """
        Equivalent to the network's forward pass.

        Args:
        -----
            x (jnp.array): inputs for the first layer
                shape (batch, self.layers[0])

        Returns:
        --------
            x (jnp.array): network output
                shape (batch, self.layers[-1])
            R_regs (List[jnp.array]): a list of R_reg arrays to be used for the loss function's regularization term.
                size len(layer_dims)-1
        """
        R_regs = []

        # Pass through each layer of the ReLUKAN
        for i, layer in enumerate(self.layers):
            x, R_reg = layer(x)

            # Add a bias term
            if self.add_bias:
                x += self.biases[i]

            # Append the regularization terms per layer in a list
            R_regs.append(R_reg)

        # Return the total output and the regularization terms
        return x, R_regs
