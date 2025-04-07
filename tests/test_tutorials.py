import pytest

from jaxkan.KAN import KAN
from jaxkan.utils.PIKAN import sobol_sample, gradf

import jax
import jax.numpy as jnp

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from flax import nnx
import optax

import numpy as np


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def req_params():
    return {'D': 5, 'flavor': 'exact'}
    

def test_function_fitting(seed, req_params):

    # Define the target function
    def f(x,y):
        return x**2 + 2*jnp.exp(y)

    # Generate sample data
    key = jax.random.key(seed)
    x_key, y_key = jax.random.split(key)

    x1 = jax.random.uniform(x_key, shape=(1000,), minval=-1, maxval=1)
    x2 = jax.random.uniform(y_key, shape=(1000,), minval=-1, maxval=1)

    y = f(x1, x2).reshape(-1, 1)
    X = jnp.stack([x1, x2], axis=1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    assert X_train.shape == (800, 2), "Incorrect X_train shape after splitting the data"
    assert X_test.shape == (200, 2), "Incorrect X_test shape after splitting the data"

    # Initialize the model
    n_in, n_out, n_hidden = X_train.shape[1], y_train.shape[1], 6    
    layer_dims = [n_in, n_hidden, n_hidden, n_out]
    
    model = KAN(layer_dims = layer_dims,
                layer_type = 'chebyshev',
                required_parameters = req_params,
                seed = seed
               )

    # Initialize optimizer
    opt_type = optax.adam(learning_rate=0.001)
    optimizer = nnx.Optimizer(model, opt_type)

    # Define the train loop
    @nnx.jit
    def train_step(model, optimizer, X_train, y_train):
    
        def loss_fn(model):
            residual = model(X_train) - y_train
            loss = jnp.mean((residual)**2)
    
            return loss
    
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
    
        return loss

    # Train the model
    num_epochs = 2000
    
    for epoch in range(num_epochs):
        # Calculate the loss
        loss = train_step(model, optimizer, X_train, y_train)

    # Get predictions
    y_pred = model(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    assert mse < 0.1, f"Final MSE {mse} does not reach below 0.1, indicating that the model does not train as it should."


def test_pde_solving(seed, req_params):

    # Generate Collocation points for PDE
    N = 2**12
    collocs = jnp.array(sobol_sample(np.array([0,-1]), np.array([1,1]), N, seed)) # (4096, 2)
    
    # Generate Collocation points for BCs
    N = 2**6
    BC1_colloc = jnp.array(sobol_sample(np.array([0,-1]), np.array([0,1]), N)) # (64, 2)
    BC1_data = - jnp.sin(np.pi*BC1_colloc[:,1]).reshape(-1,1) # (64, 1)
    BC2_colloc = jnp.array(sobol_sample(np.array([0,-1]), np.array([1,-1]), N)) # (64, 2)
    BC2_data = jnp.zeros(BC2_colloc.shape[0]).reshape(-1,1) # (64, 1)
    BC3_colloc = jnp.array(sobol_sample(np.array([0,1]), np.array([1,1]), N)) # (64, 2)
    BC3_data = jnp.zeros(BC3_colloc.shape[0]).reshape(-1,1) # (64, 1)
    
    # Create lists for BCs
    bc_collocs = [BC1_colloc, BC2_colloc, BC3_colloc]
    bc_data = [BC1_data, BC2_data, BC3_data]


    # Initialize the KAN model
    n_in, n_out, n_hidden = collocs.shape[1], 1, 6
    layer_dims = [n_in, n_hidden, n_hidden, n_out]
    
    model = KAN(layer_dims = layer_dims,
                layer_type = 'chebyshev',
                required_parameters = req_params,
                seed = seed
               )

    # Initialize optimizer
    opt_type = optax.adam(learning_rate=0.001)
    optimizer = nnx.Optimizer(model, opt_type)

    # PDE Loss
    def pde_loss(model, collocs):
        # Eq. parameter
        v = jnp.array(0.01/jnp.pi, dtype=float)
    
        def u(x):
            y = model(x)
            return y
    
        # Physics Loss Terms
        u_t = gradf(u, 0, 1)
        u_x = gradf(u, 1, 1)
        u_xx = gradf(u, 1, 2)
    
        # Residual
        pde_res = u_t(collocs) + model(collocs)*u_x(collocs) - v*u_xx(collocs)
    
        return pde_res
    
    # Define train loop
    @nnx.jit
    def train_step(model, optimizer, collocs, bc_collocs, bc_data):
    
        def loss_fn(model):
            pde_res = pde_loss(model, collocs)
            total_loss = jnp.mean((pde_res)**2)
    
            # Boundary losses
            for idx, colloc in enumerate(bc_collocs):
                # Residual = Model's prediction - Ground Truth
                residual = model(colloc)
                residual -= bc_data[idx]
                # Loss
                total_loss += jnp.mean(residual**2)
    
            return total_loss
    
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
    
        return loss

    # Train the model
    num_epochs = 5000
    
    for epoch in range(num_epochs):
        # Calculate the loss
        loss = train_step(model, optimizer, collocs, bc_collocs, bc_data)
    
    assert loss < 0.1, f"Final Training Loss {loss} does not reach below 0.1, indicating that the model does not train as it should."
    