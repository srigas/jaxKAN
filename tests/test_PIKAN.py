import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jaxkan.utils.PIKAN import sobol_sample, gradf, get_vanilla_loss, get_adaptive_loss, train_PIKAN
from jaxkan.KAN import KAN


@pytest.fixture
def sobol_data():
    
    X0 = np.array([0, -1])
    X1 = np.array([1, 1])
    N = 2**12
    
    return X0, X1, N


def test_sobol_sample(sobol_data):
    
    X0, X1, N = sobol_data
    points = sobol_sample(X0, X1, N, seed=42)
    
    assert points.shape == (N, X0.shape[0]), "Sobol sampling returned incorrect shape"
    assert np.all(points >= X0) and np.all(points <= X1), "Sobol sampling points out of bounds"


def test_gradf():
    def f(x):
        return x[0]**3 + 5*x[0]*x[1]
    
    g_1 = gradf(f, idx=0, order=1)
    g_2 = gradf(f, idx=1, order=1)
    x = jnp.array([2.0, 1.0])
    
    assert jnp.allclose(g_1(x), 17.0), "Gradient function returned incorrect value"
    assert jnp.allclose(g_2(x), 10.0), "Gradient function returned incorrect value"


@pytest.fixture
def pde_loss():
    
    def loss_fn(model, collocs):
        def u(x):
            return model(x)
        u_t = gradf(u, idx=0, order=1)
        u_xx = gradf(u, idx=1, order=2)
        return u_t(collocs) - 0.001 * u_xx(collocs) - 5 * (u(collocs) - u(collocs) ** 3)
        
    return loss_fn


@pytest.fixture
def model():
    return KAN([2, 6, 1], 'spline', {}, True)


@pytest.fixture
def training_data(sobol_data):
    
    X0, X1, N = sobol_data
    collocs = jnp.array(sobol_sample(X0, X1, N, seed=42))
    bc_collocs = [collocs[:10], collocs[10:20]]
    bc_data = [jnp.ones((10, 1)), jnp.zeros((10, 1))]
    glob_w = [jnp.array(1.0), jnp.array(0.5), jnp.array(0.5)]
    
    return collocs, bc_collocs, bc_data, glob_w


def test_get_vanilla_loss(pde_loss, model, training_data):
    
    collocs, bc_collocs, bc_data, glob_w = training_data
    loss_fn = get_vanilla_loss(pde_loss, model)
    loss, _ = loss_fn(model, collocs, bc_collocs, bc_data, glob_w, None)
    
    assert isinstance(loss, jnp.ndarray), "Vanilla loss function did not return a JAX array"
    assert loss.shape == (), "Vanilla loss function returned incorrect shape"
    assert loss.dtype == jnp.float32, "Vanilla loss function returned incorrect dtype"


def test_get_adaptive_loss(pde_loss, model, training_data):
    
    collocs, bc_collocs, bc_data, glob_w = training_data
    loc_w = [jnp.ones((collocs.shape[0],1)), jnp.ones((bc_collocs[0].shape[0],1)), jnp.ones((bc_collocs[1].shape[0],1))]
    loss_fn = get_adaptive_loss(pde_loss, model)
    loss, loc_w_new = loss_fn(model, collocs, bc_collocs, bc_data, glob_w, loc_w)
    
    assert isinstance(loss, jnp.ndarray), "Adaptive loss function did not return a JAX array"
    assert loss.shape == (), "Adaptive loss function returned incorrect shape"
    assert loss.dtype == jnp.float32, "Adaptive loss function returned incorrect dtype"
    
    assert len(loc_w_new) == len(loc_w), "Adaptive loss function returned incorrect RBA weights"
    for idx,w in enumerate(loc_w_new):
        assert isinstance(w, jnp.ndarray), "RBA weight is not a JAX array"
        assert w.shape == loc_w[idx].shape, "RBA weight has incorrect shape"


def test_train_PIKAN(pde_loss, model, training_data):
    
    collocs, bc_collocs, bc_data, glob_w = training_data
    lr_vals = {'init_lr': 0.001, 'scales': {0: 1.0, 100: 0.8}}
    loc_w = [jnp.ones((collocs.shape[0],1)), jnp.ones((bc_collocs[0].shape[0],1)), jnp.ones((bc_collocs[1].shape[0],1))]
    num_epochs = 10
    grid_extend = {0: 3, 5: 5}
    colloc_adapt = {'epochs': [], 'lower_point': np.array([0, -1]), 'upper_point': np.array([1, 1]), 'M': 2**10, 'k': jnp.array(1.0), 'c': jnp.array(1.0)}

    trained_model, train_losses = train_PIKAN(
        model, pde_loss, collocs, bc_collocs, bc_data, glob_w, lr_vals,
        collect_loss=True, adapt_state=True, loc_w=loc_w, nesterov=True,
        num_epochs=num_epochs, grid_extend=grid_extend, grid_adapt=[], colloc_adapt=colloc_adapt
    )
    
    assert train_losses.shape == (num_epochs,), "Training did not return the correct number of loss values"
    assert isinstance(trained_model, KAN), "Training did not return a valid model"
