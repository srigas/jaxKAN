import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jaxkan.models.KAN import KAN
from jaxkan.pikan.pde import (
    gradf,
    get_ac_res,
    get_burgers_res,
    get_kdv_res,
    get_sg_res,
    get_advection_res,
    get_helmholtz_res,
    get_poisson_res,
    get_wave_res,
    get_ks_res,
    get_diffusion_res
)


@pytest.fixture
def simple_model():
    """Create a simple KAN model for testing."""
    return KAN([2, 8, 1], 'spline', {'k': 3, 'G': 5}, 42)


@pytest.fixture
def sample_collocs():
    """Create sample 2D collocation points."""
    return jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])


def test_gradf_first_order(simple_model):
    """Test first-order gradient computation."""
    def u(x):
        return simple_model(x)
    
    u_x = gradf(u, [0])
    
    collocs = jnp.array([[0.5, 0.5]])
    grad_val = u_x(collocs)
    
    assert grad_val.shape == (1, 1), "Gradient shape incorrect"
    assert jnp.all(jnp.isfinite(grad_val)), "Gradient contains non-finite values"


def test_gradf_second_order(simple_model):
    """Test second-order gradient computation."""
    def u(x):
        return simple_model(x)
    
    u_xx = gradf(u, [1, 1])
    
    collocs = jnp.array([[0.5, 0.5]])
    grad_val = u_xx(collocs)
    
    assert grad_val.shape == (1, 1), "Second gradient shape incorrect"
    assert jnp.all(jnp.isfinite(grad_val)), "Second gradient contains non-finite values"


def test_gradf_mixed_partial(simple_model):
    """Test mixed partial derivative computation."""
    def u(x):
        return simple_model(x)
    
    u_tx = gradf(u, [0, 1])  # d²u/dtdx
    
    collocs = jnp.array([[0.5, 0.5]])
    grad_val = u_tx(collocs)
    
    assert grad_val.shape == (1, 1), "Mixed partial shape incorrect"
    assert jnp.all(jnp.isfinite(grad_val)), "Mixed partial contains non-finite values"


def test_gradf_higher_order(simple_model):
    """Test higher-order gradient computation."""
    def u(x):
        return simple_model(x)
    
    u_xxxx = gradf(u, [1, 1, 1, 1])  # Fourth derivative
    
    collocs = jnp.array([[0.5, 0.5]])
    grad_val = u_xxxx(collocs)
    
    assert grad_val.shape == (1, 1), "Fourth derivative shape incorrect"
    assert jnp.all(jnp.isfinite(grad_val)), "Fourth derivative contains non-finite values"


def test_allen_cahn_residual(simple_model, sample_collocs):
    """Test Allen-Cahn equation residual function."""
    ac_res = get_ac_res(D=1e-4, c=5.0)
    residual = ac_res(simple_model, sample_collocs)
    
    assert residual.shape == (3, 1), "Allen-Cahn residual shape incorrect"
    assert jnp.all(jnp.isfinite(residual)), "Allen-Cahn residual contains non-finite values"


def test_allen_cahn_custom_params(simple_model, sample_collocs):
    """Test Allen-Cahn with custom parameters."""
    ac_res = get_ac_res(D=0.001, c=10.0)
    residual = ac_res(simple_model, sample_collocs)
    
    assert jnp.all(jnp.isfinite(residual)), "Custom Allen-Cahn residual contains non-finite values"


def test_burgers_residual(simple_model, sample_collocs):
    """Test Burgers equation residual function."""
    burgers_res = get_burgers_res(nu=0.01)
    residual = burgers_res(simple_model, sample_collocs)
    
    assert residual.shape == (3, 1), "Burgers residual shape incorrect"
    assert jnp.all(jnp.isfinite(residual)), "Burgers residual contains non-finite values"


def test_kdv_residual(simple_model, sample_collocs):
    """Test KdV equation residual function."""
    kdv_res = get_kdv_res(eta=0.97, mu=0.019)
    residual = kdv_res(simple_model, sample_collocs)
    
    assert residual.shape == (3, 1), "KdV residual shape incorrect"
    assert jnp.all(jnp.isfinite(residual)), "KdV residual contains non-finite values"


def test_sine_gordon_residual(simple_model, sample_collocs):
    """Test sine-Gordon equation residual function."""
    sg_res = get_sg_res()
    residual = sg_res(simple_model, sample_collocs)
    
    assert residual.shape == (3, 1), "Sine-Gordon residual shape incorrect"
    assert jnp.all(jnp.isfinite(residual)), "Sine-Gordon residual contains non-finite values"


def test_advection_residual(simple_model, sample_collocs):
    """Test advection equation residual function."""
    advection_res = get_advection_res(c=1.0)
    residual = advection_res(simple_model, sample_collocs)
    
    assert residual.shape == (3, 1), "Advection residual shape incorrect"
    assert jnp.all(jnp.isfinite(residual)), "Advection residual contains non-finite values"


def test_helmholtz_residual(simple_model, sample_collocs):
    """Test Helmholtz equation residual function."""
    helmholtz_res = get_helmholtz_res(k=2.0)
    residual = helmholtz_res(simple_model, sample_collocs)
    
    assert residual.shape == (3, 1), "Helmholtz residual shape incorrect"
    assert jnp.all(jnp.isfinite(residual)), "Helmholtz residual contains non-finite values"


def test_poisson_residual(simple_model, sample_collocs):
    """Test Poisson equation residual function."""
    poisson_res = get_poisson_res(a1=2.0, a2=2.0)
    residual = poisson_res(simple_model, sample_collocs)
    
    assert residual.shape == (3, 1), "Poisson residual shape incorrect"
    assert jnp.all(jnp.isfinite(residual)), "Poisson residual contains non-finite values"


def test_wave_residual(simple_model, sample_collocs):
    """Test wave equation residual function."""
    wave_res = get_wave_res(c=1.5)
    residual = wave_res(simple_model, sample_collocs)
    
    assert residual.shape == (3, 1), "Wave residual shape incorrect"
    assert jnp.all(jnp.isfinite(residual)), "Wave residual contains non-finite values"


def test_kuramoto_sivashinsky_residual(simple_model, sample_collocs):
    """Test Kuramoto-Sivashinsky equation residual function."""
    ks_res = get_ks_res()
    residual = ks_res(simple_model, sample_collocs)
    
    assert residual.shape == (3, 1), "KS residual shape incorrect"
    assert jnp.all(jnp.isfinite(residual)), "KS residual contains non-finite values"


def test_diffusion_residual(simple_model, sample_collocs):
    """Test diffusion equation residual function."""
    diffusion_res = get_diffusion_res(D=0.25)
    residual = diffusion_res(simple_model, sample_collocs)
    
    assert residual.shape == (3, 1), "Diffusion residual shape incorrect"
    assert jnp.all(jnp.isfinite(residual)), "Diffusion residual contains non-finite values"


def test_diffusion_default_params(simple_model, sample_collocs):
    """Test diffusion equation with default parameters."""
    diffusion_res = get_diffusion_res()  # Default D=0.25
    residual = diffusion_res(simple_model, sample_collocs)
    
    assert jnp.all(jnp.isfinite(residual)), "Default diffusion residual contains non-finite values"


def test_diffusion_custom_params(simple_model, sample_collocs):
    """Test diffusion equation with custom parameters."""
    diffusion_res = get_diffusion_res(D=0.5)
    residual = diffusion_res(simple_model, sample_collocs)
    
    assert jnp.all(jnp.isfinite(residual)), "Custom diffusion residual contains non-finite values"


def test_factory_pattern_consistency():
    """Test that all PDE residual factories follow the same pattern."""
    factories = [
        get_ac_res,
        get_burgers_res,
        get_kdv_res,
        get_sg_res,
        get_advection_res,
        get_helmholtz_res,
        get_wave_res,
        get_ks_res,
        get_diffusion_res
    ]
    
    model = KAN([2, 8, 1], 'spline', {'k': 3, 'G': 5}, 42)
    collocs = jnp.array([[0.1, 0.2]])
    
    for factory in factories:
        res_fn = factory()  # Call with default parameters
        residual = res_fn(model, collocs)
        assert residual.shape[0] == collocs.shape[0], f"{factory.__name__} residual batch size incorrect"
        assert residual.shape[1] == 1, f"{factory.__name__} residual output dimension incorrect"
