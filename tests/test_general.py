import jax
import jax.numpy as jnp

from jaxkan.layers.utils import solve_single_lstsq, solve_full_lstsq, interpolate_moments

def test_solve_single_lstsq():
    
    A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
    B = jnp.array([[1.0], [2.0]])
    X_expected = jnp.linalg.lstsq(A, B, rcond=None)[0]
    
    X_actual = solve_single_lstsq(A, B)
    
    assert jnp.allclose(X_actual, X_expected), "solve_single_lstsq failed"

def test_solve_full_lstsq():
    
    A = jnp.array([[[2.0, 1.0], [1.0, 3.0]], [[1.0, 2.0], [2.0, 1.0]]])
    B = jnp.array([[[1.0], [2.0]], [[2.0], [3.0]]])
    X_expected = jnp.array([jnp.linalg.lstsq(A[i], B[i], rcond=None)[0] for i in range(A.shape[0])])
    
    X_actual = solve_full_lstsq(A, B)
    
    assert jnp.allclose(X_actual, X_expected), "solve_full_lstsq failed"

def test_interpolate_moments():
    
    mu_old = jnp.array([[1, 2, 3], [4, 5, 6]])
    nu_old = jnp.array([[7, 8, 9], [10, 11, 12]])
    new_shape = (2, 5)
    
    mu_new, nu_new = interpolate_moments(mu_old, nu_old, new_shape)
    
    assert mu_new.shape == new_shape, "Interpolated mu has incorrect shape"
    assert nu_new.shape == new_shape, "Interpolated nu has incorrect shape"
