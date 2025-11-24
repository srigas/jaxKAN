
import jax.numpy as jnp
from jax import random

import numpy as np

from scipy.stats.qmc import Sobol


def get_collocs_grid(ranges: list[tuple[float, float, int]]):
    """
    Generate grid-based collocation points across arbitrary dimensions.

    Args:
        ranges (list[tuple[float, float, int]]):
            List of tuples where each tuple contains (lower_bound, upper_bound, sample_size) for each dimension.
        
    Returns:
        jnp.array:
            Array of shape (total_points, n_dims) containing all grid points.
        
    Example:
        >>> # 2D grid: x in [0, 1] with 10 points, t in [0, 2] with 20 points
        >>> collocs = pde_collocs_grid(ranges=[(0.0, 1.0, 10), (0.0, 2.0, 20)])
    """
    
    # Create 1D arrays for each dimension
    grids_1d = [jnp.linspace(r[0], r[1], r[2]) for r in ranges]
    
    # Create meshgrid for arbitrary dimensions
    meshgrids = jnp.meshgrid(*grids_1d, indexing='ij')
    
    # Flatten and stack all dimensions
    collocs_pool = jnp.stack([grid.flatten() for grid in meshgrids], axis=1)

    return collocs_pool


def get_collocs_random(ranges: list[tuple[float, float]], total_points: int, seed: int = 42):
    """
    Generate random collocation points across arbitrary dimensions.

    Args:
        ranges (list[tuple[float, float]]):
            List of tuples where each tuple contains (lower_bound, upper_bound) for each dimension.
        total_points (int):
            Total number of random points to generate.
        seed (int):
            Random seed for reproducibility.
        
    Returns:
        jnp.array:
            Array of shape (total_points, n_dims) containing randomly sampled points.
        
    Example:
        >>> # 2D random samples: 100 points in [0, 1] x [0, 2]
        >>> collocs = pde_collocs_random(ranges=[(0.0, 1.0), (0.0, 2.0)], total_points=100, seed=42)
    """
    
    key = random.PRNGKey(seed)
    
    # Generate random samples for each dimension
    samples = []
    for r in ranges:
        key, subkey = random.split(key)
        samples.append(random.uniform(subkey, (total_points,), minval=r[0], maxval=r[1]))
    
    # Stack all dimensions
    collocs_pool = jnp.stack(samples, axis=1)
    
    return collocs_pool


def get_collocs_sobol(ranges: list[tuple[float, float]], total_points: int, seed: int = 42):
    """
    Generate Sobol quasi-random collocation points across arbitrary dimensions.

    Args:
        ranges (list[tuple[float, float]]):
            List of tuples where each tuple contains (lower_bound, upper_bound) for each dimension.
        total_points (int):
            Total number of Sobol points to generate. Should be a power of 2 for optimal coverage.
        seed (int):
            Random seed for reproducibility. Default is 42.
        
    Returns:
        jnp.array:
            Array of shape (total_points, n_dims) containing Sobol sampled points.
        
    Example:
        >>> # 2D Sobol samples: 1024 points in [0, 1] x [0, 2]
        >>> collocs = pde_collocs_sobol(ranges=[(0.0, 1.0), (0.0, 2.0)], total_points=1024, seed=42)
    """
    
    # Extract bounds
    dims = len(ranges)
    X0 = np.array([r[0] for r in ranges])
    X1 = np.array([r[1] for r in ranges])
    
    # Initialize Sobol sampler
    sobol_sampler = Sobol(dims, scramble=True, seed=seed)
    
    # Generate Sobol points
    points = sobol_sampler.random_base2(int(np.log2(total_points)))
    
    # Scale to the specified ranges
    points = X0 + points * (X1 - X0)
    
    # Convert to JAX array
    collocs_pool = jnp.array(points)
    
    return collocs_pool
    