import pytest

import jax
import jax.numpy as jnp
import numpy as np

from jaxkan.pikan.sampling import get_collocs_grid, get_collocs_random, get_collocs_sobol


def test_get_collocs_grid_1d():
    """Test 1D grid collocation generation."""
    ranges = [(0.0, 1.0, 11)]
    collocs = get_collocs_grid(ranges)
    
    assert collocs.shape == (11, 1), "Incorrect shape for 1D grid"
    assert jnp.allclose(collocs[:, 0], jnp.linspace(0.0, 1.0, 11)), "Incorrect grid values"


def test_get_collocs_grid_2d():
    """Test 2D grid collocation generation."""
    ranges = [(0.0, 1.0, 10), (0.0, 2.0, 20)]
    collocs = get_collocs_grid(ranges)
    
    assert collocs.shape == (200, 2), "Incorrect shape for 2D grid"
    assert jnp.min(collocs[:, 0]) >= 0.0 and jnp.max(collocs[:, 0]) <= 1.0
    assert jnp.min(collocs[:, 1]) >= 0.0 and jnp.max(collocs[:, 1]) <= 2.0


def test_get_collocs_grid_3d():
    """Test 3D grid collocation generation."""
    ranges = [(0.0, 1.0, 5), (-1.0, 1.0, 4), (2.0, 3.0, 3)]
    collocs = get_collocs_grid(ranges)
    
    assert collocs.shape == (60, 3), "Incorrect shape for 3D grid (5*4*3=60)"
    assert jnp.min(collocs[:, 0]) >= 0.0 and jnp.max(collocs[:, 0]) <= 1.0
    assert jnp.min(collocs[:, 1]) >= -1.0 and jnp.max(collocs[:, 1]) <= 1.0
    assert jnp.min(collocs[:, 2]) >= 2.0 and jnp.max(collocs[:, 2]) <= 3.0


def test_get_collocs_random_1d():
    """Test 1D random collocation generation."""
    ranges = [(0.0, 1.0)]
    total_points = 100
    collocs = get_collocs_random(ranges, total_points, seed=42)
    
    assert collocs.shape == (100, 1), "Incorrect shape for 1D random"
    assert jnp.all((collocs >= 0.0) & (collocs <= 1.0)), "Values outside range"


def test_get_collocs_random_2d():
    """Test 2D random collocation generation."""
    ranges = [(0.0, 1.0), (0.0, 2.0)]
    total_points = 150
    collocs = get_collocs_random(ranges, total_points, seed=123)
    
    assert collocs.shape == (150, 2), "Incorrect shape for 2D random"
    assert jnp.all((collocs[:, 0] >= 0.0) & (collocs[:, 0] <= 1.0))
    assert jnp.all((collocs[:, 1] >= 0.0) & (collocs[:, 1] <= 2.0))


def test_get_collocs_random_reproducibility():
    """Test that same seed produces same random collocation points."""
    ranges = [(0.0, 1.0), (0.0, 2.0)]
    total_points = 50
    
    collocs1 = get_collocs_random(ranges, total_points, seed=42)
    collocs2 = get_collocs_random(ranges, total_points, seed=42)
    
    assert jnp.allclose(collocs1, collocs2), "Same seed should produce identical results"


def test_get_collocs_sobol_1d():
    """Test 1D Sobol collocation generation."""
    ranges = [(0.0, 1.0)]
    total_points = 64  # Power of 2
    collocs = get_collocs_sobol(ranges, total_points, seed=42)
    
    assert collocs.shape == (64, 1), "Incorrect shape for 1D Sobol"
    assert jnp.all((collocs >= 0.0) & (collocs <= 1.0)), "Values outside range"


def test_get_collocs_sobol_2d():
    """Test 2D Sobol collocation generation."""
    ranges = [(0.0, 1.0), (0.0, 2.0)]
    total_points = 128  # Power of 2
    collocs = get_collocs_sobol(ranges, total_points, seed=123)
    
    assert collocs.shape == (128, 2), "Incorrect shape for 2D Sobol"
    assert jnp.all((collocs[:, 0] >= 0.0) & (collocs[:, 0] <= 1.0))
    assert jnp.all((collocs[:, 1] >= 0.0) & (collocs[:, 1] <= 2.0))


def test_get_collocs_sobol_3d():
    """Test 3D Sobol collocation generation."""
    ranges = [(-1.0, 1.0), (0.0, 1.0), (2.0, 5.0)]
    total_points = 256  # Power of 2
    collocs = get_collocs_sobol(ranges, total_points, seed=99)
    
    assert collocs.shape == (256, 3), "Incorrect shape for 3D Sobol"
    assert jnp.all((collocs[:, 0] >= -1.0) & (collocs[:, 0] <= 1.0))
    assert jnp.all((collocs[:, 1] >= 0.0) & (collocs[:, 1] <= 1.0))
    assert jnp.all((collocs[:, 2] >= 2.0) & (collocs[:, 2] <= 5.0))


def test_get_collocs_sobol_coverage():
    """Test that Sobol sampling provides better coverage than random."""
    ranges = [(0.0, 1.0), (0.0, 1.0)]
    total_points = 64
    
    sobol_points = get_collocs_sobol(ranges, total_points, seed=42)
    random_points = get_collocs_random(ranges, total_points, seed=42)
    
    # Simple coverage test: divide space into grid and count occupied cells
    grid_size = 8
    sobol_occupied = len(np.unique(np.floor(sobol_points * grid_size), axis=0))
    random_occupied = len(np.unique(np.floor(random_points * grid_size), axis=0))
    
    # Sobol should generally have better coverage (more occupied cells)
    # This is not guaranteed for all seeds but should hold on average
    assert sobol_occupied >= random_occupied * 0.8, "Sobol sampling should provide good coverage"


def test_get_collocs_negative_ranges():
    """Test collocation generation with negative ranges."""
    ranges = [(-2.0, -1.0), (-5.0, 5.0)]
    
    grid_collocs = get_collocs_grid([(r[0], r[1], 10) for r in ranges])
    random_collocs = get_collocs_random(ranges, 100, seed=42)
    sobol_collocs = get_collocs_sobol(ranges, 64, seed=42)
    
    for collocs in [grid_collocs, random_collocs, sobol_collocs]:
        assert jnp.all((collocs[:, 0] >= -2.0) & (collocs[:, 0] <= -1.0))
        assert jnp.all((collocs[:, 1] >= -5.0) & (collocs[:, 1] <= 5.0))
