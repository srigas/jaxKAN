import jax
import jax.numpy as jnp

import optax
from flax import nnx

from scipy.stats.qmc import Sobol
import numpy as np

from .general import adam_transition


def sobol_sample(X0, X1, N, seed=42):
    """
    Performs Sobol sampling.
    
    Args:
        X0 (np.ndarray):
            Lower end of sampling region.
        X1 (np.ndarray):
            Upper end of sampling region.
        N (int):
            Number of points to sample.
        seed (int):
            Seed for reproducibility.
    
    Returns:
        points (np.ndarray):
            Sampled points, shape (N, X0.shape[0])
            
    Example:
        >>> data = sobol_sample(np.array([0,-1]), np.array([1,1]), 2**12, 42)
    """
    dims = X0.shape[0]
    sobol_sampler = Sobol(dims, scramble=True, seed=seed)
    points = sobol_sampler.random_base2(int(np.log2(N)))
    points = X0 + points * (X1 - X0)
    
    return points


def gradf(f, idx, order=1):
    """
    Computes gradients of arbitrary order/argument.
    
    Args:
        f (function):
            Function to be differentiated.
        idx (int):
            Index of coordinate to differentiate.
        order (int):
            Gradient order.
    
    Returns:
        g (function):
            Gradient of f with respect to "idx" and order "order".
            
    Example:
        >>> model = KAN([2,6,1], 'spline', {}, True)
        >>>
        >>> def u(x):
        >>>    y = model(x)
        >>>    return y
        >>>
        >>> u_t = gradf(u, 0, 1) # 1st-order gradient w.r.t. first model feature
        >>> u_xx = gradf(u, 1, 2) # 2nd-order gradient w.r.t. second model feature
    """
    def grad_fn(g, idx):
        return lambda tx: jax.grad(lambda tx: jnp.sum(g(tx)))(tx)[..., idx].reshape(-1,1)

    g = lambda tx: f(tx)
    
    for _ in range(order):
        g = grad_fn(g, idx)
        
    return g


def get_vanilla_loss(pde_loss, model):
    """
    Wrapper that returns the vanilla loss function for a PIKAN.
    
    Args:
        pde_loss (function):
            Loss function corresponding to the PDE.
        model (jaxkan.KAN.KAN):
            jaxKAN KAN model instance.
    
    Returns:
        vanilla_loss_fn (function):
            Full vanilla loss function for the PIKAN.
            
    Example:
        >>> collocs = jnp.array(sobol_sample(np.array([0,-1]), np.array([1,1]), 2**12, 42))
        >>> model = KAN([2,8,1], 'cheby', {'k' : 5}, True)
        >>>
        >>> def pde_loss(model, collocs):
        >>>     def u(x):
        >>>         y = model(x)
        >>>         return y
        >>>
        >>>     u_t = gradf(u, 0, 1)
        >>>     u_xx = gradf(u, 1, 2)
        >>>
        >>>     return u_t(collocs) - 0.001*u_xx(collocs) - 5*(u(collocs)-(u(collocs)**3))
        >>>
        >>> loss_fn = get_vanilla_loss(pde_loss, model)
    """
    
    def vanilla_loss_fn(model, collocs, bc_collocs, bc_data, glob_w, loc_w):
        """
        Full vanilla loss function for a PIKAN.
        
        Args:
            model (jaxkan.KAN.KAN):
                jaxKAN KAN model instance.
            collocs (jnp.array):
                Collocation points for the PDE loss.
            bc_collocs (List[jnp.array]):
                List of collocation points for the boundary losses.
            bc_data (List[jnp.array]):
                List of data corresponding to bc_collocs.
            glob_w (List[jnp.array]):
                Global weights for each loss function's term.
            loc_w (NoneType):
                Placeholder to ensure a uniform train_step() between vanilla and adaptive loss functions.
            
        Returns:
            total_loss (float):
                The total loss function's value.
            None (NoneType):
                Placeholder to ensure a uniform train_step() between vanilla and adaptive loss functions.
        """
        # Calculate PDE loss
        pde_res = pde_loss(model, collocs)
        total_loss = glob_w[0]*jnp.mean((pde_res)**2)

        # Boundary losses
        for idx, colloc in enumerate(bc_collocs):
            # Residual = Model's prediction - Ground Truth
            bc_res = model(colloc) - bc_data[idx]
            # Loss
            total_loss += glob_w[idx+1]*jnp.mean(bc_res**2)
        
        return total_loss, None
        
    return vanilla_loss_fn
    

def get_adaptive_loss(pde_loss, model):
    """
    Wrapper that returns the adaptive loss function for a PIKAN.
    
    Args:
        pde_loss (function):
            Loss function corresponding to the PDE.
        model (jaxkan.KAN.KAN):
            jaxKAN KAN model instance.
    
    Returns:
        adaptive_loss_fn (function):
            Full adaptive loss function for the PIKAN.
            
    Example:
        >>> collocs = jnp.array(sobol_sample(np.array([0,-1]), np.array([1,1]), 2**12, 42))
        >>> model = KAN([2,8,1], 'cheby', {'k' : 5}, True)
        >>>
        >>> def pde_loss(model, collocs):
        >>>     def u(x):
        >>>         y = model(x)
        >>>         return y
        >>>
        >>>     u_t = gradf(u, 0, 1)
        >>>     u_xx = gradf(u, 1, 2)
        >>>
        >>>     return u_t(collocs) - 0.001*u_xx(collocs) - 5*(u(collocs)-(u(collocs)**3))
        >>>
        >>> loss_fn = get_adaptive_loss(pde_loss, model)
    """
    
    def adaptive_loss_fn(model, collocs, bc_collocs, bc_data, glob_w, loc_w):
        """
        Full adaptive loss function for a PIKAN.
        
        Args:
            model (jaxkan.KAN.KAN):
                jaxKAN KAN model instance.
            collocs (jnp.array):
                Collocation points for the PDE loss.
            bc_collocs (List[jnp.array]):
                List of collocation points for the boundary losses.
            bc_data (List[jnp.array]):
                List of data corresponding to bc_collocs.
            glob_w (List[jnp.array]):
                Global weights for each loss function's term.
            loc_w (List[jnp.array]):
                Local RBA weights for each loss function's term.
            
        Returns:
            total_loss (float):
                The total loss function's value.
            new_loc_w (List[jnp.array]):
                Updated RBA weights based on residuals
        """
        # Loss parameter
        eta = jnp.array(0.0001, dtype=float)
        # Placeholder list for RBA weights
        new_loc_w = []

        # Calculate PDE loss
        pde_res = pde_loss(model, collocs)
        
        # New RBA weights
        abs_res = jnp.abs(pde_res)
        loc_w_pde = ((jnp.array(1.0)-eta)*loc_w[0]) + ((eta*abs_res)/jnp.max(abs_res))
        new_loc_w.append(loc_w_pde)
        
        # Weighted Loss
        total_loss = glob_w[0]*jnp.mean((loc_w_pde*pde_res)**2)

        # Boundary losses
        for idx, colloc in enumerate(bc_collocs):
            # Residual = Model's prediction - Ground Truth
            bc_res = model(colloc) - bc_data[idx]

            # New RBA weight
            abs_res = jnp.abs(bc_res)
            loc_w_bc = ((jnp.array(1.0)-eta)*loc_w[idx+1]) + ((eta*abs_res)/jnp.max(abs_res))
            new_loc_w.append(loc_w_bc)
            # Weighted Loss
            total_loss += glob_w[idx+1]*jnp.mean((loc_w_bc*bc_res)**2)
            
        return total_loss, new_loc_w

    return adaptive_loss_fn


def train_PIKAN(model, pde_loss, collocs, bc_collocs, bc_data, glob_w, lr_vals, collect_loss=True, adapt_state=True, loc_w=None, nesterov=True, num_epochs=3001, grid_extend={0: 3}, grid_adapt=[], colloc_adapt={'epochs': []}):
    """
    PIKAN Adaptive training routine.
    
    Args:
        model (jaxkan.KAN.KAN):
            jaxKAN KAN model instance.
        pde_loss (function):
            Loss function corresponding to the PDE.
        collocs (jnp.array):
                Collocation points for the PDE loss.
        bc_collocs (List[jnp.array]):
            List of collocation points for the boundary losses.
        bc_data (List[jnp.array]):
            List of data corresponding to bc_collocs.
        glob_w (List[jnp.array]):
            Global weights for each loss function's term.
        lr_vals (dict):
            Dictionary containing information about the optimizer's scheduler.
        collect_loss (bool):
            Boolean that determines if training loss data are collected or not.
        adapt_state (bool):
            Boolean that determines if adaptive state transition is applied after grid extension.
        loc_w (List[jnp.array]):
            Local RBA weights for each loss function's term.
        nesterov (bool):
            Boolean that determines if Nesterov momentum is used for Adam.
        num_epochs (int):
            Number of training epochs.
        grid_extend (dict):
            Dictionary of epochs during which to perform grid extension and new values for grid.
        grid_adapt (List):
            List of epochs during which to perform grid adaptation.
        colloc_adapt (dict):
            Dictionary containing information about the RDA method.
        
    Returns:
        model, None / train_losses (tuple):
            Trained jaxKAN KAN model instance and values of the loss function per epoch, if collect_loss == True.
        
    Example:
        >>> collocs = jnp.array(sobol_sample(np.array([0,-1]), np.array([1,1]), 2**12, 42))
        >>>
        >>> BC1_colloc = jnp.array(sobol_sample(np.array([0,-1]), np.array([0,1]), 2**6, 42))
        >>> BC1_data = ((BC1_colloc[:,1]**2)*jnp.cos(jnp.pi*BC1_colloc[:,1])).reshape(-1,1)
        >>>
        >>> BC2_colloc = jnp.array(sobol_sample(np.array([0,-1]), np.array([1,-1]), 2**6, 42))
        >>> BC2_data = -jnp.ones(BC2_colloc.shape[0]).reshape(-1,1)
        >>>
        >>> BC3_colloc = jnp.array(sobol_sample(np.array([0,1]), np.array([1,1]), 2**6, 42))
        >>> BC3_data = -jnp.ones(BC3_colloc.shape[0]).reshape(-1,1)
        >>>
        >>> bc_collocs = [BC1_colloc, BC2_colloc, BC3_colloc]
        >>> bc_data = [BC1_data, BC2_data, BC3_data]
        >>>
        >>> model = KAN([2,8,8,1], 'spline', {'k': 4, 'G': 3, 'grid_e': 0.02}, True)
        >>>
        >>> def pde_loss(model, collocs):
        >>>     def u(x):
        >>>         y = model(x)
        >>>         return y
        >>>
        >>>     u_t = gradf(u, 0, 1)
        >>>     u_xx = gradf(u, 1, 2)
        >>>
        >>>     pde_res = u_t(collocs) - 0.001*u_xx(collocs) - 5.0*(u(collocs)-(u(collocs)**3))
        >>>
        >>>     return pde_res
        >>>
        >>> num_epochs = 50000
        >>> lr_vals = {'init_lr' : 0.001, 'scales' : {0 : 1.0, 15_000 : 0.6, 25_000 : 0.8}}
        >>>
        >>> grid_extend = {0 : 3, 5000 : 8, 20_000 : 12}
        >>> grid_adapt = []
        >>>
        >>> glob_w = [jnp.array(1.0, dtype=float)]*4
        >>> loc_w = [jnp.ones((collocs.shape[0],1)), jnp.ones((BC1_colloc.shape[0],1)),
        >>>          jnp.ones((BC2_colloc.shape[0],1)), jnp.ones((BC3_colloc.shape[0],1))]
        >>>
        >>> colloc_adapt = {'lower_point' : np.array([0,-1]), 'upper_point' : np.array([1,1]),
        >>>                 'M' : 2**16, 'k' : jnp.array(1.0, dtype=float), 'c' : jnp.array(1.0, dtype=float),
        >>>                 'epochs' : [10_000, 20_000]}
        >>>
        >>> model, train_losses = train_PIKAN(model, pde_loss, collocs, bc_collocs, bc_data, glob_w=glob_w, 
        >>>                                   lr_vals=lr_vals, collect_loss=True, adapt_state=True, loc_w=loc_w,
        >>>                                   nesterov=True, num_epochs=num_epochs, grid_extend=grid_extend,
        >>>                                   grid_adapt=grid_adapt, colloc_adapt=colloc_adapt)
    """
    # Setup scheduler for optimizer
    scheduler = optax.piecewise_constant_schedule(
        init_value=lr_vals['init_lr'],
        boundaries_and_scales=lr_vals['scales']
    )

    opt_type = optax.adam(learning_rate=scheduler, nesterov=nesterov)

    # Define the optimizer
    optimizer = nnx.Optimizer(model, opt_type)

    # Define loss function
    if loc_w is None:
        loss_fn = get_vanilla_loss(pde_loss, model)
    else:
        loss_fn = get_adaptive_loss(pde_loss, model)

    # Define train loop
    @nnx.jit
    def train_step(model, optimizer, collocs, bc_collocs, bc_data, glob_w, loc_w):
        
        (loss, loc_w), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, collocs, bc_collocs, bc_data, glob_w, loc_w)
        optimizer.update(grads)

        return loss, loc_w

    # Check if we perform adaptive collocation points resampling
    if colloc_adapt['epochs']:
        # Define the function
        def resample_collocs(model, collocs, sample, k=jnp.array(1.0), c=jnp.array(1.0)):
            # Calculate residuals of PDE
            resids = jnp.abs(pde_loss(model, sample))
            # Raise to power k
            ek = jnp.power(resids, k)
            # Divide by mean and add c
            px = (ek/jnp.mean(ek)) + c
            # Normalize
            px_norm = (px / jnp.sum(px))[:,0]
            # Draw ids for the sampled points using px_norm
            X_ids = jax.random.choice(key=jax.random.PRNGKey(0), a=sample.shape[0], shape=(collocs.shape[0],), replace=False, p=px_norm)
            # Replace collocation points
            new_collocs = sample[X_ids]
        
            return new_collocs

        # Sample M points from Sobol
        M = colloc_adapt['M']
        lp = colloc_adapt['lower_point']
        up = colloc_adapt['upper_point']
        sample = jnp.array(sobol_sample(lp, up, M))
        # Draw k, c hyperparameters
        k, c = colloc_adapt['k'], colloc_adapt['c']

    if collect_loss:
        # Initialize train_losses
        train_losses = jnp.zeros((num_epochs,))

    # Start training
    update_val = grid_extend[0]
    
    for epoch in range(num_epochs):
        # Check if we have to adapt collocation points
        if epoch in colloc_adapt['epochs']:
            # Get new colloc points
            collocs = resample_collocs(model, collocs, sample, k, c)
        
            # Restart training loc_w for the PDE if we're doing RBA
            if loc_w is not None:
                new_loc_w = jnp.full_like(loc_w[0], jnp.mean(loc_w[0]))
                loc_w[0] = new_loc_w
                
            # A grid adaptation is necessary for base/spline after resampling
            if model.layer_type in ['base', 'spline']:
                G = model.layers[0].grid.G
                model.update_grids(collocs, G)
            
        # Check if we're in an adapt epoch - this pertains only to base/spline layers
        if (epoch in grid_adapt) and (model.layer_type in ['base', 'spline']):
            model.update_grids(collocs, update_val)
            
        # Check if we're in an extension epoch
        if epoch in grid_extend.keys():
            # If the layer is not base/spline type, then there should be no update on epoch 0
            if (model.layer_type not in ['base', 'spline']) and (epoch == 0):
                pass
            else:
                print(f"Epoch {epoch}: Performing grid update")
                # Get grid size
                update_val = grid_extend[epoch]
                # Perform the update
                model.update_grids(collocs, update_val)
        
                # Optimizer Transition
                if adapt_state:
                    _, model_state = nnx.split(model)
                    old_state = optimizer.opt_state
                    adam_transition(old_state, model_state)
                else:
                    optimizer = nnx.Optimizer(model, opt_type)
            
        # Calculate the loss
        loss, loc_w = train_step(model, optimizer, collocs, bc_collocs, bc_data, glob_w, loc_w)

        if collect_loss:
            # Append the loss
            train_losses = train_losses.at[epoch].set(loss)
    
    if collect_loss:
        return model, train_losses
    else:
        return model, None
