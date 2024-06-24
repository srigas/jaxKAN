import jax
import jax.numpy as jnp
from jax import vmap

import optax
from flax import linen as nn

from scipy.stats.qmc import Sobol
import numpy as np

import time


def interpolate_moments(mu_old, nu_old, new_shape):
    '''
        Performs a linear interpolation to assign values to the first and second-order moments of gradients
        of the c_i basis functions coefficients after grid extension
        Note:
            num_basis = G+k for splines and G+1 for R basis functions
            new_num_basis = G'+k for splines and G'+1 for R basis functions
        
        Args:
        -----
            mu_old (jnp.array): first-order moments before extension
                shape (n_in*n_out, num_basis)
            nu_old (jnp.array): second-order moments before extension
                shape (n_in*n_out, num_basis)
            new shape (tuple): (n_in*n_out, new_num_basis)
        
        Returns:
        --------
            mu_new (jnp.array): first-order moments after extension
                shape (n_in*n_out, new_num_basis)
            nu_new (jnp.array): second-order moments after extension
                shape (n_in*n_out, new_num_basis)
    '''
    old_shape = mu_old.shape
    size = old_shape[0]
    old_j = old_shape[1]
    new_j = new_shape[1]
    
    # Create new indices for the second dimension
    old_indices = jnp.linspace(0, old_j - 1, old_j)
    new_indices = jnp.linspace(0, old_j - 1, new_j)

    # Vectorize the interpolation function for use with vmap
    interpolate_fn = lambda old_row: jnp.interp(new_indices, old_indices, old_row)

    # Apply the interpolation function to each row using vmap
    mu_new = vmap(interpolate_fn)(mu_old)
    nu_new = vmap(interpolate_fn)(nu_old)
    
    return mu_new, nu_new


@jax.jit
def state_transition(old_state, variables):
    '''
        Performs the state transition for the optimizer after grid extension
        
        Args:
        -----
            old_state (tuple): collection of adam state and scheduler state before extension
            variables (dict): variables dict of KAN model
        
        Returns:
        --------
            new_state (tuple): collection of adam state and scheduler state after extension
    '''
    # Copy old state
    adam_count = old_state[0].count
    #adam_count = jnp.array(0, dtype=jnp.int32)
    adam_mu, adam_nu = old_state[0].mu, old_state[0].nu

    # Get all layer-related keys, so that we do not touch the other parameters
    layer_keys = {k for k in adam_mu.keys() if k.startswith('layers_')}

    for key in layer_keys:
        # Find the c_basis shape for this layer
        c_shape = variables['params'][key]['c_basis'].shape
        # Get new mu and nu
        mu_new, nu_new = interpolate_moments(adam_mu[key]['c_basis'], adam_nu[key]['c_basis'], c_shape)
        # Set them
        adam_mu[key]['c_basis'], adam_nu[key]['c_basis'] = mu_new, nu_new

    # Make new adam state
    adam_state = optax.ScaleByAdamState(adam_count, adam_mu, adam_nu)
    # Make new scheduler state
    extra_state = optax.ScaleByScheduleState(adam_count)
    # Make new total state
    new_state = (adam_state, extra_state)

    return new_state


def sobol_sample(X0, X1, N, seed=42):
    '''
        Performs Sobol sampling
        
        Args:
        -----
            X0 (np.ndarray): lower end of sampling region
                shape (dims,)
            X1 (np.ndarray): upper end of sampling region
                shape (dims,)
            N (int): number of points to sample
            seed (int): seed for reproducibility
        
        Returns:
        --------
            points (np.ndarray): sampled points
                shape (N,dims)
    '''
    dims = X0.shape[0]
    sobol_sampler = Sobol(dims, scramble=True, seed=seed)
    points = sobol_sampler.random_base2(int(np.log2(N)))
    points = X0 + points * (X1 - X0)
    
    return points


def gradf(f, idx, order=1):
    '''
        Computes gradients of arbitrary order
        
        Args:
        -----
            f (function): function to be differentiated
            idx (int): index of coordinate to differentiate
            order (int): gradient order
        
        Returns:
        --------
            g (function): gradient of f
    '''
    def grad_fn(g, idx):
        return lambda tx: jax.grad(lambda tx: jnp.sum(g(tx)))(tx)[..., idx].reshape(-1,1)

    g = lambda tx: f(tx)
    
    for _ in range(order):
        g = grad_fn(g, idx)
        
    return g


def get_vanilla_loss(pde_loss, model, variables):
    '''
        Wrapper that returns the loss function for a vanilla PIKAN
        
        Args:
        -----
            pde_loss (function): loss function corresponding to the PDE
            model: model from the models module
            variables (dict): variables dict of KAN model
        
        Returns:
        --------
            vanilla_loss_fn (function): loss function for the PIKAN
    '''
    
    @jax.jit
    def vanilla_loss_fn(params, collocs, bc_collocs, bc_data, glob_w, loc_w, state):
        '''
            Loss function for a vanilla PIKAN
            
            Args:
            -----
                params (dict): trainable parameters of the model
                collocs (jnp.array): collocation points for the PDE loss
                bc_collocs (List[jnp.array]): list of collocation points for the boundary losses
                bc_data (List[jnp.array]): list of data corresponding to bc_collocs
                glob_w (List[jnp.array]): global weights for each loss function's term
                loc_w (NoneType): placeholder to ensure a uniform train_step()
                state (dict): non-trainable parameters of the model
                
            Returns:
            --------
                total_loss (float): the loss function's value
                None (NoneType): placeholder to ensure a uniform train_step()
        '''     
        # Calculate PDE loss
        pde_res = pde_loss(params, collocs, state)
        total_loss = glob_w[0]*jnp.mean((pde_res)**2)
        
        # Define the model function
        variables = {'params' : params, 'state' : state}
        
        def u(vec_x):
            y, spl = model.apply(variables, vec_x)
            return y
            
        # Boundary Losses
        for idx, colloc in enumerate(bc_collocs):
            # Residual = Model's prediction - Ground Truth
            residual = u(colloc) - bc_data[idx]
            # Loss
            total_loss += glob_w[idx+1]*jnp.mean((residual)**2)

        # return a tuple to be the same as adaptive_loss_fn and thus be able
        # to define a common train step
        return total_loss, None
        
    return vanilla_loss_fn


def get_adapt_loss(pde_loss, model, variables):
    '''
        Wrapper that returns the loss function for an adaptive PIKAN
        
        Args:
        -----
            pde_loss (function): loss function corresponding to the PDE
            model: model from the models module
            variables (dict): variables dict of KAN model
        
        Returns:
        --------
            adaptive_loss_fn (function): loss function for the PIKAN
    '''
    
    @jax.jit
    def adaptive_loss_fn(params, collocs, bc_collocs, bc_data, glob_w, loc_w, state):
        '''
            Loss function for an adaptive PIKAN
            
            Args:
            -----
                params (dict): trainable parameters of the model
                collocs (jnp.array): collocation points for the PDE loss
                bc_collocs (List[jnp.array]): list of collocation points for the boundary losses
                bc_data (List[jnp.array]): list of data corresponding to bc_collocs
                glob_w (List[jnp.array]): global weights for each loss function's term
                loc_w (List[jnp.array]): local RBA weights for each loss function's term
                state (dict): non-trainable parameters of the model
                
            Returns:
            --------
                total_loss (float): the loss function's value
                loc_w (List[jnp.array]): updated RBA weights based on residuals
        '''
        # Loss parameter
        eta = jnp.array(0.0001, dtype=float)
        # Placeholder list for RBA weights
        new_loc_w = []
    
        # Calculate PDE loss
        pde_res = pde_loss(params, collocs, state)
        
        # New RBA weights
        abs_res = jnp.abs(pde_res)
        loc_w_pde = ((jnp.array(1.0)-eta)*loc_w[0]) + ((eta*abs_res)/jnp.max(abs_res))
        new_loc_w.append(loc_w_pde)
        
        # Weighted Loss
        total_loss = glob_w[0]*jnp.mean((loc_w_pde*pde_res)**2)
    
        # Define the model function
        variables = {'params' : params, 'state' : state}
        
        def u(vec_x):
            y, spl = model.apply(variables, vec_x)
            return y
    
        # Boundary Losses
        for idx, colloc in enumerate(bc_collocs):
            # Residual = Model's prediction - Ground Truth
            bc_res = u(colloc) - bc_data[idx]
            # New RBA weight
            abs_res = jnp.abs(bc_res)
            loc_w_bc = ((jnp.array(1.0)-eta)*loc_w[idx+1]) + ((eta*abs_res)/jnp.max(abs_res))
            new_loc_w.append(loc_w_bc)
            # Weighted Loss
            total_loss += glob_w[idx+1]*jnp.mean((loc_w_bc*bc_res)**2)
    
        return total_loss, new_loc_w

    return adaptive_loss_fn


def train_PIKAN(model, variables, pde_loss, collocs, bc_collocs, bc_data, glob_w, lr_vals, adapt_state=True, loc_w=None, nesterov=True, num_epochs = 3001, grid_extend={0 : 3}, grid_adapt=[]):
    '''
        Training routine for a PIKAN
        
        Args:
        -----
            model: model from the models module
            variables (dict): dict containing the params and state dicts
            pde_loss (function): loss function corresponding to the PDE
            collocs (jnp.array): collocation points for the PDE loss
            bc_collocs (List[jnp.array]): list of collocation points for the boundary losses
            bc_data (List[jnp.array]): list of data corresponding to bc_collocs
            glob_w (List[jnp.array]): global weights for each loss function's term
            lr_vals (dict): dict containing information about the scheduler
            adapt_state (bool): boolean that determines if adaptive state transition is applied during grid extension
            loc_w (List[jnp.array]): local RBA weights for each loss function's term
            nesterov (bool): boolean that determines if Nesterov momentum is used for Adam
            num_epochs (int): number of training epochs
            grid_extend (dict): dict of epochs during which to perform grid extension and new values of G
            grid_adapt (List): list of epochs during which to perform grid adaptation
            colloc_adapt (dict): dict containing information about the RDA method
            
        Returns:
        --------
            model: trained model
            variables (dict): dict containing the params and state dicts of the trained model
            train_losses (jnp.array): values of the loss function per epoch
                shape (num_epochs,)
    '''
    # Setup scheduler for optimizer
    scheduler = optax.piecewise_constant_schedule(
        init_value=lr_vals['init_lr'],
        boundaries_and_scales=lr_vals['scales']
    )
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=scheduler, nesterov=nesterov)
    # Initialize optimizer state
    opt_state = optimizer.init(variables['params'])

    # Define loss function
    if loc_w is None:
        loss_fn = get_vanilla_loss(pde_loss, model, variables)
    else:
        loss_fn = get_adapt_loss(pde_loss, model, variables)

    # Define train loop
    @jax.jit
    def train_step(params, opt_state, collocs, bc_collocs, bc_data, glob_w, loc_w, state):
        
        (loss, loc_w), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, collocs, bc_collocs, bc_data, glob_w, loc_w, state)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        new_variables = {'params': params, 'state': state}
        
        return new_variables, opt_state, loss, loc_w

    # Check if we perform adaptive collocation points resampling
    if colloc_adapt['epochs']:
        # Define the function
        @jax.jit
        def resample_collocs(variables, collocs, sample, k=jnp.array(1.0), c=jnp.array(1.0)):
            # Calculate residuals of PDE
            resids = jnp.abs(pde_loss(variables['params'], sample, variables['state']))
            # Raise to power k
            ek = jnp.power(resids, k)
            # Divide by mean and add c
            px = (ek/jnp.mean(ek)) + c
            # Normalize
            px_norm = (px / jnp.sum(px))[:,0]
            # Draw ids for the sampled points using px_norm
            # Note that they key can be added as a function argument, for now there's no reason to
            X_ids = jax.random.choice(key=jax.random.PRNGKey(0), a=sample.shape[0], shape=(collocs.shape[0],), replace=False, p=px_norm)
            # Replace collocation points
            new_collocs = sample[X_ids]
        
            return new_collocs
    
        # Sample M points from Sobol
        M = colloc_adapt['M']
        sample = jnp.array(sobol_sample(np.array([0,-1]), np.array([1,1]), M))
        # Draw k, c hyperparameters
        k, c = colloc_adapt['k'], colloc_adapt['c']

    # Initialize train_losses
    train_losses = jnp.zeros((num_epochs,))

    # Start training
    start_time=time.time()
    G_val = grid_extend[0]
    
    for epoch in range(num_epochs):
        # Check if we have to adapt collocation points
        if epoch in colloc_adapt['epochs']:
            # Get new colloc points
            collocs = resample_collocs(variables, collocs, sample, k, c)
            # Restart training loc_w for the PDE if we're doing RBA
            if loc_w is not None:
                new_loc_w = jnp.full_like(loc_w[0], jnp.mean(loc_w[0]))
                loc_w[0] = new_loc_w
            # Adapt grid to new collocation points
            updated_variables = model.apply(variables, collocs, G_val, method=model.update_grids)
            variables = updated_variables.copy()
        # Check if we're in an update epoch
        if epoch in grid_adapt:
            updated_variables = model.apply(variables, collocs, G_val, method=model.update_grids)
            variables = updated_variables.copy()
        # Check if we're in an extension epoch
        if epoch in grid_extend.keys():
            print(f"Epoch {epoch}: Performing grid update")
            # Get grid size
            G_val = grid_extend[epoch]
            # Perform the update
            updated_variables = model.apply(variables, collocs, G_val, method=model.update_grids)
            variables = updated_variables.copy()
            # Optimizer Transition
            if adapt_state:
                opt_state = state_transition(opt_state, variables)
            else:
                opt_state = optimizer.init(variables['params'])
            
        # Calculate the loss
        params, state = variables['params'], variables['state']
        variables, opt_state, loss, loc_w  = train_step(params, opt_state, collocs, bc_collocs, bc_data, glob_w, loc_w, state)

        # Append the loss
        train_losses = train_losses.at[epoch].set(loss)

    # Calculate training time
    end_time = time.time()
    elapsed = end_time-start_time
    print(f"Total Time: {elapsed} s")
    print(f"Average time per iteration: {elapsed/num_epochs:.4f} s")
    
    return model, variables, train_losses