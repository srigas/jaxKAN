import jax
import jax.numpy as jnp
import optax

def _grad(f, idx, order=1):
    """
    Compute the higher order gradient of function f with respect to the idx-th column of the input.
    """
    def grad_fn(g, idx):
        return lambda tx: jax.grad(lambda tx: jnp.sum(g(tx)))(tx)[..., idx].reshape(-1,1)

    g = lambda tx: f(tx)
    for _ in range(order):
        g = grad_fn(g, idx)
    return g


def create_los_mean_KAN(losses, datas, sigmas):
    """
    Creates three loss functions given a list of losses [functions], datas [arrays], sigmas [arrays]
    Inputs:
    - losses: function with this signature, loss(params,colloc,data,sigma) -> array
    For example a data loss returns the los value of every datapoint or a physics loss which return the los of every collocation point.
    - datas: the corresponding data (i.e the obvservables for the data loss). For the physics loss this can be None
    - sigmas: the error of the observations if our los functions needs them
    Returns:
    - L: trainable los function. It has a signature (params,X,W,state) -> mean(losses)
    X are the inputs (collocation points for physics, x for data)
    W are the Weights for the different loss functions
    state is the KAN grid state
    - L_no_W: The same with L but without Weights and without aggregations
    - L_W: The same with L without aggregations
    
    """
    @jax.jit
    def L(params, collocs, W,state):
        variables = {'params' : params, 'state' : state}
        los=0.0
        for loss, colloc, data, sigma, w in zip(losses, collocs, datas, sigmas, W):
            los+=w*jnp.mean(loss(variables, colloc, data, sigma))
        return los

    @jax.jit
    def L_no_W(params, collocs,state):
        variables = {'params' : params, 'state' : state}
        los=[]
        for loss, colloc, data, sigma in zip(losses, collocs, datas, sigmas):
            los.append(jnp.mean(loss(variables, colloc, data, sigma)))
        return jnp.array(los)
    @jax.jit
    def L_W(params, collocs,W,state):
        variables = {'params' : params, 'state' : state}
        los=[]
        for loss, colloc, data, sigma, w in zip(losses, collocs, datas, sigmas, W):
            los.append(w*jnp.mean(loss(variables, colloc, data, sigma)))
        return jnp.array(los)
    return L,L_no_W,L_W

@jax.jit
def smooth_state_transition(old_state, variables):

    # Copy old state
    adam_count, adam_mu, adam_nu = old_state[0].count, old_state[0].mu, old_state[0].nu

    # Try a zero count
    adam_count = jnp.array(0, dtype=jnp.int32)

    # Get all layer-related keys, so that we do not touch the other parameters
    layer_keys = {k for k in adam_mu.keys() if k.startswith('layers_')}
    
    for key in layer_keys:
        # Find the c_basis shape for this layer
        c_shape = variables['params'][key]['c_basis'].shape
        # Find the c_basis for this layer and initialize its mu to zero
        adam_mu[key]['c_basis'] = jnp.zeros(c_shape, dtype=jnp.float32)
        # Find the c_basis for this layer and initialize its nu to zero
        adam_nu[key]['c_basis'] = jnp.zeros(c_shape, dtype=jnp.float32)

    # Make new adam state
    adam_state = optax.ScaleByAdamState(adam_count, adam_mu, adam_nu)
    # Make new empty state
    empty_state = optax.EmptyState()
    # Make new total state
    new_state = (adam_state, empty_state)

    return new_state