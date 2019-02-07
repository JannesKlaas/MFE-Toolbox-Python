import numpy as np
def stationary_bootstrap(data,B,w):
    """
    Implements the stationary bootstrap for bootstrapping stationary, dependent series

    USAGE:
       [BSDATA, INDICES] = stationary_bootstrap(DATA,B,W)

    INPUTS:
       DATA   - T by 1 vector of data to be bootstrapped
       B      - Number of bootstraps
       W      - Average block length. P, the probability of starting a new block is 
       defined P=1/W

    OUTPUTS:
       BSDATA  - T by B matrix of bootstrapped data
       INDICES - T by B matrix of locations of the original BSDATA=DATA(indexes);

    COMMENTS:
       To generate bootstrap sequences for other uses, such as bootstrapping vector processes
       set DATA to (1:N)'.  

    See also block_bootstrap

    
    Implementation by Jannes Klaas
    jannes.klaas.mfe18@said.oxford.edu
    
    Adapted from Kevin Sheppard
    """
    (t,k) = data.shape
    assert k == 1, 'DATA must be a column vector'
    assert t>=2, 'DATA must have at least 2 observations.'
    assert w >= 0, 'Average block length W must be a positive scalar.'
    assert (isinstance(B,int)) and(B >= 0), 'Number of bootstrap samples B must be a \
    positive scalar integer'
    
    # Probability of new block
    p=1/w
    
    #Set up the bsdata and indices
    indices = np.zeros((t,B))
    indices[0,:] = np.ceil(t*np.random.rand(1,B))
    
    # Set up random numbers
    select=np.random.rand(t,B) < p
    indices[select] = np.ceil(np.random.rand(1,np.sum(select))*t).flatten()
    
    
    for i in range(1,t):
        # Determine whether we stay (rand>p) or move to a new starting value
        indices[i,~select[i,:]] = indices[i-1,~select[i,:]]
    # Make sure indices don't go out of bound
    indices[indices>t-1] = indices[indices>t-1]-t-1
    # Indices need to be integers
    indices = indices.astype(int)
    # Sample data
    bsdata=data[indices]
    # Get rid of extra dimension that comes from sampling
    bsdata = bsdata.reshape((t,B))
    
    return bsdata, indices