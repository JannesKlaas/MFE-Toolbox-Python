import numpy as np
def block_bootstrap(data,B,w):
    #Implements a circular block bootstrap for bootstrapping stationary, dependent series
    #INPUTS:
    #    DATA   - T by 1 vector of data to be bootstrapped
    #    B      - Number of bootstraps
    #    W      - Block length
  
    #OUTPUTS:
    #    BSDATA  - T by B matrix of bootstrapped data
    #    INDICES - T by B matrix of locations of the original BSDATA=DATA(indexes);
  
    #COMMENTS:
    #   To generate bootstrap sequences for other uses, such as bootstrapping vector processes
    #   set DATA to (1:N)'.
    
    # Input checking
    (t,k)=data.shape
    assert k == 1, 'Data must be a column vector'
    assert t > 1, 'Need at least 2 observations'
    assert (w > 1) and (w < t), 'Block length must be positive and\
    less than number of observations'
    assert B > 1, 'Number of bootstraps must be larger than one'
    
    # Compute the number of blocks needed
    n_blocks = np.ceil(t/w).astype(int)
    
    # Generate the starting points
    block_start = np.floor(np.random.rand(n_blocks,B)*t)+1
    indices = np.zeros((n_blocks*w,B))
    index = 0
    adder = np.tile(np.arange(0,w).reshape((w,1)),(1,B))
    for i in range(0,t,w):
        indices[i:(i+w),:] = np.tile(block_start[index,:],(w,1)) + adder
        index += 1
    indices = indices[0:t,:]
    indices[indices > t-1] = indices[indices > t-1]-(t+1)
    indices = indices.astype(int)
    bsdata = data[indices]
    
    bsdata = bsdata.reshape((t,B))
    return bsdata, indices