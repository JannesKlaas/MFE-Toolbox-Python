#import bootstrap
from .stationary_bootstrap import stationary_bootstrap
from .block_bootstrap import block_bootstrap

import numpy as np
def mcs(losses,alpha,B,w,boot = 'STATIONARY'):
    """
     Compute the model confidence set of Hansen, Lunde and Nason
    
     USAGE:
       [INCLUDEDR,PVALSR,EXCLUDEDR,INCLUDEDSQ,PVALSSQ,EXCLUDEDSQ] = mcs(LOSSES,ALPHA,B,W,BOOT)
    
     INPUTS:
       LOSSES     - T by K matrix of losses
       ALPHA      - The final pval to use in the MCS
       B          - Number of bootstrap replications
       W          - Desired block length
       BOOT       - [OPTIONAL] 'STATIONARY' or 'BLOCK'.  Stationary will be used as default.
    
     OUTPUTS:
       INCLUDEDR  - Included models using R method
       PVALSR     - Pvals using R method
       EXCLUDEDR  - Excluded models using R method
       INCLUDEDSQ - Included models using SQ method
       PVALSSQ    - Pvals using SQ method
       EXCLUDEDSQ - Excluded models using SQ method
    
     COMMENTS:
       This version of the MCS operates on quantities that should be "bads", such as losses.  
       If the quantities of interest are "goods", such as returns, simply call MCS with 
       -1*LOSSES
    
     EXAMPLES
       MCS with 5     size, 1000 bootstrap replications and an average block length of 12
           losses = bsxfun(@plus,chi2rnd(5,[1000 10]),linspace(.1,1,10));
           [includedR, pvalsR] = mcs(losses, .05, 1000, 12)
       MCS on "goods"
           gains = bsxfun(@plus,chi2rnd(5,[1000 10]),linspace(.1,1,10));
           [includedR, pvalsR] = mcs(-gains, .05, 1000, 12)
       MCS with circular block bootstrap
           [includedR, pvalsR] = mcs(losses, .05, 1000, 12, 'BLOCK')
    
     See also BSDS
 
     Implementation by Jannes Klaas
     jannes.klaas.mfe18@said.oxford.edu
     
     Adapted from Kevin Sheppard
    """
    # get the length of the data
    t, M0 = losses.shape
    assert t >= 2, 'Losses must have at least 2 observations'
    assert (alpha > 0) and (alpha < 1), 'Alpha must be between 0 and 1'
    assert (B >= 1) and (np.floor(B) == B), 'B must be a positive integer'
    assert (w >= 1) and (np.floor(w) == w), 'w must be a positive integer'
    boot = boot.upper()
    assert (boot == 'STATIONARY') or (boot == 'BLOCK'), 'Boot must be "STATIONARY" \
    or "BLOCK"'
    
    
    bsdata = None
    if boot == 'BLOCK':
        bsdata,_ = bootstrap.block_bootstrap(np.expand_dims(np.arange(0,t),-1),B,w)
    else:
        bsdata,_ = bootstrap.stationary_bootstrap(np.expand_dims(np.arange(0,t),-1),B,w)
       
    # The i,j element contains the l(i,t)-l(j,t)
    dijbar=np.zeros((M0,M0))
    for j in range(0,M0):
        dijbar[j,:] = np.mean(losses - np.tile(np.expand_dims(losses[:,j],-1),(1,M0)),axis=0)
    
    
    #for each j, compute dij*-bar using the BSdata, than the compute
    #var(dijbar)
    dijbarstar = np.zeros((M0,M0,B))
    
    for b in range(0,B):
        meanworkdata = np.mean(losses[bsdata[:,b],:],axis=0)
        for j in range(0,M0):
            #The i,j element contains the l(b,i,t)-l(b,j,t)
            dijbarstar[j,:,b] = meanworkdata - meanworkdata[j]
    
    import ipdb; ipdb.set_trace()
    vardijbar = np.mean((dijbarstar-np.tile(np.expand_dims(dijbar,-1),(1,1,B)))**2,axis=2)
    
    # Identity does the same as diag(ones(M0,1)) in matlab
    vardijbar = vardijbar + np.identity(M0)
    
    z0 = (dijbarstar - np.tile(np.expand_dims(dijbar,-1),(1,1,B)))
    z0 = z0 / np.tile(np.sqrt(np.expand_dims(vardijbar,-1)),(1,1,B))
    
    zdata0=dijbar / np.sqrt(vardijbar)
    
    excludedR = np.zeros((M0,1)) -1
    pvalsR = np.ones((M0,1))
    
    
    for i in range(0,M0-1):
        included = np.setdiff1d(np.arange(0,M0),excludedR)
        m = len(included)
        z=z0[included,included,:]
        empdistTR = np.max(np.abs(z))
        zdata = zdata0[included,included]
        TR = np.max(zdata)
        pvalsR[i] = np.mean(empdistTR>TR)
        # Finally compute the model to remove, which depends on the maximum
        # standardized average (among the remaining models)
        # 1. compute dibar
        dibar =  np.mean(dijbar[included,included],axis=0) * (m/(m-1))
        # 2. compute var(dibar)
        dibstar = np.mean(dijbarstar[included,included,:],axis=0) * (m/(m-1))
        vardi = np.mean((dibstar.T - np.tile(dibar,(B,1)))**2 )
        t = dibar / np.sqrt(vardi)
        modeltoremove = np.argmax(t)
        excludedR[i] = included[modeltoremove]
    # The MCS pval is the max up to that point
    maxpval=pvalsR[0]
    for i in range(1,M0):
        if pvalsR[i] < maxpval:
            pvalsR[i] = maxpval  
        else: 
            maxpval=pvalsR[i]
    
    # Add the final remaining model to excluded
    excludedR[-1] = np.setdiff1d(np.arange(0,M0),excludedR)
    # The included models are all of these where the first pval is > alpha
    pl = np.where(pvalsR>=alpha)[0][0]
    includedR = excludedR[pl:M0]
    excludedR = excludedR[0:pl-1]
    
    excludedSQ = np.zeros((M0,1))
    pvalsSQ = np.ones((M0,1))
    
    for i in range(1,M0-1):
        included = np.setdiff1d(np.arange(0,M0),excludedSQ)
        m = len(included)
        z = z0[included,included,:]
        empdistTSQ = np.sum(z**2)/2
        zdata = zdata0[included,included]
        TSQ = np.sum(zdata**2)/2
        pvalsSQ[i] = np.mean(empdistTSQ>TSQ)
        
        # Finally compute the model to remove, which depends on the maximum
        # standardized average (among the remaining models)
        # 1. compute dibar
        dibar =  np.mean(dijbar[included,included],axis=0) * (m/(m-1))
        # 2. compute var(dibar)
        dibstar = np.mean(dijbarstar[included,included,:],axis=0) * (m/(m-1))
        vardi = np.mean((dibstar.T - np.tile(dibar,(B,1)))**2 )
        t = dibar / np.sqrt(vardi)
        modeltoremove = np.argmax(t)
        excludedSQ[i] = included[modeltoremove]
    
    maxpval=pvalsSQ[0]
    for i in range(1,M0):
        if pvalsSQ[i] < maxpval:
            pvalsSQ[i] = maxpval  
        else: 
            maxpval=pvalsSQ[i]
    
    # Add the final remaining model to excluded
    excludedSQ[-1] = np.setdiff1d(np.arange(0,M0),excludedSQ)
    
    # The included models are all of these where the first pval is > alpha
    pl = np.where(pvalsSQ>=alpha)[0][0]
    includedSQ = excludedSQ[pl:M0]
    excludedSQ = excludedSQ[0:pl-1]
    
    return includedR,pvalsR,excludedR,includedSQ,pvalsSQ,excludedSQ
        