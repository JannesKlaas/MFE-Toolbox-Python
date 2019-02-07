#import bootstrap
from .stationary_bootstrap import stationary_bootstrap
from .block_bootstrap import block_bootstrap

import numpy as np
def bsds(bench,models,B,w,test_type = 'STUDENTIZED',boot='STATIONARY'):
    """
     Calculate Whites and Hansens p-vals for out-performance using unmodified data or 
     studentized residuals, the latter often providing better power, particularly when 
     the losses functions are heteroskedastic

     USAGE:
       [C,U,L] = bsds_studentized(BENCH,MODELS,B,W,TEST_TYPE,BOOT)

     INPUTS:
       BENCH  - Losses from the benchmark model
       MODELS - Losses from each of the models used for comparison
       B      - Number of Bootstrap replications
       W      - Desired block length
       TEST_TYPE   - String, either 'STANDARD' or 'STUDENTIZED'.  'STUDENTIZED' is the default, 
       and generally leads to better power.
       BOOT   - [OPTIONAL] 'STATIONARY' or 'BLOCK'.  Stationary is used as the default.

     OUTPUTS:
       C      - Consistent P-val(Hansen)
       U      - Upper P-val(White) (Original RC P-vals)
       L      - Lower P-val(Hansen)

     COMMENTS:
       This version of the BSDS operates on quantities that should be 'bads', such as losses.  
       The null hypothesis is that the average performance of  the benchmark is as small as the
       minimum average performance across the models.  The alternative is that the minimum 
       average loss across the models is smaller than the the average performance of the 
       benchmark.

       If the quantities of interest are 'goods', such as returns, simple call 
       bsds_studentized with -1*BENCH and -1*MODELS

     EXAMPLES:
       Standard Reality Check with 1000 bootstrap replications and a window size of 12
           bench = randn(1000,1).^2;
           models = randn(1000,100).^2;
           [c,realityCheckPval] = bsds(bench, models, 1000, 12)
       Standard Reality Check with 1000 bootstrap replications, a window size of 12 and a 
       circular block bootstrap
           [c,realityCheckPval] = bsds(bench, models, 1000, 12, 'BLOCK')
       Hansen's P-values
           SPAPval = bsds(bench, models, 1000, 12)
       Both Pvals on "goods"
           bench = .01 + randn(1000,1);
           models = randn(1000,100);
           [SPAPval,realityCheckPval] = bsds(-bench, -models, 1000, 12)

     See also MCS
     Implementation by Jannes Klaas
     jannes.klaas.mfe18@said.oxford.edu
     Adapted from Kevin Sheppard
    """
    
    # Input checking
    isStudentized = (test_type == 'STUDENTIZED')
    (tb,kb)=bench.shape
    (t,k) = models.shape
    assert kb == 1, 'BENCH must be a column vector'
    assert tb > 2, 'BENCH must have at least 2 observations.'
    assert t == tb, 'BENCH and MODELS must have the same number of observations.'
    assert B >= 1, 'B must be a positive scalar integer'
    assert w >=1, 'W must be a positive scalar integer'
    boot = boot.upper()
    assert (boot == 'STATIONARY')  or (boot == 'BLOCK'), 'Only STATIONARY or BLOCK boot\
    type supported'
    
    bsdata = None
    if boot == 'BLOCK':
        bsdata,_ = block_bootstrap(np.expand_dims(np.arange(0,t),-1),B,w)
    else:
        bsdata,_ = stationary_bootstrap(np.expand_dims(np.arange(0,t),-1),B,w)
    
    # Compute loss differences
    diffs = models - np.tile(bench,(1,k))
    
    q=1/w
    i = np.arange(1,t)
    kappa = ((t-i)/t) * (1-q)**i + i/t * (1-q)**(t-i)
    
    variances = np.zeros((k,1))
    
    for i in range(0,k):
        
        workdata = diffs[:,i] - np.mean(diffs[:,i])
        variances[i] = workdata.T.dot(workdata)/t
        
        for j in range(0,t-1):
            variances[i] = variances[i] + 2*kappa[j]*workdata[0:t-j].T.dot(workdata[j:t])/t
    
    # A new used the log(log(t)) rule
    Anew = np.sqrt((variances/t)*2*np.log(np.log(t)));    
    
    # Only recenter if the average is reasonably small or the model is better
    # (in which case mean(diffs) is negative).  If it is unreasonably large set
    # the mean adjustment to 0
    ms = np.expand_dims(np.mean(diffs,axis=0),-1)
    gc = ms*(ms<Anew)
    #gc = np.mean(diffs,axis=0)*(np.mean(diffs,axis=0)<Anew)
    
    
    # The lower assumes that every loss function that is worse than BM is
    # unimportant for the asymptotic distribution, hence if its mean is
    # less than 0, g=0.  This is different from the consistent where the
    # threshold was it had to be greater than -A(i)
    gl = np.clip(np.mean(diffs,axis=0),a_min = None,a_max = 0)
    
    
    #Then the upper, which assumes all models used are reasonably close to
    # the benchmark that they could be better
    gu = np.mean(diffs,axis=0)
    
    
    # Perf will hold the boostrapped statistics for B iterations
    perfc = np.zeros((B,k))
    perfl = np.zeros((B,k))
    perfu = np.zeros((B,k))
    
    stdDev = np.sqrt(variances) if isStudentized else np.ones((1,k))
    
    for i in range(0,k):
        # the i'th column of perf holds the B bootstrapped statistics
        workdata = diffs[:,i]
        mworkdata = np.mean(workdata[bsdata])
        perfc[:,i] = (mworkdata-gc[i]).T/stdDev[i]
        perfl[:,i] = (mworkdata-gl[i]).T/stdDev[i]
        perfu[:,i] = (mworkdata-gu[i]).T/stdDev[i]
        
    # Compute the test statistic
    stat = np.min(np.mean(diffs,axis=0)/stdDev)
    
    # Compute the min in each row
    perfc = np.min(perfc,axis=1)
    perfc = np.clip(perfc,a_min = None, a_max = 0)
    
    perfl = np.min(perfl,axis=1)
    perfl = np.clip(perfl,a_min = None, a_max = 0)
    
    perfu = np.min(perfu,axis=1)
    perfu = np.clip(perfu,a_min = None, a_max = 0)
    
    c=np.mean(perfc<stat)
    l=np.mean(perfl<stat)
    u=np.mean(perfu<stat)
    return c, l, u