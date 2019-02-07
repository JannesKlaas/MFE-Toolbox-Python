##
## UNFINISHED DO NOT USE
##

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
    
    
    
    