#!//anaconda/bin/python

import numpy as np

from noddi_model import noddi
from scipy.optimize import fmin_powell
from gridsearch_example import cost_func_floor, gridsearch, example, cost_func
from dipy.sims.voxel import add_noise


ex = example()
var = ex[0]
param = ex[1] 
opt = ex[2]
x0 = (gridsearch(cost_func, param, opt))[0]
x0 = np.array(x0)
x_optim = fmin_powell(cost_func_floor, x0, args=(param,opt), xtol=0.0001, ftol=0.0001, maxiter=100, maxfun=None, full_output=0, disp=1, retall=0, callback=None, direc=None)
x_optim_cost = cost_func(x_optim, param, opt)
print x_optim 


GT_S = opt['GT_S'] 
noisy_S = add_noise(GT_S, 30, GT_S[0], noise_type='rician')
opt['GT_S'] = noisy_S 
x0 = (gridsearch(cost_func, param, opt))[0]
x0 = np.array(x0)
x_optim = fmin_powell(cost_func_floor, x0, args=(param,opt), xtol=0.0001, ftol=0.0001, maxiter=1000, maxfun=None, full_output=0, disp=1, retall=0, callback=None, direc=None)
x_optim_cost = cost_func(x_optim, param, opt)
print x_optim, "optimized variables"
print var, "original variables"
print x_optim_cost, "cost of optimized variables (with noisy Signal)"
print cost_func(var, param, opt), "cost of original variables (with noisy Signal)"



