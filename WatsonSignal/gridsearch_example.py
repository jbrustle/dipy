#!//anaconda/bin/python

import numpy as np

from noddi_model import noddi

from scipy.optimize import fmin_powell

def example():
	param = {}

	gradients = np.genfromtxt('../dipy/data/gtab_isbi2013_2shell.txt', delimiter = ',')

	bvals = np.linalg.norm(gradients, axis = 1)
	bvals[0] = 1
	bvecs = gradients / bvals[:,None]
	bvals[0] = 0
	bvecs[0] = [0,0,0]
	param['grad_dirs'] = bvecs


	smalldel = 0.03199439 * np.ones_like(bvals)
	smalldel[np.where(bvals<1)] =  0.0
	delta = smalldel
	param['smalldel'] = smalldel
	param['delta'] = delta

	G = np.ones_like(bvals)
	G[np.where(bvals<1)] =  0.0
	G[np.where((bvals>1400)&(bvals<1600))] =  0.03098386671263059
	G[np.where((bvals>2400)&(bvals<2600))] =  0.04
	param['G'] = G

	roots = 0
	param['roots'] = roots

	param['d_iso'] = 0.000000003000
	param['d_par'] = 0.000000001700

	param['b0'] = 1


	var = [0.35, 0.5, 0.25, 0.25 * np.pi, 0.25 * np.pi]
	var = [0.25, 0.35, 0.75, 0.27 * np.pi, 0.23 * np.pi]
	GT_S = noddi(var, **param)


	opt = {}
	opt['GT_S'] = GT_S

	return (var, param, opt)


def cost_func(var, param, opt):
	model = noddi(var, **param)
	return np.linalg.norm(opt['GT_S'] - model, 2)**2

def cost_func_floor(var, param, opt):
	if var[1] < 0:
		var[1] = 0.01
	return cost_func(var, param, opt)




def gridsearch(func, param, opt):
	# searches over 2**linspace(0,7,8) for kappa (var[1])
	var_bounds = np.array([[0, 1], 
						   [0, 7],
						   [0, 1], 
						   [0, 0.5*np.pi],
						   [0, 2*np.pi]])

	var_step = np.array([5,8,5,10,20])

	grid = []
	for i in range(5):
		grid.append(np.linspace(var_bounds[i,0], var_bounds[i,1], var_step[i]))

	# nd_grid = np.meshgrid(grid[0],grid[1],2**grid[2],grid[3],grid[4])

	cost_min = np.inf
	var_min = [0,0,0,0,0]
	for i0 in range(var_step[0]):
		for i1 in range(var_step[1]):
			print i1
			for i2 in range(var_step[2]):
				for i3 in range(var_step[3]):
					for i4 in range(var_step[4]):
						# var = [nd_grid[0][i0,i1,i2,i3,i4],
						# 	   nd_grid[1][i0,i1,i2,i3,i4],
						# 	   nd_grid[2][i0,i1,i2,i3,i4],
						# 	   nd_grid[3][i0,i1,i2,i3,i4],
						# 	   nd_grid[4][i0,i1,i2,i3,i4]]
						var = []
						var.append(grid[0][i0])
						var.append(2**grid[1][i1])
						var.append(grid[2][i2])
						var.append(grid[3][i3])
						var.append(grid[4][i4])

						# print(var)
						cost = func(var, param, opt)
						if cost < cost_min:
							var_min = var
							cost_min = cost
	return (var_min, cost_min)



"""
# def G_from_q(q, smalldel):
# 	gyro = 267.513
# 	return (2*np.pi*q)/(smalldel*gyro)

# def q_from_b(b, smalldel, bigdel):
# 	return  (b/(bigdel - (1/3.)*smalldel))**(0.5) / (2*np.pi)

# def G_from_b(b, smalldel, bigdel):
# 	return G_from_q(q_from_b(b, smalldel, bigdel), smalldel)





# def q_from_G(G, smalldel):
# 	gyro = 267.513
# 	return gyro*G*smalldel/(2*np.pi)

# def b_from_q(q, smalldel, bigdel):
# 	return (2*np.pi*q)**2 * (bigdel - (1/3.)*smalldel)


# def b_from_G(G, smalldel, bigdel):
# 	return b_from_q(q_from_G(G, smalldel), smalldel, bigdel)
"""

"""
ex = example()
var = ex[0]
param = ex[1] 
opt = ex[2]
x0 = (gridsearch(cost_func, param, opt))[0]
x0 = np.array(x0)
x_optim = fmin_powell(cost_func_floor, x0, args=(param,opt), xtol=0.0001, ftol=0.0001, maxiter=100, maxfun=None, full_output=0, disp=1, retall=0, callback=None, direc=None)
x_optim_cost = cost_func(x_optim, param, opt)
print "var: ", var
print "x0: ", x0 
print "x_optim: ", x_optim
print "x_optim_cost: ", x_optim_cost

"""

