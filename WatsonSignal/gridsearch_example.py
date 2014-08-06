#!//anaconda/bin/python

import numpy as np

from noddi_model import noddi

param = {}

gradients = np.genfromtxt('../dipy/data/gtab_isbi2013_2shell.txt', delimiter = ',')

bvals = np.linalg.norm(gradients, axis = 1)

bvecs = gradients / bvals[:,None]
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

GT_S = noddi(var, param)


opt = {}
opt['GT_S'] = GT_S


def cost_func(var, param, opt):
	model = noddi(var, param)
	return np.linalg.norm(GT_S - model, 2)**2




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
var = [0.35, 0.5, 0.25, 0.25 * np.pi, 0.25 * np.pi]
print cost_func(var, param, opt)