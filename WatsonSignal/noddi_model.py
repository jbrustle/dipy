#!//anaconda/bin/python

import numpy as np
from WatsonSHStickTortIsoV_BO import WatsonSHStickTortIsoV_BO
from dipy.core.geometry import sphere2cart

def noddi(var, param):

	grad_dirs = param['grad_dirs']
	G = param['G'] 
	delta = param['delta'] 
	smalldel = param['smalldel'] 
	roots= param['roots']
	
	x = np.zeros(7)

	x[0] = var[0]
	x[1] = param['d_par']
	x[2] = var[1] 
	x[3] = param['d_iso']
	x[4] = var[2]
	x[5] = param['b0']  

	fibredir = sphere2cart(1,var[3],var[4])
	fibredir = np.array(fibredir)
	fibredir = fibredir.T

	E = WatsonSHStickTortIsoV_BO(x, grad_dirs, G, delta, smalldel, fibredir, roots)

	"""
	x(1) is the volume fraction of the intracellular space.
    x(2) is the free diffusivity of the material inside and outside the cylinders.
    x(3) is the concentration parameter of the Watson's distribution.
    x(4) is the volume fraction of the isotropic compartment.
    x(5) is the diffusivity of the isotropic compartment.
    x(6) is the measurement at b=0.
	"""


	return E

