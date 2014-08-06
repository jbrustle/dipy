#!//anaconda/bin/python

from  dipy.core.geometry import sphere2cart as sph2cart

from WatsonSHStickTortIsoV_BO import WatsonSHStickTortIsoV_BO

import numpy as np

def noddi(var, grad_dirs, G, delta, smalldel, roots, b0, d_par, d_iso):
    """
    var[0] is the volume fraction of the intracellular space.
    var[1] is the concentration parameter of the Watson's distribution.
    var[2] is the volume fraction of the isotropic compartment.
    var[3] theta (angle) of direction vector along the symmetry axis of the Watson's ([0, pi])
    var[4] phi (angle) of direction vector along the symmetry axis of the Watson's ([0, 2pi])
    """
    fibredir = np.array(sph2cart(1, var[3], var[4]))[:,None]

    x = np.zeros(6)
    x[0] = var[0]
    x[1] = d_par
    x[2] = var[1]
    x[3] = var[2]
    x[4] = d_iso
    x[5] = b0

    return WatsonSHStickTortIsoV_BO(x, grad_dirs, G, delta, smalldel, fibredir, roots)






"""
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

    return E

"""