#!//anaconda/bin/python

from __future__ import division

import numpy as np
import scipy as sc
import mpmath as mp

from mpmath import erfi

def HDiffCoeff(dPar, dPerp, kappa):
	
	dw = np.zeros((2,1))
	dParMdPerp = dPar - dPerp

	if (kappa < 1e-5):
		dParP2dPerp = dPar + 2*dPerp
		k2 = kappa*kappa
		dw[0] = dParP2dPerp/3 + 4*dParMdPerp*kappa/45 + 8*dParMdPerp*k2/945
		dw[1] = dParP2dPerp/3 - 2*dParMdPerp*kappa/45 - 4*dParMdPerp*k2/945

	else:
		sk = np.sqrt(kappa)
		dawsonf = 0.5*np.exp(-kappa)*np.sqrt(np.pi)*mp.erfi(sk)
		factor = sk/dawsonf
		dw[0] = (-dParMdPerp+2*dPerp*kappa+dParMdPerp*factor)/(2*kappa)
		dw[1] = (dParMdPerp+2*(dPar+dPerp)*kappa-dParMdPerp*factor)/(4*kappa)

	return dw

	