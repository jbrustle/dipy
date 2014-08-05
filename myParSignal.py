#!//anaconda/bin/python

from __future__ import division

import numpy as np
import scipy as sc

def ParSignal(x, G, delta, smalldel):
	
	d=x

	# Radial wavenumbers
	GAMMA = 2.675987E8
	smalldel = np.array(smalldel)
	G = np.array(G)
	modQ = GAMMA*smalldel*G
	modQ_Sq = modQ**2

	# diffusion time for PGSE, in a matrix for the computation below.
	difftime = (delta-smalldel/3)

	# Parallel component
	LE =-modQ_Sq*difftime*d

	return LE



def test():
	x=2
	G=[[1],[2],[3],[4],[5],[6]]
	delta=[[3],[4],[5],[6],[7],[8]]
	smalldel=[[5],[6],[7],[8],[9],[1]]

	print ParSignal(x, G, delta, smalldel)
	return 0