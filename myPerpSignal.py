#!//anaconda/bin/python

from __future__ import division

import numpy as np
import scipy as sc
import mpmath as mp

from mpmath import erfi

def PerpSignal(d, R, G, delta, smalldel, roots):
	
	# Radial wavenumbers
	GAMMA = 2.675987E8;

	# number of gradient directions, i.e. number of measurements
	l_q=np.size(G)
	l_a=np.size(R)
	k_max=np.size(roots)

	R_mat=np.tile(R,(l_q,1))
	R_mat=R_mat.T.ravel()
	R_mat=np.tile(R_mat,(k_max,1,1))
	R_mat=np.transpose(R_mat)
	R_matSq=np.square(R_mat)

	#	if not((isinstance( roots, ( int, long ) )) or (isinstance( roots, ( float, long ) ))):
	#	roots=np.array(roots)

	roots=np.squeeze(np.array([roots]))
	root_m=np.reshape(roots,(1,1,k_max))
	alpha_mat=np.tile(root_m,(l_q*l_a, 1, 1))/R_mat
	amSq=np.power(alpha_mat,2)
	amP6=np.power(amSq,3)

	deltamx=np.tile(delta,(1,l_a))
	deltamx_rep=deltamx.T.ravel()
	deltamx_rep=np.vstack(deltamx_rep)
	deltamx_rep = np.tile(deltamx_rep,(1, 1, k_max))
	deltamx_rep=np.reshape(deltamx_rep,((l_a)**2,1,l_a))

	smalldelmx=np.tile(smalldel,(1,l_a))
	smalldelmx_rep=smalldelmx.T.ravel()
	smalldelmx_rep=np.vstack(smalldelmx_rep)
	smalldelmx_rep =np.tile(smalldelmx_rep,(1, 1, k_max))
	smalldelmx_rep=np.reshape(smalldelmx_rep,((l_a)**2,1,l_a))

	Gmx=np.tile(G,(1,l_a))
	GmxSq = np.square(Gmx)

	#Perpendicular component

	sda2 = np.multiply(smalldelmx_rep,amSq)
	bda2 = np.multiply(deltamx_rep,amSq)
	emdsda2 = np.exp(-d*sda2)
	emdbda2 = np.exp(-d*bda2)
	emdbdmsda2 = np.exp(-d*(bda2 - sda2))
	emdbdpsda2 = np.exp(-d*(bda2 + sda2))

	sumnum1 = 2*d*sda2
	# the rest can be reused in dE/dR
	sumnum2 = - 2 + 2*emdsda2 + 2*emdbda2
	sumnum2 = sumnum2 - emdbdmsda2 - emdbdpsda2
	sumnum = sumnum1 + sumnum2

	sumdenom = d**2 * np.multiply(amP6,np.multiply(R_matSq,amSq) - 1)


	sumterms = np.divide(sumnum,sumdenom)
	s = sumterms.sum(-1).sum(-1)
	s = np.reshape(s,(l_q,l_a))

	LE = -2*(GAMMA**2)*GmxSq*s.T
	
	print LE
	return LE

def test():	
	R=[[1,2,3,4,5,6]]
	G=[[1],[2],[3],[4],[5],[6]]
	roots=[[1,2,3,4,5,6]]
	delta=[[3],[4],[5],[6],[7],[8]]
	d=1.2
	smalldel=[[5],[6],[7],[8],[9],[1]]
	A = PerpSignal(d,R,G,delta,smalldel,roots)
	B = A[:,4]
	C = B/(1.E36) 
	error = np.zeros(np.shape(C))
	error[0] = (C[0]+0.0220)
	error[1] = (C[1]+0.1846)
	error[2] = (C[2]+0.7307)
	error[3] = (C[3]+2.0538) 
	error[4] = (C[4]+4.7181) 
	error[5] = (C[5]+0.1758) 
	error= np.absolute(error)

	print error.sum()

	# check if function works correctly
	if (error.sum() > 0.00015):
		print "Example failed"
	return 0

