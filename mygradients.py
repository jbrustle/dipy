#!//anaconda/bin/python

import numpy as np
import nibabel as nib



def gradient(bvals,bvecs):
	#number of vectors
	N=bvals.size
	#reshape given data
	arranged_bvals=np.array([bvals])
	arranged_bvals=arranged_bvals.reshape(1,N)
	arranged_bvecs=np.array([bvecs])
	arranged_bvecs=arranged_bvecs.reshape(3,N)
	#generate gradients
	gradient=arranged_bvecs*arranged_bvals
	gradient=gradient.transpose()
	return gradient
"""
test_bvecs=np.array([[1,2,3],[4,5,6],[7,8,9],[7,8,9]])
test_bvals=np.array([10,100,1000,10000])
#print test_bvecs.shape
#print test_bvals.shape
#print test_bvals*test_bvecs.T
#test_gradients=mygradient(test_bvals,test_bvecs)
print gradient(test_bvals,test_bvecs)

"""