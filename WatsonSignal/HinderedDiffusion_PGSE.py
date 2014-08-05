#!//anaconda/bin/python

from __future__ import division
import numpy as np

def SynthMeasHinderedDiffusion_PGSE(x, grad_dirs, G, delta, smalldel, fibredir):
 	"""
 	Substrate: Anisotropic hindered diffusion compartment
 	Pulse sequence: Pulsed gradient spin echo
 	Signal approximation: N/A

 	[E,J]=SynthMeasHinderedDiffusion_PGSE(x, grad_dirs, G, delta, smalldel, fibredir)
 	returns the measurements E according to the model and the Jacobian J of the
 	measurements with respect to the parameters.  The Jacobian does not
 	include derivates with respect to the fibre direction.

 	x is the list of model parameters in SI units:
 	x(1) is the free diffusivity of the material inside and outside the cylinders.
 	x(2) is the hindered diffusivity outside the cylinders in perpendicular directions.

 	grad_dirs is the gradient direction for each measurement.  It has size [N
 	3] where N is the number of measurements.

 	G, delta and smalldel are the gradient strength, pulse separation and
 	pulse length of each measurement in the protocol.  Each has
 	size [N 1].

 	fibredir is a unit vector along the cylinder axis.  It must be in
 	Cartesian coordinates [x y z]' with size [3 1].
 	"""


	dPar=x[0]
	dPerp=x[1]

	# Radial wavenumbers
	GAMMA = 2.675987E8
	modQ = GAMMA*smalldel*G
	modQ_Sq = modQ**2

	#Angles between gradient directions and fibre direction.
	cosTheta = np.dot(grad_dirs, fibredir)
	cosThetaSq = cosTheta**2
	sinThetaSq = 1-cosThetaSq

	# b-value
	bval = (delta-smalldel/3)*modQ_Sq
	bval = np.reshape(bval,(bval.size,1))

	# Find hindered signals
	E=np.exp(np.multiply(-bval,(((dPar - dPerp)*cosThetaSq) + dPerp)))

	return E

def test_SynthMeasHinderedDiffusion_PGSE():


    grad_dirs = [1.0000,         0 ,        0,
    0.1327,   -0.7399,    0.6595,
    -0.9183,    0.3799,   -0.1114,
    -0.9654,   -0.1533,   -0.2108,
    0.6086,    0.7845,   -0.1192,
    0.6392,   -0.4376,   -0.6324,
    0.7895,   -0.2093,    0.5769,
    0.1350,    0.3625,   -0.9221,
    0.3443,   -0.0935,   -0.9342,
    1.0000,         0,         0,
    0.6823,   -0.7065,   -0.1879,
    0.1417,   -0.5505,   -0.8227,
    0.6996,    0.1677,   -0.6946,
    -0.5914,    0.6905,   -0.4166,
    0.9352,   -0.1362,   -0.3270,
    0.6448,    0.2485,    0.7228,
    -0.7270,   -0.6007,   -0.3326,
    -0.1562,   -0.9406,    0.3015,
    1.0000,         0,         0,
    -0.1962,   -0.1494,   -0.9691,
    0.2934,   -0.9528,    0.0787,
    0.4674,    0.6288,   -0.6214,
    0.2426,   -0.8630,   -0.4432,
    -0.3219,   -0.6682,   -0.6708,
    0.2297,    0.9437,    0.2379,
    -0.3920,    0.3217,   -0.8619,
    0.8877,    0.3878,   -0.2482,
    1.0000,         0,         0,
    -0.9003,   -0.0791,   -0.4280,
    0.7262,   -0.5380,   -0.4280,
    0.9134,   -0.3765,   -0.1549,
    0.1883,   -0.5651,   -0.8032,
    -0.1034,   -0.3541,   -0.9295,
    0.3334,    0.9428,   -0.0083,
    0.6925,   -0.7125,   -0.1127,
    -0.6849,   -0.0588,   -0.7262,
    1.0000,         0,         0,
    0.6120,   -0.0294,   -0.7903,
    -0.3721,   -0.0463,   -0.9270,
    0.2860,   -0.8403,    0.4605,
    -0.5451,    0.2954,   -0.7846,
    0.9744,    0.2037,    0.0954,
    0.6816,    0.5668,   -0.4627,
    0.8335,   -0.1898,   -0.5190,
    -0.1703,   -0.6685,   -0.7239,
    1.0000,         0,         0,
    -0.2515,   -0.8866,   -0.3881,
    0.0963,   -0.8518,   -0.5149,
    -0.2692,    0.6059,   -0.7486,
    -0.4646,   -0.3828,   -0.7985,
    0.9730,   -0.1742,    0.1514,
    0.9757,   -0.0017,   -0.2190,
    0.4331,   -0.7085,   -0.5571,
    0.2902,   -0.2167,   -0.9321,
    1.0000,         0,         0,
    -0.2878,    0.9529,   -0.0954,
    0.6711,    0.7301,   -0.1288,
    0.3743,    0.6295,   -0.6809,
    0.7562,    0.3986,    0.5189,
    -0.8321,    0.5133,   -0.2103,
    -0.5807,   -0.7825,   -0.2245,
    0.3946,   -0.8859,   -0.2439,
    0.5750,    0.3345,   -0.7467,
    1.0000,         0,         0,
    0.0526,    0.7931,   -0.6069,
    0.0488,    0.9639,   -0.2616,
    0.8348,    0.1918,   -0.5161,
    0.5524,   -0.3924 ,  -0.7355,
    0.3931,    0.8412 ,  -0.3712,
    0.3026,    0.1875,   -0.9345,
    -0.5809,    0.7824,   -0.2245,
    0.8316,    0.5252,    0.1807,
    1.0000,         0,         0,
    -0.8127,    0.2760,   -0.5132,
    0.0987,    0.4874,   -0.8676,
    -0.5963,    0.5886,   -0.5458,
    0.0150,   -0.9886,   -0.1497,
    -0.5067,   -0.6580,   -0.5571,
    0.8914,    0.4062,   -0.2010,
    -0.2006,    0.2950,   -0.9342,
         0,         0,   -1.0000]

    grad_dirs = np.array(grad_dirs)
    grad_dirs=grad_dirs.reshape(81,3) 

    G=[         0,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
         0,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
         0,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
    0.0237,
         0,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
         0,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
         0,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
         0,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
         0,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
         0,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400]
    G=np.array(G)

    smalldel=[ 0,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
         0,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
         0,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
         0,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
         0,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
         0,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
         0,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
         0,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
         0,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297,
    0.0297]
    smalldel=np.array(smalldel)
    delta=smalldel
    fibredir =  [[-0.7419], [0.1710], [0.6484]]
    fibredir=np.array(fibredir)
    roots = 0
    x = np.array([0.000000000017,0,1.2767])
    xh = np.array([  7.77607182e-12, 4.61196409e-12])


    sample = SynthMeasHinderedDiffusion_PGSE(xh, grad_dirs, G, delta, smalldel, fibredir)

    error = abs(abs(sample[1])- 0.999509696963447) + abs(abs(sample[2])-0.994589255971989) + abs(abs(sample[3])-0.996350179848374) 
    error = error + abs(abs(sample[4])-0.998141620811197) + abs(abs(sample[5])-0.989074623461130) + abs(abs(sample[6])-0.999268966504316) 


    test=True
    if (error > 1.E-12):
        test=False
    return test




