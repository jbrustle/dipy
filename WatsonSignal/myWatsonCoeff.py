#!//anaconda/bin/python

from __future__ import division

import numpy as np
import mpmath as mp

from mpmath import erfi


def WatsonSHCoefficients(k):


    # Maximum order of SH coefficients
    n=6

    # Computing the SH coefficients
    C = np.zeros(n+1);

    # 0th order is a constant
    C[0] = 2*np.sqrt(np.pi);

    # Precompute some values
    sk = np.sqrt(k)
    sk2 = sk*k
    sk3 = sk2*k
    sk4 = sk3*k
    sk5 = sk4*k
    sk6 = sk5*k
    sk7 = sk6*k
    
    k2 = k*k
    k3 = k2*k
    k4 = k3*k
    k5 = k4*k
    k6 = k5*k
    k7 = k6*k

    
    erfik = mp.erfi(sk)
    ierfik = 1./erfik
    ek=np.exp(k)

    # dawson integral 
    dawsonk=0.5*np.sqrt(np.pi)*erfik/ek

    #Compute coefficients by hand
    C[1] = 3*sk - (3 + 2*k)*dawsonk
    C[1] = np.sqrt(5)*C[1]*ek
    C[1] = C[1]*ierfik/k

    C[2] = (105 + 60*k + 12*k2)*dawsonk
    C[2] = C[2] -105*sk + 10*sk2
    C[2] = .375*C[2]*ek/k2
    C[2] = C[2]*ierfik

    C[3] = -3465 - 1890*k - 420*k2 - 40*k3
    C[3] = C[3]*dawsonk
    C[3] = C[3] + 3465*sk - 420*sk2 + 84*sk3
    C[3] = C[3]*np.sqrt(13*np.pi)/64./k3
    C[3] = C[3]/dawsonk

    C[4] = 675675 + 360360*k + 83160*k2 + 10080*k3 + 560*k4
    C[4] = C[4]*dawsonk
    C[4] = C[4] - 675675*sk + 90090*sk2 - 23100*sk3 + 744*sk4
    C[4] = np.sqrt(17)*C[4]*ek
    C[4] = C[4]/512./k4
    C[4] = C[4]*ierfik

    C[5] = -43648605 - 22972950*k - 5405400*k2 - 720720*k3 - 55440*k4 - 2016*k5
    C[5] = C[5]*dawsonk
    C[5] = C[5] + 43648605*sk - 6126120*sk2 + 1729728*sk3 - 82368*sk4 + 5104*sk5
    C[5] = np.sqrt(21*np.pi)*C[5]/4096./k5
    C[5] = C[5]/dawsonk

    C[6] = 7027425405 + 3666482820*k + 872972100*k2 + 122522400*k3  + 10810800*k4 + 576576*k5 + 14784*k6
    C[6] = C[6]*dawsonk
    C[6] = C[6] - 7027425405*sk + 1018467450*sk2 - 302630328*sk3 + 17153136*sk4 - 1553552*sk5 + 25376*sk6
    C[6] = 5*C[6]*ek
    C[6] = C[6]/16384./k6
    C[6] = C[6]*ierfik

    return C

def test_WatsonSHCoefficients():    
    sample =  WatsonSHCoefficients(4)
    error = abs(abs(sample[0])-3.5449) + abs(abs(sample[1])-4.4147) + abs(abs(sample[2]-2.1933)) + abs(abs(sample[3])-0.7212) + abs(abs(sample[4])-0.1782) + abs(abs(sample[5])-0.0353) + abs(abs(sample[6])-0.0058)
    
    test=True
    if (error>1.E-03):
        test=False

    return test


    
