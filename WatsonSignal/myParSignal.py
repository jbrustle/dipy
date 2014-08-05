#!//anaconda/bin/python

from __future__ import division

import numpy as np


def ParSignal(x, G, delta, smalldel):
    
    x=np.array(x)
    if(x.size>1):
        d=x[0]
    else:   
        d=x    
        
    # Radial wavenumbers
    GAMMA = 2.675987E8
    smalldel = np.array(smalldel)
    G = np.array(G)
    modQ = GAMMA*smalldel*G
    modQ_Sq = np.square(modQ)

    # diffusion time for PGSE, in a matrix for the computation below.
    difftime = (delta-(smalldel/3.))

    # Parallel component
    LE = np.multiply(-modQ_Sq, difftime)*d

    return LE



def test_ParSignal():
    x=np.array([2,3,4,5,6,7])
    G=[[1],[2],[3],[4],[5],[6]]
    delta=[[3],[4],[5],[6],[7],[8]]
    smalldel=[[5],[6],[7],[8],[9],[1]]

    sample = ParSignal(x, G, delta, smalldel)
    
    error = abs(abs(sample[0])/1.E18-4.7739) + abs(abs(sample[1])/1.E19-4.1247) + abs(abs(sample[2])/1.E20-1.6842) 
    error = error + abs(abs(sample[3])/1.E20-4.8885) + abs(abs(sample[4])/1.E21-1.1601) + abs(abs(sample[5])/1.E19-3.9528) 

    test=True
    if (error>1.E-03):
        test=False

    return test








