#!//anaconda/bin/python

from __future__ import division

import numpy as np

def SynthMeasIsoGPD(d, G, delta, smalldel):

    GAMMA = 2.675987E8
    modQ = GAMMA*smalldel*G
    modQ_Sq = modQ**2
    difftime = delta-smalldel/3

    E = np.exp(-difftime*modQ_Sq*d)

    return E

def test_SynthMeasIsoGPD():

    G=[     0,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
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
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192]
    smalldel=np.array(smalldel)
    delta=smalldel
    d = 3.E-09
    sample = SynthMeasIsoGPD(d, G, delta, smalldel)


    error = abs(abs(sample[2])-  0.122456428252982) + abs(abs(sample[3])- 0.122456428252982) 
    error = error + abs(abs(sample[31])- 0.002478752176666) + abs(abs(sample[32])- 0.002478752176666)

    test=True
    if (error > 1.E-12):
        test=False
    return test


