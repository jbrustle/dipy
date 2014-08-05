#!//anaconda/bin/python

from __future__ import division

import numpy as np
import scipy as sc
import mpmath as mp

from scipy.special import erf

def LegendreGaussianIntegral(x,n):
    x = np.array(x)
    exact = np.where(x>0.05)
    exact = np.vstack(exact)
    approx = np.where(x<=0.05)
    approx = np.vstack(approx)  

    mn=n+1

    I = np.zeros((np.size(x),mn))
    sqrtx = np.sqrt(x[exact])

    tmp = np.zeros(((np.size(approx))/2))
    tmp2 = np.append(tmp,np.sqrt(np.pi)*sc.special.erf(sqrtx)/sqrtx)
    I[:,0] = tmp2
    dx = 1/x[exact]
    emx = -np.exp(-x[exact])


    for i in range(1,mn,1):
        emx=emx.reshape(I[exact,0].shape)
        dx=dx.reshape(I[exact,0].shape)
        I[exact,i] = emx + (i-0.5)*I[exact,i-1]
        I[exact,i] = I[exact,i]*dx

    # Computing the legendre gaussian integrals for large enough x
    L = np.zeros((np.size(x),n+1))
    for i in range(0,n+1,1):
        if i == 0:
            L[exact,0] = I[exact,0]
        elif i == 1:
            L[exact,1] = -0.5*I[exact,0] + 1.5*I[exact,1]
        elif i == 2:
            L[exact,2] = 0.375*I[exact,0] - 3.75*I[exact,1] + 4.375*I[exact,2]
        elif i == 3:
            L[exact,3] = -0.3125*I[exact,0] + 6.5625*I[exact,1] - 19.6875*I[exact,2] + 14.4375*I[exact,3]
        elif i == 4:
            L[exact,4] = 0.2734375*I[exact,0] - 9.84375*I[exact,1] + 54.140625*I[exact,2] - 93.84375*I[exact,3] + 50.2734375*I[exact,4]
        elif i == 5:
            L[exact,5] = -(63/256)*I[exact,0] + (3465/256)*I[exact,1] - (30030/256)*I[exact,2] + (90090/256)*I[exact,3] - (109395/256)*I[exact,4] + (46189/256)*I[exact,5]
        elif i == 6:
            L[exact,6] = (231/1024)*I[exact,0] - (18018/1024)*I[exact,1] + (225225/1024)*I[exact,2] - (1021020/1024)*I[exact,3] + (2078505/1024)*I[exact,4] - (1939938/1024)*I[exact,5] + (676039/1024)*I[exact,6]


    #Computing the legendre gaussian integrals for small x

    x=np.squeeze(x)
    x2=np.square(x[approx])
    x3=x2*x[approx]
    x4=x3*x[approx]
    x5=x4*x[approx]
    x6=x5*x[approx]
    for i in range(0,n+1,1):
        if i == 0:
            L[approx,0] = 2 - 2*x[approx]/3 + x2/5 - x3/21 + x4/108
        elif i == 1:
          L[approx,1] = -4*x[approx]/15 + 4*x2/35 - 2*x3/63 + 2*x4/297
        elif i == 2:
            L[approx,2] = 8*x2/315 - 8*x3/693 + 4*x4/1287
        elif i == 3:
            L[approx,3] = -16*x3/9009 + 16*x4/19305
        elif i == 4:
            L[approx,4] = 32*x4/328185
        elif i == 5:
            L[approx,5] = -64*x5/14549535
        elif i == 6:
            L[approx,6] = 128*x6/760543875

    return L

def test():
    LegendreGaussianIntegral([0.5,0.78,0.36,0.48,0.74,0.95],6)
    return 0

