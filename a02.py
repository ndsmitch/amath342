from __future__ import division

from math import sin
from methods import (
    NystromTwoStepMethod,
    AdamsBashforthThreeStepMethod,
    BDFTwoStepMethod,
    ERKThreeStageMethod
)
from numpy import array as vec

import time

def Q2():
    ''' Two step Nystrom implementation for y' + y = sin(t*t)
    '''
    f = lambda t, y: vec([- y[0] + sin(t*t)])
    NystromTwoStepMethod(
        f=f,
        y0=vec([0]),
        h=0.04,
        t0=0,
        tf=8
    ).plot_coord(
        title='2b) Two Step Nystrom Method Implementation',
        filename='a02/q2.png'
    )

def Q5():
    ''' magnitude plot of 3-step adams-bashforth and 2-step bdf implementations
    '''
    # ODE function for y'=f ; y'' + 10y' + 25y = 0
    def _f(x1, x2, x3, x4):
        return vec([
            -20*x1 + 10*x2 +  0*x3 +  0*x4,
             10*x1 - 20*x2 + 10*x3 +  0*x4,
              0*x1 + 10*x2 - 20*x3 + 10*x4,
              0*x1 +  0*x2 + 10*x3 - 20*x4,
        ])
    f = lambda t, y: _f(y[0], y[1], y[2], y[3])

    N = 550
    start = time.time()
    AdamsBashforthThreeStepMethod(
        f=f,
        y0=vec([1,1,1,1]),
        h=10/N,
        t0=0,
        tf=10
    ).plot_mag(
        title='5a) Three Step Adams-Bashforth - Magnitude',
        filename='a02/q5AB.png'
    )
    end = time.time()
    print('AB3: {}'.format(end-start))
    
    N = 50
    start = time.time()
    BDFTwoStepMethod(
        f=f,
        y0=vec([1,1,1,1]),
        h=10/N,
        t0=0,
        tf=10
    ).plot_mag(
        title='5b) Two Step BDF - Magnitude',
        filename='a02/q5BDF.png'
    )
    end = time.time()
    print('BD2: {}'.format(end-start))

def Q6():
    ''' Plot the 3-stage ERK method approximations for the Rossler attractor
    '''
    # ODE function for y'=f ; Rossler attractor
    _f = lambda x, y, z: vec([ -(y + z), x + 0.2*y, 0.2 + (x - 5.7)*z ])
    
    f = lambda t, x: _f(x[0], x[1], x[2])
    ERKThreeStageMethod(
        f=f,
        y0=vec([0,0,0]),
        h=0.02,
        t0=0,
        tf=500
    ).plot3D(
        title='6) Three Stage ERK - Rossler Attractor',
        filename='a02/q6.png'
    )

if __name__ == "__main__":
    Q2()
    Q5()
    Q6()

