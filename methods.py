from collections import defaultdict
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from sympy import linsolve, symbols


class NumericalMethod(object):
    ''' Generic EXPLICIT numerical method.
    Must be subclassed to implement a self.step method.
    Steps through numerical method in initializer.
    Methods to calculate error or generate plots.

    y: [np.array] ; numerical method solutions entries
    f: [np.array] ; numerical method derivative entries
    t: [num] ; numerical method time steps
    '''

    def __init__(self, f, y0, h, t0, tf):
        ''' assign params
        y_0: [num] ; initial conditions
        f: defaultdict{int: fn} ; memoized vector field for ode y'=f
        h: num ; time step
        t_0 num ; initial time
        t_f num ; final time
        '''
        self.N = int(((tf-t0)/h)//1)
        self.t = [t0 + n*h for n in range(self.N+1)] # N+1 to accommodate t0, tf
        self.y = [np.array(y0)]
        self.f = [f(y0)]
        for n in range(self.N):
            self.step(h, n, f)

    def plotWithSolution(self, y, title):
        t = np.linspace(self.t[0], self.t[-1], 500)
        y = [y(_t) for _t in t]

        fig, ax = plt.subplots()
        ax.plot(self.t, [pt[0] for pt in self.y], 'r')
        ax.plot(t, y, 'b')
        ax.set(xlabel='t', ylabel='y(t)', title=title)
        ax.legend(['Approximate', 'Actual'])
        return fig

    def maxError(self, y):
        ''' find max error between known solution y and self.y vals
        '''
        return max(abs(self.y[n][0] - y(self.t[n])) for n in range(self.N))


class EulersMethod(NumericalMethod):
    ''' Implementation of Eulers Method
    '''
    def step(self, h, n, f):
        ''' Calculate the (n+1)th step and append it onto self.y, self.f
        '''
        self.y.append(self.y[n] + h*self.f[n])
        self.f.append(f(self.y[-1]))


class ThetaMethod(NumericalMethod):
    ''' Implementation of Theta Method
    '''
    def __init__(self, theta, *args):
        self.theta = theta
        super(ThetaMethod, self).__init__(*args)

    def step(self, h, n, f):
        ''' Calculate the (n+1)th step and append it onto self.y, self.f
        '''
        y1, y2 = symbols('y1, y2')
        y = np.array([y1, y2])
        step = np.array(next(iter(linsolve([
            y1 - self.y[n][0] - h*self.theta*self.f[n][0] - h*(1-self.theta)*f(y)[0],
            y2 - self.y[n][1] - h*self.theta*self.f[n][1] - h*(1-self.theta)*f(y)[1]
            ],
            (y1, y2)
        ))))
        self.y.append(step)
        self.f.append(f(step))

