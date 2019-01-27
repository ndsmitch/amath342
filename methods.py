import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from math import floor
from numpy import array as vec
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
        self.t = [t0 + n*h for n in range(self.N+1)] # N+1 to include tf
        self.y = [y0]
        self.f = [f(y0)]
        for n in range(self.N):
            try:
                self.step(h, n, f)
            except:
                f_len = len(self.f)
                if f_len < len(self.y):  # rollback change
                    self.y = self.y[:f_len]
                self.t = self.t[:f_len]
                break

    def plot_first_coord(self, title, filename, y=None):
        ''' plot approximated first coordinate of the solution
        against provided function y(t) if provided
        '''
        fig, ax = plt.subplots()
        ax.plot(self.t, [pt[0] for pt in self.y], 'r')

        if y is not None:
            t = np.linspace(float(self.t[0]), float(self.t[-1]), 500)
            y = [y(_t) for _t in t]
            ax.plot(t, y, 'b')
            ax.legend(['Approximate', 'Actual'])

        ax.set(xlabel='t', ylabel='y(t)', title=title)
        fig.savefig(filename)
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
        try:
            self.f.append(f(self.y[-1]))
        except:
            print([h, n, self.y[-1]])
            print(self.y[:5])
            raise


class ThetaMethod(NumericalMethod):
    ''' Implementation of Theta Method
    '''
    def __init__(self, theta, f, y0, h, t0, tf):
        self.theta = theta
        super(ThetaMethod, self).__init__(f, y0, h, t0, tf)

    def step(self, h, n, f):
        ''' Calculate the (n+1)th step and append it onto self.y, self.f
        '''
        y1, y2 = symbols('y1, y2')
        y = np.array([y1, y2])
        theta = self.theta
        step = vec(next(iter(linsolve([
            y1 - self.y[n][0] - h*theta*self.f[n][0] - h*(1-theta)*f(y)[0],
            y2 - self.y[n][1] - h*theta*self.f[n][1] - h*(1-theta)*f(y)[1]
            ],
            (y1, y2)
        ))))
        self.y.append(vec(step))
        self.f.append(vec(f(step)))

