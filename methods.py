from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from math import floor, sqrt
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
        self.f = [f(t0, y0)]
        for n in range(self.N):
            self.step(h, n, f)

    def plot_coord(self, title, filename,
                   y=None, x_coord=None, y_coord=0, xlabel='t', ylabel='y(t)'):
        ''' plot approximated first coordinate of the solution
        against provided function y(t) if provided
        '''
        x_pts = self.t if x_coord is None else [pt[x_coord] for pt in self.y]

        fig, ax = plt.subplots()
        ax.plot(x_pts, [pt[y_coord] for pt in self.y], color='black')

        if y is not None:
            t = np.linspace(float(self.t[0]), float(self.t[-1]), 500)
            y = [y(_t) for _t in t]
            ax.plot(t, y, 'b')
            ax.legend(['Approximate', 'Actual'])

        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        fig.savefig(filename)
        return fig

    def plot_mag(self, title, filename,
                 xlabel='t', ylabel='|y(t)|'):
        ''' plot approximated magnitude the solution
        '''
        mag = lambda y: sqrt(sum(x*x for x in y))
        x_pts = self.t

        fig, ax = plt.subplots()
        ax.plot(x_pts, [mag(pt) for pt in self.y], color='black')
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        fig.savefig(filename)
        return fig

    def plot3D(self, title, filename):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = [y[0] for y in self.y]
        ys = [y[1] for y in self.y]
        zs = [y[2] for y in self.y]
        ax.plot(xs, ys, zs, color='black', linewidth=0.5)
        ax.set(xlabel='x(t)', ylabel='y(t)', zlabel='z(t)', title=title)
        fig.savefig(filename)

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
        self.f.append(f(self.t[n+1], self.y[n+1]))


class ThetaMethod(NumericalMethod):
    ''' Implementation of Theta Method
    '''
    def __init__(self, theta, f, y0, h, t0, tf):
        self.theta = theta
        super(ThetaMethod, self).__init__(f, y0, h, t0, tf)

    def step(self, h, n, f):
        ''' Calculate the (n+1)th step and append it onto self.y, self.f
        '''
        y1, y2, y3, y4 = symbols('y1, y2, y3, y4')
        y = np.array([y1, y2, y3, y4])
        theta = self.theta
        step = vec(next(iter(linsolve([
            y[q] - self.y[n][q] - h*theta*self.f[n][q] - h*(1-theta)*f(self.t[n], y)[q]
            for q in range(4)
            ],
            (y1, y2, y3, y4)
        ))))
        self.y.append(vec(step))
        self.f.append(vec(f(self.t[n+1], step)))


class NystromTwoStepMethod(EulersMethod):
    ''' Implementation of two step Nystrom Method
    '''
    def step(self, h, n, f):
        ''' Calculate the (n+1)th step and append it onto self.y, self.f
        '''
        if not n:  # n is zero, use Euler's method for first step
            return super(NystromTwoStepMethod, self).step(h, n, f)

        self.y.append(self.y[n-1] + h*2*self.f[n])
        self.f.append(f(self.t[n+1], self.y[n+1]))


class AdamsBashforthThreeStepMethod(EulersMethod):
    ''' Implementation of three step Adams-Bashforth Method
    '''
    def step(self, h, n, f):
        ''' Calculate the (n+1)th step and append it onto self.y, self.f
        '''
        if n < 2:  # use Euler's method for y1, y2
            return super(AdamsBashforthThreeStepMethod, self).step(h, n, f)

        self.y.append(self.y[n-1] + h*(
            (5/12)*self.f[n-3] - (4/3)*self.f[n-2] + (23/12)*self.f[n-1]
        ))
        self.f.append(f(self.t[n+1], self.y[n+1]))


class BDFTwoStepMethod(ThetaMethod):
    ''' Implementation of two-step BDF Method
    First step Calculated with Backward Euler method (theta=0)
    '''
    def __init__(self, f, y0, h, t0, tf):
        ''' Call ThetaMethod initializer with theta = 0 (Backward Euler)
        '''
        super(BDFTwoStepMethod, self).__init__(0, f, y0, h, t0, tf)

    def step(self, h, n, f):
        ''' Calculate the (n+1)th step and append it onto self.y, self.f
        Implemenation for 4D state variable
        '''
        if not n:  # use Backward Euler method for y1
            return super(BDFTwoStepMethod, self).step(h, n, f)

        y1, y2, y3, y4 = symbols('y1, y2, y3, y4')
        y = np.array([y1, y2, y3, y4])
        step = vec(next(iter(linsolve([
            y[q] - (4/3)*self.y[n][q] + (1/3)*self.y[n-1][q] - (2/3)*h*f(self.t[n], y)[q]
            for q in range(4)
            ],
            (y1, y2, y3, y4)
        ))))
        self.y.append(vec(step))
        self.f.append(vec(f(self.t[n+1], step)))


class ERKThreeStageMethod(NumericalMethod):
    def __init__(self, f, y0, h, t0, tf):
        ''' Set RK Params then initialize numerical method
        '''
        # Pad 0 index with None for linear algebra formalism (index from 1)
        self.a = [None,
                  [None, 0,   0,   0],
                  [None, 2/3, 0,   0],
                  [None, 0,   2/3, 0]]
        self.b = [None, 1/4, 3/8, 3/8]
        self.c = [None, 0, 2/3, 2/3]

        super(ERKThreeStageMethod, self).__init__(f, y0, h, t0, tf)

    def step(self, h, n, f):
        ''' Calculate the (n+1)th step and append it onto self.y, self.f
        '''
        # k_j = y[n] + h(sum{i=1, j-1} a_ji * f(t[n] + c_i*h, k_i))
        k = {}
        def _k(j):
            return self.y[n] + h*sum(
                self.a[j][i]*f(self.t[n]+self.c[i], k[i])
                for i in range(1, j)  # i = 1, 2, ... j-1
            )
        k[1] = _k(1)
        k[2] = _k(2)
        k[3] = _k(3)
        y = self.y[n] + h*sum(
            self.b[i]*f(self.t[n]+self.c[i], k[i])
            for i in range(1, 4)  # i = 1, 2, 3
        )
        self.y.append(y)
        self.f.append(f(self.t[n+1], self.y[n+1]))
