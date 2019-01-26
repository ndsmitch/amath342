from methods import EulersMethod, ThetaMethod
import numpy as np
from math import exp
from functools import partial

def f(x):
    ''' ODE function for y'=f
    Q3-Q4) y'' + 10y' + 25y = 0
    '''
    return np.array([
        x[1],
        -10*(x[1] + 2.5 * x[0])
    ])

def y(t):
    ''' Exact solution to y'=f
    Q3-Q4) y'' + 10y' + 25y = 0
    '''
    return exp(-5*t) * (5*t + 1)

def write_max_error(e_file, method, error):
    e_file.write('{0}\n{1}\n'.format(method, error))

def plot_solution(h, title, Method, q, e_file):
    sol = Method(f, [1,0], h, 0, 1)
    _h = str(h).replace('.','')
    sol.plotWithSolution(y, title.format(h)).savefig('a01/q{0}_h{1}.png'.format(q, _h))
    write_max_error(e_file, '{0} ; h={1}'.format(type(sol).__name__, h), sol.maxError(y))

if __name__ == "__main__":
    e_file = open('a01/q3q4_maxerror.txt', 'w')
    for h in (0.5, 0.05, 0.005):
        title = 'Actual vs Approximated y(t); Euler\'s Method with h={0}'
        plot_solution(h, title, EulersMethod, '3', e_file)
    
        title = 'Actual vs Approximated y(t); Theta Method with h={0}'
        Method = partial(ThetaMethod, 0.5)  # theta = 0.5
        plot_solution(h, title, Method, '4', e_file)
    e_file.close()
    
