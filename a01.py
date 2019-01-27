from functools import partial
from math import exp
from methods import EulersMethod, ThetaMethod
from numpy import array as vec

def Q3():
    ''' for 3 h-values, plot the theta/euler method vs solution
    and find max error
    '''
    # ODE function for y'=f ; y'' + 10y' + 25y = 0
    f = lambda x: vec([ x[1], -10*(x[1] + 2.5 * x[0]) ])
    # Exact solution to y'=f ; y'' + 10y' + 25y = 0
    y = lambda t: exp(-5*t) * (5*t + 1)

    e_file = open('a01/q3_maxerror.txt', 'w')
    for h in (0.5, 0.05, 0.005):
        sol = EulersMethod(
            f=f,
            y0=vec([1,0]),
            h=h,
            t0=0,
            tf=1
        )
        sol.plot_coord(
            title='Actual vs Approximated y(t); ' \
                  'Euler\'s Method with h={0}'.format(h),
            filename='a01/q3c_h{0}.png'.format(str(h).replace('.', '')),
            y=y
        )
        e_file.write('EulersMethod ; h={0}\n{1}\n'.format(h, sol.maxError(y)))
    
        sol = ThetaMethod(
            theta=0.5,
            f=f,
            y0=vec([1,0]),
            h=h,
            t0=0,
            tf=1
        )
        sol.plot_coord(
            title='Actual vs Approximated y(t); '
                  'Theta Method with h={0}'.format(h),
            filename='a01/q3e_h{0}.png'.format(str(h).replace('.', '')),
            y=y
        )
        e_file.write('ThetaMethod ; h={0}\n{1}\n'.format(h, sol.maxError(y)))

    e_file.close()

def Q4():
    ''' for 2 rho values, plot the euler method approximations
    for Lorenz equations
    '''
    # ODE function for y'=f ; Lorenz equations
    _f = lambda rho, x, y, z: vec([ 10*(y-x), x*(rho - z) - y, x*y - (8/3)*z ])
    
    for rho in (14, 28):
        f = lambda x: _f(rho, x[0], x[1], x[2])
        EulersMethod(
            f=f,
            y0=vec([1,0,0]),
            h=0.002,
            t0=0,
            tf=50
        ).plot_coord(
            title='Approximated z(x); ' \
                  'Euler\'s Method with rho={0}'.format(rho),
            filename='a01/q4_rho{0}.png'.format(rho),
            x_coord=0,  # plot x on horz
            y_coord=2,  # plot z on vert
            xlabel='x(t)',
            ylabel='z(t)'
        )

if __name__ == "__main__":
    Q3()
    Q4()

