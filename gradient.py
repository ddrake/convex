from numpy import *

def f(x):
    return x[0]**2 + x[0]*x[1] + x[1]**2 + x[0] - x[1]

def grad_f(x):
    gf = array([2*x[0]+x[1]+1,x[0]+2*x[1]-1])
    return gf

def gradient(max_gradf=1.0e-4, x0=[5,10], t=0.1):
    x1s = []
    x2s = []
    fs = []
    xk = x0
    gfk = grad_f(xk)
    gfk_n2 = gfk.dot(gfk)
    while gfk_n2 > max_gradf:
        gfk = grad_f(xk)
        gfk_n2 = gfk.dot(gfk)
        xk -= t*gfk
        fk = f(xk)
        x1s.append(xk[0])
        x2s.append(xk[1])
        fs.append(fk)
    return fs, x1s, x2s

# this should have faster convergence, but currently slightly slower.
def nesterov(max_gradf=1.0e-4, x0=[5,10], t=0.1):
    x1s = []
    x2s = []
    fs = []
    gs = []
    xk = x0
    yk = x0
    gfk = grad_f(xk)
    gfk_n2 = gfk.dot(gfk)
    tk = 1
    k = 1
    while gfk_n2 > max_gradf:
        gfk = grad_f(yk)
        xk1 = yk - t*gfk
        tk1 = (1.0 + sqrt(1.0 + 4.0*tk*tk))/2.0
        g = (tk-1)/tk1
        g = (k-2)/(k+1)
        yk = xk1 + g*(xk1 - xk)
        fk = f(xk)
        x1s.append(xk1[0])
        x2s.append(xk1[1])
        gs.append(g)
        fs.append(fk)
        tk = tk1
        xk = xk1
        gfk_n2 = gfk.dot(gfk)
        k+=1
    return fs, x1s, x2s, gs


