
#  min      ...
#  s.t.     ...
#  var      x

import numpy as np
from numdifftools import Derivative

def dx(f, x):
    return abs(0-f(x))
 
def newton(f, df, x0, epsilon):
    delta = dx(f, x0)
    while delta > epsilon:
        x0 = x0 - f(x0)/df(x0)
        delta = dx(f, x0)
    print('Root at: ', x0)
    print('f(x) at root: ', f(x0))
    return delta

def f(x):
    return 2*x**4 - 4*x**2 + x - 0.5
 
def df(x):
    return 8*x**3 - 8*x + 1

x0s = [-2., -0.5, 0.5, 2.]
for x0 in x0s:
    newton(f, df, x0, 1e-5)