
#  min      ...
#  s.t.     ...
#  var      x

import numpy as np
from numdifftools import Derivative

def backtrack(dfx, x0, step):
    incumbent = x0 # result
    iters = 0
    acc = 1e-4
    while (acc >= 1e-4):
        newincumbent = incumbent - step*dfx(incumbent)
        acc = np.absolute(newincumbent - incumbent)
        incumbent = newincumbent
        iters += 1
    return incumbent, iters, acc, step

def show_results(func, incumbent, iters, acc, step):
    print("min f: ", func)
    print("at x = ", incumbent)
    print("iters: ", iters)
    print("acc:   ", acc)
    print("step:  ", step, "\n")



f = lambda x: 2*x**2 - 0.5
incumbent, iters, acc, step = backtrack(dfx = Derivative(f), x0 = 3., step = 0.001)
show_results(f(incumbent), incumbent, iters, acc, step)


f = lambda x: 2*x**4 - 4*x**2 + x - 0.5
x0s = [-2., -0.5, 0.5, 2.]
for x0 in x0s:
    incumbent, iters, acc, step = backtrack(dfx = Derivative(f), x0 = x0, step = 0.001)
    show_results(f(incumbent), incumbent, iters, acc, step)
