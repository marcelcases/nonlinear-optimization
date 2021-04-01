#  Exercise 6 - Gradient Descent Methods, Backtracking Line Search and Newton's method (unconstrained)
#  TOML-MIRI
#  Marcel Cases
#  01-apr-2021
#
#  min(1)   f(x) = 2*x^2 - 0.5,   var  x
#  min(2)   f(x) = 2*x^4 - 4*x^2 + x - 0.5,   var  x

#%%
#  Backtracking Line Search
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



#%%
#  Newton's Method for optimization (SLSQP solver from Scipy uses this method)

from scipy.optimize import minimize
import numpy as np
from numdifftools import Jacobian, Hessian

print('\nSOLVING USING SCIPY\n')

# First function to optimize
fun = lambda x: 2*x**2 - 0.5 # objective function
fun_Jac = lambda x: Jacobian(lambda x: fun(x))(x).ravel() # Jacobian
fun_Hess = lambda x: Hessian(lambda x: fun(x))(x) # Hessian
cons = () # unconstrained
bnds = ((None, None), )*1 # unbounded

# initial guess
x0 = 3.0

res = minimize(fun, x0, bounds=bnds, constraints=cons, jac=fun_Jac, hess=fun_Hess)
print('\n',res)
print("JAC+HESS: optimal value p*", res.fun)
print("JAC+HESS: optimal var: x = ", res.x)


# Second function to optimize
fun = lambda x: 2*x**4 - 4*x**2 + x - 0.5 # objective function
fun_Jac = lambda x: Jacobian(lambda x: fun(x))(x).ravel() # Jacobian
fun_Hess = lambda x: Hessian(lambda x: fun(x))(x) # Hessian
cons = () # unconstrained
bnds = ((None, None), )*1 # unbounded

# initial guesses
x0s = [-2., -0.5, 0.5, 2.]

for x0 in x0s:
    res = minimize(fun, x0, bounds=bnds, constraints=cons, jac=fun_Jac, hess=fun_Hess)
    print('\n',res)
    print("JAC+HESS: optimal value p*", res.fun)
    print("JAC+HESS: optimal var: x = ", res.x)



#%%
#  Newton's Method for finding roots

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
# %%
