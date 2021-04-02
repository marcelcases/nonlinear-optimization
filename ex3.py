#  Exercise 3
#  TOML-MIRI
#  Marcel Cases
#  28-mar-2021
#
#  min      x1^2 + x2^2
#  s.t.     x1^2 + x1*x2 + x2^2 ≤ 3
#           3*x1 + 2*x2 ≥ 3
#  var      x1, x2

#%%
from scipy.optimize import minimize
import numpy as np
from numdifftools import Jacobian, Hessian
import time

print('\nSOLVING USING SCIPY\n')

# Objective function
fun = lambda x: x[0]**2 + x[1]**2

# Jacobian
fun_Jac = lambda x: Jacobian(lambda x: fun(x))(x).ravel()

# Hessian
fun_Hess = lambda x: Hessian(lambda x: fun(x))(x)

# constraints functions
cons = ({'type': 'ineq', 'fun': lambda x: -x[0]**2 - x[0]*x[1] - x[1]**2 + 3},
        {'type': 'ineq', 'fun': lambda x: 3*x[0] + 2*x[1] - 3})

# bounds, if any, e.g. x1 and x2 have to be positive
bnds = ((None, None), (None, None))
bnds = ((None, None), )*2

# initial guess
x0 = (10,10)
# x0 = (1,1)


# Method SLSQP uses Sequential Least SQuares Programming to minimize a function 
# of several variables with any combination of bounds, equality and inequality constraints. 

start_time = time.time()*1000
res = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)
end_time = time.time()*1000
print(res)
print("optimal value p*", res.fun)
print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])
print("exec time (ms): ", end_time - start_time)

start_time = time.time()*1000
res2 = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons,jac=fun_Jac)
end_time = time.time()*1000
print('\n',res2)
print("JAC: optimal value p*", res2.fun)
print("JAC: optimal var: x1 = ", res2.x[0], " x2 = ", res2.x[1])
print("exec time (ms): ", end_time - start_time)

start_time = time.time()*1000
res3 = minimize(fun, x0,  bounds=bnds, constraints=cons,jac=fun_Jac,hess=fun_Hess)
end_time = time.time()*1000
print('\n',res3)
print("JAC+HESS: optimal value p*", res3.fun)
print("JAC*HESS: optimal var: x1 = ", res3.x[0], " x2 = ", res3.x[1])
print("exec time (ms): ", end_time - start_time)


#%%
import cvxpy as cp
import numpy as np
import time

print('\nSOLVING USING CVXPY\n')

# Create two scalar optimization variables.
x = cp.Variable(2, name='x')

# Constraints
P1 = np.array(np.mat('1. 0.5; 0.5 1.'))
f1 = cp.quad_form(x, P1)
# f1 = x[0]**2+x[0]*x[1]+x[1]**2
f2 = 3.*x[0]+2.*x[1]
constraints = [f1<=3., f2>=3.]

# Form objective.
P0 = np.array(np.mat('1. 0.; 0. 1.'))
f0 = cp.quad_form(x, P0)
obj = cp.Minimize(f0)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
start_time = time.time()*1000
print("solve", prob.solve())  # Returns the optimal value.
end_time = time.time()*1000
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x[0].value, " x2 = ", x[1].value)
print("optimal dual variables lambda1 = ", constraints[0].dual_value)
print("optimal dual variables lambda2 = ", constraints[1].dual_value)
print("exec time (ms): ", end_time - start_time)

# %%
