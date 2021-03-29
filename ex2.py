
#  min      x1^2 + x2^2
#  s.t.     ...
#  var      x1, x2


#%%
from scipy.optimize import minimize
from numdifftools import Jacobian

print('\nSOLVING USING SCIPY\n')

# Objective function
fun = lambda x: x[0]**2 + x[1]**2 

# Jacobian
fun_Jac = lambda x: Jacobian(lambda x: fun(x))(x).ravel()

# constraints
cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 0.5},
        {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1.},
        {'type': 'ineq', 'fun': lambda x: x[0]**2 + x[1]**2 - 1.},
        {'type': 'ineq', 'fun': lambda x: 9*x[0] + x[1] - 9.},
        {'type': 'ineq', 'fun': lambda x: x[0]**2 - x[1]},
        {'type': 'ineq', 'fun': lambda x: -x[0] + x[1]**2}
        )


# bounds, if any, e.g. x1 and x2 have to be positive
bnds = ((None, None), (None, None))
bnds = ((None, None), )*2

# initial guess
x0 = (10,10)

# Method SLSQP uses Sequential Least SQuares Programming to minimize a function 
# of several variables with any combination of bounds, equality and inequality constraints. 

res = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)
print(res)
print("optimal value p*", res.fun)
print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])

res2 = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons,jac=fun_Jac)
print('\n',res2)
print("JAC: optimal value p*", res2.fun)
print("JAC: optimal var: x1 = ", res2.x[0], " x2 = ", res2.x[1])
