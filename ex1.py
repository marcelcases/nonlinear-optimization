
#  min      ...
#  s.t.     ...
#  var      ...


#%%
from scipy.optimize import minimize
import numpy as np
from numdifftools import Jacobian

# Objective function
fun = lambda x: np.exp(x[0]) * (4*x[0]**2 + 2*x[1]**2 + 4*x[0]*x[1] + 2*x[1] + 1)

# Jacobian
fun_Jac = lambda x: Jacobian(lambda x: fun(x))(x).ravel()

# constraints
cons = ({'type': 'ineq', 'fun': lambda x: -x[0]*x[1] + x[0] + x[1] - 1.5},
        {'type': 'ineq', 'fun': lambda x: x[0]*x[1] + 10}
        )


# bounds, if any, e.g. x1 and x2 have to be positive
bnds = ((None, None), (None, None))
bnds = ((None, None), )*2

# initial guesses
x0 = [(0,0),
      (10,20),
      (-10,1),
      (-30,-30)
      ]

# Method SLSQP uses Sequential Least SQuares Programming to minimize a function 
# of several variables with any combination of bounds, equality and inequality constraints. 

res = minimize(fun, x0[0], method='SLSQP', bounds=bnds, constraints=cons)
print(res)
print("optimal value p*", res.fun)
print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])

res2 = minimize(fun, x0[0], method='SLSQP', bounds=bnds, constraints=cons,jac=fun_Jac)
print('\n',res2)
print("JAC: optimal value p*", res2.fun)
print("JAC: optimal var: x1 = ", res2.x[0], " x2 = ", res2.x[1])
