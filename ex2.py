
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
x0 = (10,10) # feasible initian point
# x0 = (0,0) # non-feasible initian point

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


## Plots
# %%
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
f = lambda x: x[0]**2 + x[1]**2 

x = np.linspace(-1, 1, 30)
y = np.linspace(-1, 1, 30)

X, Y = np.meshgrid(x, y)
Z = f([X,Y])
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none', alpha=.8)
# ax.scatter(1, 1, color='black')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f')
ax.view_init(50, 135)

# %%
