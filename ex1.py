
#  min      ...
#  s.t.     ...
#  var      ...


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
x0s = [(0,0),
      (10,20),
      (-10,1),
      (-30,-30)
      ]

# Method SLSQP uses Sequential Least SQuares Programming to minimize a function 
# of several variables with any combination of bounds, equality and inequality constraints. 

for x0 in x0s:
      res1 = minimize(fun, x0[0], method='SLSQP', bounds=bnds, constraints=cons)
      print('\n',res1)
      print("optimal value p*", res1.fun)
      print("optimal var: x1 = ", res1.x[0], " x2 = ", res1.x[1])

for x0 in x0s:
      res2 = minimize(fun, x0[0], method='SLSQP', bounds=bnds, constraints=cons,jac=fun_Jac)
      print('\n',res2)
      print("JAC: optimal value p*", res2.fun)
      print("JAC: optimal var: x1 = ", res2.x[0], " x2 = ", res2.x[1])




## Plots
# %%
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
f = lambda x: np.exp(x[0]) * (4*x[0]**2 + 2*x[1]**2 + 4*x[0]*x[1] + 2*x[1] + 1)

x = np.linspace(-15, 15, 30)
y = np.linspace(-50, 20, 30)

X, Y = np.meshgrid(x, y)
Z = f([X,Y])
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none', alpha=.8)
ax.scatter(-9.54,1.04,0.02, color='black')
ax.scatter(1.18,-1.73,3.06, color='black')
ax.scatter(-9.54,1.04,0.02, color='black')
ax.scatter(1.06,-6.45,141.03, color='black')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f')
ax.view_init(50, 135)

# %%
