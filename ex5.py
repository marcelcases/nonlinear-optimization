
#  min      ...
#  s.t.     ...
#  var      x

#%%
import cvxpy as cp
import numpy as np

print('\nSOLVING USING CVXPY\n')

# Create two scalar optimization variables.
x = cp.Variable(2, name='x')

# Form objective.
f0 = x[0]**2 + x[1]**2
obj = cp.Minimize(f0)

# Constraints
f1 = (x[0] - 1)**2 + (x[1] - 1)**2
f2 = (x[0] - 1)**2 + (x[1] + 1)**2
constraints = [f1<=1., f2<=1.]

# Form and solve problem.
prob = cp.Problem(obj, constraints)
print("solve", prob.solve())  # Returns the optimal value.
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x[0].value, " x2 = ", x[1].value)
print("optimal dual variables lambda1 = ", constraints[0].dual_value)
print("optimal dual variables lambda2 = ", constraints[1].dual_value)
