
#  min      ...
#  s.t.     ...
#  var      x


#%%
import cvxpy as cp
import numpy as np

print('\nSOLVING USING CVXPY\n')

# Create two scalar optimization variables.
x = cp.Variable(1, name='x')

# Constraints
f1 = (x - 2)*(x - 4)
constraints = [f1<=0.]

# Form objective.
f0 = x**2 + 1
obj = cp.Minimize(f0)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
print("solve", prob.solve())  # Returns the optimal value.
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x.value)
print("optimal dual variables lambda1 = ", constraints[0].dual_value)
