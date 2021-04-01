#  Exercise 7 - Network Utility problem
#  TOML-MIRI
#  Marcel Cases
#  01-apr-2021
#
#  min      log x1 + log x2 + log x3
#  s.t.     x1 + x3 ≤ 1
#           x1 + x2 ≤ 2
#           x3 ≤ 1
#           x1, x2, x3 ≥ 0
#  var      x1, x2, x3

import cvxpy as cp

print('\nSOLVING USING CVXPY\n')

# Create two scalar optimization variables.
x = cp.Variable(3, name='x')

# Form objective.
f0 = cp.log(x[0]) + cp.log(x[1]) + cp.log(x[2])
obj = cp.Maximize(f0)

# Constraints
f1 = x[0] + x[2]
f2 = x[0] + x[1]
f3 = x[2]
f4, f5, f6 = x[0], x[1], x[2]
constraints = [f1<=1., f2<=2., f3<=1., f4>=0., f5>=0., f6>=0.]

# Form and solve problem.
prob = cp.Problem(obj, constraints)
print("solve", prob.solve())  # Returns the optimal value.
print("status:", prob.status)
print("optimal value p* =", prob.value)
print("optimal var: x1 =", x[0].value, " x2 =", x[1].value, " x3 =", x[2].value)
print("optimal dual variables lambda1 =", constraints[0].dual_value,
                                "  lambda2 =", constraints[1].dual_value,
                                "  lambda3 =", constraints[2].dual_value,
                                "  lambda4 =", constraints[3].dual_value,
                                "  lambda5 =", constraints[4].dual_value
                                )
