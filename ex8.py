#  Resource Allocation problem
#  min      ...
#  s.t.     ...
#  var      x

import cvxpy as cp

print('\nSOLVING USING CVXPY\n')

# Create two scalar optimization variables.
x1,x2,x3 = cp.Variable(1, name='x1'), cp.Variable(1, name='x2'), cp.Variable(1, name='x3')
r12,r23,r32 = cp.Variable(1, name='r12'), cp.Variable(1, name='r23'), cp.Variable(1, name='r32')

# Form objective.
f0 = cp.log(x1) + cp.log(x2) + cp.log(x3)
obj = cp.Maximize(f0)

# Constraints
f1 = x1 + x2 - r12
f2 = x1 - r23
f3 = x3 - r32
f4 = r12 + r23 + r32
constraints = [f1<=0., f2<=0., f3<=0., f4<=1.]

# Form and solve problem.
prob = cp.Problem(obj, constraints)
print("solve", prob.solve())  # Returns the optimal value.
print("status:", prob.status)
print("optimal value p* =", prob.value)
print("optimal var: x1 =", x1.value, " x2 =", x2.value, " x3 =", x3.value, " r12 =", r12.value, " r23 =", r23.value, " r32 =", r32.value)
print("optimal dual variables lambda1 =", constraints[0].dual_value,
                                "  lambda2 =", constraints[1].dual_value,
                                "  lambda3 =", constraints[2].dual_value,
                                "  u1 =", constraints[3].dual_value
                                )
