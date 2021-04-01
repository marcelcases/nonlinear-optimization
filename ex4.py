#  Exercise 4
#  TOML-MIRI
#  Marcel Cases
#  28-mar-2021
#
#  min      x^2 + 1
#  s.t.     (x-2)(x-4) ≤ 0
#  var      x


#%%
import cvxpy as cp
import numpy as np

print('\nSOLVING USING CVXPY\n')

# Create two scalar optimization variables.
x = cp.Variable(1, name='x')

# Form objective.
f0 = x**2 + 1
obj = cp.Minimize(f0)

# Constraints
f1 = x**2 - 6*x + 8 # (x - 2)*(x - 4)
constraints = [f1<=0.]

# Form and solve problem.
prob = cp.Problem(obj, constraints)
print("solve", prob.solve())  # Returns the optimal value.
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x = ", x.value)
print("optimal dual variables lambda = ", constraints[0].dual_value)



## Plots
### plot2D
#%%
import matplotlib.pyplot as plt
import numpy as np
# 100 linearly spaced numbers
x_plot = np.linspace(-5,5,100)
# the function(s)
f0_plot = x_plot**2 + 1
f1_plot = x_plot**2 - 6*x_plot + 8
# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# plot the function(s)
plt.plot(x_plot, f0_plot, 'g', label="f0 (obj)", color="blue")
plt.plot(x_plot, f1_plot, 'g', label="f1 (constraint 1)", color="green")

# plot the minima of obj. func.
plt.plot(x.value, prob.value, marker='o', markersize=10, color="red")

# plot the legend
plt.legend()

# show the plot
plt.show()