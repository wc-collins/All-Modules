#%%

import numpy as np
import matplotlib.pyplot as plt

"""LINEAR INTERPOLATION"""

# Known points
x_points = [1, 3]
y_points = [1, 3]

x1, x2 = 1, 3
y1, y2 = 1, 3

# Make the Z and Y matrices
Z = np.array([[1, x1], [1, x2]])

Y = np.array([y1, y2])

# Compute the A matrix and print it
A = np.linalg.solve(Z, Y)

print("A = ", A)

# # Compute one specific point (x=2)
a1, a2 = A
x_value = 2
y_value = a1 + a2 * x_value
print("At x = 2, y =", y_value)

# # Plot the line
x = np.linspace(0, 5, 100)
y = a1 + a2 * x 
plt.plot(x, y)

# # Plot known points (as blue circles)
plt.scatter(x_points, y_points, color='blue', s=60, label='Known Points')

# # Plot the interpolated point (in red)
plt.scatter(x_value, y_value, color='red', zorder=5, label=f'Interpolated Point (x={x_value}, y={y_value:.2f})')

# # Add labels and grid
plt.title('Line with Interpolated Point (red)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
# %%
