import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your CSV data as a multiline string (for demonstration)
csv_data = """Filename,Depths,Percent White Pixels
MASK_Sk658 Llobe ch010017.jpg,45,0.66
MASK_Sk658 Llobe ch010018.jpg,90,0.8
MASK_Sk658 Llobe ch010019.jpg,60,0.75
MASK_Sk658 Llobe ch010021.jpg,30,0.57
MASK_Sk658 Llobe ch010022.jpg,80,0.79
MASK_Sk658 Llobe ch010023.jpg,100,0.89"""

# Read CSV from string (in real life, you'd do pd.read_csv("your_file.csv"))
from io import StringIO
df = pd.read_csv(StringIO(csv_data))

# Sort data by Depths (optional but helps with plotting nicely)
df = df.sort_values('Depths')

# Extract x and y points
x_points = df['Depths'].values
y_points = df['Percent White Pixels'].values

print("Data points:")
for x, y in zip(x_points, y_points):
    print(f"Depth: {x}, Percent White Pixels: {y}")

# ---- Linear interpolation ----
# Pick two points (e.g. depths 30 and 100)
x_lin = x_points[[0, -1]]  # smallest and largest depths
y_lin = y_points[[0, -1]]

# Build matrix and solve for a1 + a2*x (linear)
Z_lin = np.array([[1, x_lin[0]], [1, x_lin[1]]])
Y_lin = np.array([y_lin[0], y_lin[1]])
A_lin = np.linalg.solve(Z_lin, Y_lin)
a1_lin, a2_lin = A_lin
print("\nLinear coefficients:", A_lin)

# Linear interpolation function
def linear_func(x):
    return a1_lin + a2_lin * x

# ---- Quadratic interpolation ----
# Pick three points (e.g. depths 30, 60, 100)
x_quad = x_points[[0, 2, -1]]
y_quad = y_points[[0, 2, -1]]

Z_quad = np.array([
    [1, x_quad[0], x_quad[0]**2],
    [1, x_quad[1], x_quad[1]**2],
    [1, x_quad[2], x_quad[2]**2]
])
Y_quad = np.array([y_quad[0], y_quad[1], y_quad[2]])

A_quad = np.linalg.solve(Z_quad, Y_quad)
a1_quad, a2_quad, a3_quad = A_quad
print("\nQuadratic coefficients:", A_quad)

# Quadratic interpolation function
def quad_func(x):
    return a1_quad + a2_quad*x + a3_quad*x**2

# ---- Plot ----
x_plot = np.linspace(min(x_points) - 10, max(x_points) + 10, 400)

plt.figure(figsize=(10,6))

# Plot original data points
plt.scatter(x_points, y_points, color='blue', s=60, label='Original Data')

# Plot linear interpolation line
plt.plot(x_plot, linear_func(x_plot), label='Linear Interpolation', color='green')

# Plot quadratic interpolation curve
plt.plot(x_plot, quad_func(x_plot), label='Quadratic Interpolation', color='red')

plt.title('Linear vs Quadratic Interpolation on Percent White Pixels vs Depth')
plt.xlabel('Depth')
plt.ylabel('Percent White Pixels')
plt.grid(True)
plt.legend()
plt.show()
