import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cylinder parameters
R = 1.0  # radius of the cylinder
H = 5.0  # height of the cylinder (adjust this as needed)

# Example 2D curve data
# Here `x` ranges from 0 to 2*pi and `y` ranges from -H/2 to H/2
x_2d = np.linspace(0, 2 * np.pi * R, 100)
y_2d = np.sin(x_2d / R) * (H / 2)  # Example function to project onto 2D

# Convert 2D coordinates (x_2d, y_2d) to 3D coordinates (X, Y, Z)
theta = x_2d / R
z = y_2d

X = R * np.cos(theta)
Y = R * np.sin(theta)
Z = z

# Plotting the curve on the cylinder
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(X, Y, Z, label='Curve on cylinder')

# Plotting the cylinder surface for better visualization
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(-H / 2, H / 2, 50)
U, V = np.meshgrid(u, v)

X_cylinder = R * np.cos(U)
Y_cylinder = R * np.sin(U)
Z_cylinder = V

ax.plot_surface(X_cylinder, Y_cylinder, Z_cylinder, alpha=0.3, color='gray')

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()