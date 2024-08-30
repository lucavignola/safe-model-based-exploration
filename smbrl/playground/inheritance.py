import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Specify the number of curves
n_curves = 10

# Create a colormap
cmap = cm.viridis  # You can also use other colormaps like cm.plasma, cm.inferno, etc.
norm = mcolors.Normalize(vmin=0, vmax=n_curves-1)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Plot each curve with a color from the colormap
fig, ax = plt.subplots()
for i in range(n_curves):
    ax.plot(x, y + i, color=cmap(norm(i)), label=f'Curve {i + 1}')

# Add a colorbar
cbar = plt.colorbar(sm, ax=ax, ticks=range(n_curves))
cbar.set_label('Curve Index')
cbar.set_ticks(np.arange(n_curves))
cbar.set_ticklabels([f'{i + 1}' for i in range(n_curves)])

# Show the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot with Colormap and Colorbar')
plt.show()