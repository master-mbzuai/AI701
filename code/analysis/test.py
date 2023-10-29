import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Sample data
x = [10, 20, 30, 50, 70, 90]
y = [0.6, 0.8, 0.9, 0.93, 0.94, 0.95]

# Normalize y-values between 0 and 1
norm = mcolors.Normalize(vmin=min(y), vmax=max(y))

# Choose the colormap
colormap = plt.cm.viridis

# Get the color for each dot
dot_colors = [colormap(norm(value)) for value in y]

# Create the scatter plot
plt.scatter(x, y, c=dot_colors, s=100)  # s sets the size of the dots

# Plot the colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap))
cbar.set_label('Parameters (Mcal)')

plt.xlabel('Number')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Compression - 200 epochs training')
plt.grid(True)
plt.show()