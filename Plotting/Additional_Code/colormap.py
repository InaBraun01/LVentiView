import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex
import colorsys
import numpy as np

# Gradient for plotting
gradient = np.linspace(0, 1, 256).reshape(1, -1)

# 1️⃣ Continuous colormap
colors_continuous = [
    "#ef476fff", "#ffd166ff", "#06d6a0ff", "#118ab2ff", "#073b4cff"
]
cmap_continuous = LinearSegmentedColormap.from_list("continuous_cmap", colors_continuous, N=256)

# 2️⃣ Diverging colormap
colors_diverging = ["#ff164d", "#fff6eb", "#00aee7"]
cmap_diverging = LinearSegmentedColormap.from_list("diverging_cmap", colors_diverging, N=256)

# 3️⃣ Discrete color colormap
cmap_discrete = ListedColormap(colors_continuous)

# 4️⃣ Black-and-white gradient
cmap_bw = plt.get_cmap("Greys")

# 5️⃣ Discrete colors in black-and-white
def hex_to_gray(hex_color):
    hex_color = hex_color[:7]
    r = int(hex_color[1:3], 16)/255
    g = int(hex_color[3:5], 16)/255
    b = int(hex_color[5:7], 16)/255
    gray = 0.2126*r + 0.7152*g + 0.0722*b
    return (gray, gray, gray)

gray_colors = [hex_to_gray(c) for c in colors_continuous]
cmap_discrete_bw = ListedColormap(gray_colors)

# Create figure with 5 subplots
fig, axes = plt.subplots(4, 1, figsize=(8, 8), constrained_layout=True)

# Plot each colormap
axes[0].imshow(gradient, aspect='auto', cmap=cmap_continuous)
axes[0].set_title("Continuous Colormap")
axes[0].axis('off')

axes[1].imshow(gradient, aspect='auto', cmap=cmap_diverging)
axes[1].set_title("Diverging Colormap")
axes[1].axis('off')

axes[2].imshow(gradient, aspect='auto', cmap=cmap_discrete)
axes[2].set_title("Discrete Colormap")
axes[2].axis('off')

axes[3].imshow(gradient, aspect='auto', cmap=cmap_discrete_bw)
axes[3].set_title("Discrete Colors in Black & White")
axes[3].axis('off')

plt.savefig("colormap.png")

plt.show()