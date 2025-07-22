import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

indices = ["0020", "0040", "0060", "0080", "0100", "0120", "0140", "0160"]

def read_contour(file_path):
    x_coords, y_coords = [], []
    with open(file_path, 'r') as file:
        for line in file:
            x_str, y_str = line.strip().split()
            x_coords.append(float(x_str))
            y_coords.append(float(y_str))
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])
    return x_coords, y_coords

# Store all coordinates to find global limits
all_x, all_y = [], []

contours = []
for index in indices:
    icontour_file = f"/data.lfpn/ibraun/Code/paper_volume_calculation/Patient_Segmentations/SCD_ManualContours/Healthy/SC-N-02/contours-manual/IRCCI-expert/IM-0001-{index}-icontour-manual.txt"
    ocontour_file = f"/data.lfpn/ibraun/Code/paper_volume_calculation/Patient_Segmentations/SCD_ManualContours/Healthy/SC-N-02/contours-manual/IRCCI-expert/IM-0001-{index}-ocontour-manual.txt"
    
    ix, iy = read_contour(icontour_file)
    ox, oy = read_contour(ocontour_file)
    
    all_x.extend(ix + ox)
    all_y.extend(iy + oy)
    
    contours.append(((ix, iy), (ox, oy)))

# Compute global axis limits with some padding
padding = 5
x_min, x_max = min(all_x) - padding, max(all_x) + padding
y_min, y_max = min(all_y) - padding, max(all_y) + padding

# Create subplots
fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 4))

if len(indices) == 1:
    axes = [axes]

for i, index in enumerate(indices):
    (ix, iy), (ox, oy) = contours[i]
    
    ax = axes[i]
    ax.plot(ix, iy, 'r-')
    ax.plot(ox, oy, 'b-')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(f"Frame {index}")

    # Set shared axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # y is inverted


plt.tight_layout()
plt.savefig("Segmentation_masks_SC_N_02.png")
plt.show()
