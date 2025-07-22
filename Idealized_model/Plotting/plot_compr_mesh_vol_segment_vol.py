import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

df_meshes = pd.read_csv('/data.lfpn/ibraun/Code/paper_volume_calculation/outputs_healthy_GUI/Corbinian/mesh_volumes_cal.csv')
df_seg_masks_gen = pd.read_csv('/data.lfpn/ibraun/Code/paper_volume_calculation/outputs_healthy_GUI/Corbinian/Bp_volumes_gen_seg_masks.csv')
df_seg_masks = pd.read_csv('/data.lfpn/ibraun/Code/paper_volume_calculation/outputs_healthy_GUI/Corbinian/Bp_volumes_slices.csv')

bp_seg_masks = df_seg_masks["bp_vol"]
bp_seg_masks_gen = df_seg_masks_gen["bp_vol"]

bp_meshes = df_meshes["bp_volume"]

time_step = df_meshes["Time"]

# Set the style for a clean look without grid
plt.style.use('default')

# Create figure with appropriate size and spacing
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# Define a modern color palette
colors = {
    'meshes': '#4B6FA5',      # Modern blue
    'seg_masks': '#C10E21',  # JAHA red
    'seg_masks_gen': "#2E8B57"
}

# Volumes from meshes
line1 = ax.plot(time_step, bp_meshes, 
                color=colors['meshes'], linewidth=2.5, 
                label="Volume from meshes", alpha=0.6, zorder=3)
scatter1 = ax.scatter(time_step, bp_meshes, 
                     color=colors['meshes'], s=50, alpha=0.5, linewidth=1, zorder=4)


# Volumes from segmentation masks
line2 = ax.plot(time_step, bp_seg_masks, 
                color=colors['seg_masks'], linewidth=2.5, 
                label="Volume from segmentation", alpha=0.6, zorder=3)
scatter2 = ax.scatter(time_step, bp_seg_masks, 
                     color=colors['seg_masks'], s=50, alpha=0.5, linewidth=1, zorder=4)

# # Volumes from sliced volumetric mesh
# line3 = ax.plot(time_step, bp_seg_masks_gen, 
#                 color=colors['seg_masks_gen'], linewidth=2.5, 
#                 label="Volume from sliced mesh", alpha=0.6, zorder=3)
# scatter3 = ax.scatter(time_step, bp_seg_masks_gen, 
#                      color=colors['seg_masks_gen'], s=50, alpha=0.5, linewidth=1, zorder=4)


# Beautiful labels and title with better sizing
ax.set_xlabel('Distance [mm]', fontsize=16, fontweight='bold', color='#000000')
ax.set_ylabel('Volume [ml]', fontsize=16, fontweight='bold', color='#000000')
ax.set_title('Blood Pool Volume Analysis', fontsize=18, fontweight='bold', 
             color='#000000', pad=15)

# Improve tick styling
ax.tick_params(axis='both', which='major', labelsize=11, colors='#000000')

# Remove grid and clean up spines
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#000000')
ax.spines['bottom'].set_color('#000000')

# Beautiful legend with better positioning - no border
legend = ax.legend(loc='best', frameon=False, fontsize=14, title_fontsize=14)
legend.get_title().set_fontweight('bold')

# Add clean white background
ax.set_facecolor('#FFFFFF')

# Use subplots_adjust for better control over spacing
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9)

# Save with high quality
plt.savefig("/data.lfpn/ibraun/Code/paper_volume_calculation/outputs_healthy_GUI/Corbinian/plot_compr_bp.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.2)
plt.show()