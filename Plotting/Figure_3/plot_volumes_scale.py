import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the style for a clean look without grid
plt.style.use('default')

# Create figure with appropriate size and spacing
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)


df_offset = pd.read_csv('/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Results_Monkey/Segment_volumes_offset_MRI_reso_scale.csv')  # Replace with your actual file path
df_no_top = pd.read_csv('/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Results_Monkey/Segment_volumes_no_top_MRI_reso_scale.csv')  # Replace with your actual file path
df_all = pd.read_csv('/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Results_Monkey/Segment_volumes_all_MRI_reso_scale.csv')  # Replace with your actual file path

myo_ground_truth = 8.18 #for the mesh with scaling factor 1
bp_ground_truth = 8.58 #for the mesh with scaling factor 1

#########RELATIVE DIFFERENCES#######################################3
# y_offset = abs(df_offset['bp_vol'] - bp_ground_truth*df_offset['scale_factor']**3)/(bp_ground_truth*df_offset['scale_factor']**3)*100
# y_no_top = abs(df_no_top['bp_vol'] - bp_ground_truth*df_no_top['scale_factor']**3)/(bp_ground_truth*df_no_top['scale_factor']**3)*100
# y_all = abs(df_all['bp_vol'] - bp_ground_truth*df_all['scale_factor']**3)/(bp_ground_truth*df_all['scale_factor']**3)*100

##########SCALING FACTOR#####################################################
y_offset = bp_ground_truth*df_offset['scale_factor']**3/df_offset['bp_vol']
y_no_top = bp_ground_truth*df_no_top['scale_factor']**3/df_no_top['bp_vol']
y_all = bp_ground_truth*df_all['scale_factor']**3/df_all['bp_vol']

# Define a modern color palette
colors = {
    'no_top': '#118ab2ff',      # Modern blue
    'all_slices': '#ef476fff',  # JAHA red
    'offset': "#06d6a0ff",      # matching green that also colorblind people can differentiate
    'ground_truth': "#000000", # black
}

#W/O Top
line1 = ax.plot(bp_ground_truth*df_no_top['scale_factor'][2:]**3, y_no_top[2:], 
                color=colors['no_top'], linewidth=2.5, 
                label="Without Top Slice", alpha=0.6, zorder=3)
scatter1 = ax.scatter(bp_ground_truth*df_no_top['scale_factor'][2:]**3, y_no_top[2:], 
                     color=colors['no_top'], s=50, alpha=0.5, linewidth=1, zorder=4)

#ALl Slices
line2 = ax.plot(bp_ground_truth*df_all['scale_factor']**3, y_all, 
                color=colors['all_slices'], linewidth=2.5, 
                label="All Slices", alpha=0.6, zorder=3)
scatter2 = ax.scatter(bp_ground_truth*df_all['scale_factor']**3, y_all, 
                     color=colors['all_slices'], s=50, alpha=0.5, linewidth=1, zorder=4)

# With Offset
line3 = ax.plot(bp_ground_truth*df_offset['scale_factor']**3, y_offset, 
                color=colors['offset'], linewidth=2.5, 
                label="W Offset", alpha=0.6, zorder=3)
scatter3 = ax.scatter(bp_ground_truth*df_offset['scale_factor']**3, y_offset, 
                     color=colors['offset'], s=50, alpha=0.5, linewidth=1, zorder=4)

# Ground truth lines with improved styling
gt_line = ax.axhline(y=1.0, color=colors['ground_truth'], linewidth=2.5, 
                     label='Ground Truth', alpha=0.9, zorder=2)

# Beautiful labels and title with better sizing
ax.set_xlabel('Ground-Truth Volume [ml]', fontsize=16, fontweight='bold', color='#000000')
ax.set_ylabel('Scaling Factor', fontsize=16, fontweight='bold', color='#000000')
ax.set_title('Scaling Factor Volume Analysis', fontsize=18, fontweight='bold', 
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
legend = ax.legend(loc='best', frameon=False, fontsize=11, title_fontsize=12)
legend.get_title().set_fontweight('bold')

# Add clean white background
ax.set_facecolor('#FFFFFF')

# Use subplots_adjust for better control over spacing
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9)

# Save with high quality
plt.savefig("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Graphics/plot_bp_scaling_factor.pdf", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.2)
plt.savefig("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Graphics/plot_bp_scaling_factor.svg", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.2)
# plt.show()