import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load the CSV file
df_offset = pd.read_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Results_Monkey/Segment_volumes_new_bound_offset_MRI_reso.csv")  # Replace with your actual file path
df_no_top = pd.read_csv('/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Results_Monkey/Segment_volumes_new_bound_no_top_MRI_reso.csv')  # Replace with your actual file path
df_all = pd.read_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Results_Monkey/Segment_volumes_new_bound_all_MRI_reso.csv")  # Replace with your actual file path

fil_df_offset = df_offset[df_offset['distance'] > 0.0]
#fil_df_offset_high = df_offset_high[df_offset_high['distance'] > 0.0]

fil_df_no_top = df_no_top[df_no_top['distance'] > 0.0]
fil_df_all = df_all[df_all['distance'] > 0.0]

# Set the style for a clean look without grid
plt.style.use('default')

# Create figure with appropriate size and spacing
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# Define a modern color palette
colors = {
    'no_top': '#118ab2ff',      # Modern blue
    'all_slices': '#ef476fff',  # JAHA red
    'offset': "#06d6a0ff",      # matching green that also colorblind people can differentiate
    'ground_truth': "#000000", # black
    'volume_mesh': "#868686",   # grey
    'MRI_reso': "#ffd166ff"
}

# print("W/O Slice")
# print(((fil_df_no_top['bp_vol'][397] - 8.58)/8.58)*100)

# print("All Slice")
# print(((fil_df_all['bp_vol'][397] - 8.58)/8.58)*100)

# print("With Offset")
# print(((fil_df_offset['bp_vol'][397] - 8.58)/8.58)*100)
# sys.exit()


# Plot the main data with improved styling
# W/O Top Slice
line1 = ax.plot(fil_df_no_top['distance'], fil_df_no_top['myo_vol'], 
                color=colors['no_top'], linewidth=2.5, 
                label="Without Top Slice", alpha=0.6, zorder=3)
scatter1 = ax.scatter(fil_df_no_top['distance'], fil_df_no_top['myo_vol'], 
                     color=colors['no_top'], s=50, alpha=0.5, linewidth=1, zorder=4)

# All Slices
line2 = ax.plot(fil_df_all['distance'], fil_df_all['myo_vol'], 
                color=colors['all_slices'], linewidth=2.5, 
                label="All Slices", alpha=0.6, zorder=3)
scatter2 = ax.scatter(fil_df_all['distance'], fil_df_all['myo_vol'], 
                     color=colors['all_slices'], s=50, alpha=0.5, linewidth=1, zorder=4)

# With Offset
line3 = ax.plot(fil_df_offset['distance'], fil_df_offset['myo_vol'], 
                color=colors['offset'], linewidth=2.5, 
                label="W Offset", alpha=0.6, zorder=3)
scatter3 = ax.scatter(fil_df_offset['distance'], fil_df_offset['myo_vol'], 
                     color=colors['offset'], s=50, alpha=0.5, linewidth=1, zorder=4)

# Ground truth lines with improved styling
#Blood pool volumes
# gt_line = ax.axhline(y=8.58, color=colors['ground_truth'], linewidth=2.5, 
#                      label='Ground Truth', alpha=0.9, zorder=2)
# vm_line = ax.axhline(y=8.65, color=colors['volume_mesh'], linewidth=2.5, 
#                      linestyle='--', label='Volume Meshes', alpha=0.9, zorder=2)

#Myocardium volumes
gt_line = ax.axhline(y=8.18, color=colors['ground_truth'], linewidth=2.5, 
                     label='Ground Truth', alpha=0.9, zorder=2)
vm_line = ax.axhline(y=8.20, color=colors['volume_mesh'], linewidth=2.5, 
                     linestyle='--', label='Volume Meshes', alpha=0.9, zorder=2)

MRI_line = ax.axvline(x=8.0, color=colors['MRI_reso'], linewidth=2.5, 
                    label='MRI Resolution', alpha=0.9, zorder=2)


# Enhance the axes
ax.set_xscale('log')
ax.invert_xaxis()

# Beautiful labels and title with better sizing
ax.set_xlabel('Distance [mm]', fontsize=16, fontweight='bold', color='#000000')
ax.set_ylabel('Volume [ml]', fontsize=16, fontweight='bold', color='#000000')
# ax.set_title('Blood Pool Volume Analysis', fontsize=18, fontweight='bold', 
#              color='#000000', pad=15)

ax.set_title('Myocardium Volume Analysis', fontsize=18, fontweight='bold', 
             color='#000000', pad=15)
# ax.set_ylim(0, 14.5)

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
plt.savefig("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Graphics/plot_calculated_myo_vol.pdf", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.2)
plt.savefig("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Graphics/plot_calculated_myo_vol.svg", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.2)
# plt.show()