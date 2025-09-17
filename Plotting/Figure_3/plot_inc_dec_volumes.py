import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

df_offset_de = pd.read_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Results_Monkey/Segment_volumes_new_bound_offset_MRI_reso_decre_bp.csv")  # Replace with your actual file path
df_no_top_de = pd.read_csv('/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Results_Monkey/Segment_volumes_new_bound_no_top_MRI_reso_decre_bp.csv')  # Replace with your actual file path
df_all_de = pd.read_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Results_Monkey/Segment_volumes_new_bound_all_MRI_reso_decre_bp.csv")  # Replace with your actual file path

df_offset_in = pd.read_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Results_Monkey/Segment_volumes_new_bound_offset_MRI_reso_incre_bp.csv")  # Replace with your actual file path
df_all_in = pd.read_csv('/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Results_Monkey/Segment_volumes_new_bound_all_MRI_reso_incre_bp.csv')  # Replace with your actual file path
df_no_top_in = pd.read_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Results_Monkey/Segment_volumes_new_bound_no_top_MRI_reso_incre_bp.csv")  # Replace with your actual file path

fil_df_offset_de = df_offset_de[df_offset_de['distance'] > 0.0]
fil_df_no_top_de = df_no_top_de[df_no_top_de['distance'] > 0.0]
fil_df_all_de = df_all_de[df_all_de['distance'] > 0.0]

fil_df_offset_in = df_offset_in[df_offset_in['distance'] > 0.0]
fil_df_no_top_in = df_no_top_in[df_no_top_in['distance'] > 0.0]
fil_df_all_in = df_all_in[df_all_in['distance'] > 0.0]


# Set the style for a clean look without grid
plt.style.use('default')

# Create figure with appropriate size and spacing
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# Define a modern color palette
colors = {
    'no_top_de': '#0d6e8f',    # Modern blue
	'no_top_in': '#4db0cd',      # Modern blue
    'all_slices_de': '#bf3859',  # JAHA red
	'all_slices_in': '#f47a99',  # JAHA red
    'offset_de': "#05ab80",      # matching green that also colorblind people can differentiate
	'offset_in': "#3fe6bc",      # matching green that also colorblind people can differentiate
    'ground_truth': "#000000", # black
    'volume_mesh': "#868686"   # grey
}

############ W/O Top Slice #######################
#Decrease
line1_de = ax.plot(fil_df_no_top_de['distance'], fil_df_no_top_de['myo_vol'], 
                color=colors['no_top_de'], linewidth=2.5, 
                label="Without Top Slice - Underestimate", alpha=0.6, zorder=3)
scatter1_de = ax.scatter(fil_df_no_top_de['distance'], fil_df_no_top_de['myo_vol'], 
                     color=colors['no_top_de'], s=50, alpha=0.5, linewidth=1, zorder=4)
#Increase
line1_in = ax.plot(fil_df_no_top_in['distance'], fil_df_no_top_in['myo_vol'], 
                color=colors['no_top_in'], linewidth=2.5, 
                label="Without Top Slice - Overestimate", alpha=0.6, zorder=3)
scatter1_in = ax.scatter(fil_df_no_top_in['distance'], fil_df_no_top_in['myo_vol'], 
                     color=colors['no_top_in'], s=50, alpha=0.5, linewidth=1, zorder=4)

############ ALL Slices #######################
#Decrease
line2_de = ax.plot(fil_df_all_de['distance'], fil_df_all_de['myo_vol'], 
                color=colors['all_slices_de'], linewidth=2.5, 
                label="All Slices - Underestimate", alpha=0.6, zorder=3)
scatter2_de = ax.scatter(fil_df_all_de['distance'], fil_df_all_de['myo_vol'], 
                     color=colors['all_slices_de'], s=50, alpha=0.5, linewidth=1, zorder=4)
#Increase
line2_in = ax.plot(fil_df_all_in['distance'], fil_df_all_in['myo_vol'], 
                color=colors['all_slices_in'], linewidth=2.5, 
                label="All Slices - Overestimate", alpha=0.6, zorder=3)
scatter2_in = ax.scatter(fil_df_all_in['distance'], fil_df_all_in['myo_vol'], 
                     color=colors['all_slices_in'], s=50, alpha=0.5, linewidth=1, zorder=4)

############ W Offset #######################
#Decrease
line3_de = ax.plot(fil_df_offset_de['distance'], fil_df_offset_de['myo_vol'], 
                color=colors['offset_de'], linewidth=2.5, 
                label="W Offset - Underestimate", alpha=0.6, zorder=3)
scatter3_de = ax.scatter(fil_df_offset_de['distance'], fil_df_offset_de['myo_vol'], 
                     color=colors['offset_de'], s=50, alpha=0.5, linewidth=1, zorder=4)

#Increase
line3_in = ax.plot(fil_df_offset_in['distance'], fil_df_offset_in['myo_vol'], 
                color=colors['offset_in'], linewidth=2.5, 
                label="W Offset - Overestimate", alpha=0.6, zorder=3)
scatter3_in = ax.scatter(fil_df_offset_in['distance'], fil_df_offset_in['myo_vol'], 
                     color=colors['offset_in'], s=50, alpha=0.5, linewidth=1, zorder=4)


# Ground truth lines with improved styling
#Blood pool
# gt_line = ax.axhline(y=8.58, color=colors['ground_truth'], linewidth=2.5, 
#                      label='Ground Truth', alpha=0.9, zorder=2)
# vm_line = ax.axhline(y=8.65, color=colors['volume_mesh'], linewidth=2.5, 
#                      linestyle='--', label='Volume Meshes', alpha=0.9, zorder=2)

#Myocardium
gt_line = ax.axhline(y=8.18, color=colors['ground_truth'], linewidth=2.5, 
                     label='Ground Truth', alpha=0.9, zorder=2)
vm_line = ax.axhline(y=8.20, color=colors['volume_mesh'], linewidth=2.5, 
                     linestyle='--', label='Volume Meshes', alpha=0.9, zorder=2)

# Enhance the axes
ax.set_xscale('log')
ax.invert_xaxis()

# Beautiful labels and title with better sizing
ax.set_xlabel('Distance [mm]', fontsize=16, fontweight='bold', color='#000000')
ax.set_ylabel('Volume [ml]', fontsize=16, fontweight='bold', color='#000000')
ax.set_title('Effect of Segmentation Masks', fontsize=18, fontweight='bold', 
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
plt.savefig("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Graphics/plot_calculated_myo_vol_decrease_increase.pdf", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.2)
plt.savefig("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Graphics/plot_calculated_myo_vol_decrease_increase.svg", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.2)
# plt.show()