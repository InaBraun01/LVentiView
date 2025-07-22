import sys 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker


def compute_thickness_map(seg_stack, n_theta_bins=36, label_of_interest=2):
    Z, H, W = seg_stack.shape
    print(Z)
    
    thickness_map = np.full((Z, n_theta_bins), np.nan)
    
    # Create azimuthal bins [0, 2pi]
    theta_bins = np.linspace(0, 2 * np.pi, n_theta_bins + 1)

    #calculate center from basal plane
    seg_slice = seg_stack[8]
    
    # Extract points of label_of_interest
    ys, xs = np.where(seg_slice == label_of_interest)

    xs = xs* 1.40625 #scale according to MRI resolution
    ys = ys* 1.40625 #scale according to MRI resolution
    
    # Compute centroid (mean of points)
    cx, cy = xs.mean(), ys.mean()

    
    for z in range(Z):
        seg_slice = seg_stack[z]
        
        # Extract points of label_of_interest
        ys, xs = np.where(seg_slice == label_of_interest)

        xs = xs* 1.40625 #scale according to MRI resolution
        ys = ys* 1.40625 #scale according to MRI resolution

        if len(xs) == 0:
            continue  # no object in this slice
        
        # Convert points to polar coordinates relative to centroid
        x_shifted = xs - cx
        y_shifted = ys - cy
        
        r = np.sqrt(x_shifted**2 + y_shifted**2)
        theta = np.arctan2(y_shifted, x_shifted)  # [-pi, pi]
        
        # Wrap theta to [0, 2pi]
        theta = (theta + 2 * np.pi) % (2 * np.pi)
        
        # Digitize theta into bins
        theta_idx = np.digitize(theta, theta_bins) - 1  # zero based
        
        for ti in range(n_theta_bins):
            mask = (theta_idx == ti)
            if not np.any(mask):
                continue
            
            r_values = r[mask]
            thickness = r_values.max() - r_values.min()
            thickness_map[z, ti] = thickness   #10mm is the distance between slices in z direction
    
    return thickness_map

def plot_thickness_map(thickness_map):
    """
    Plot the radial thickness map with a custom colormap and legend for invalid points.
    Assumes thickness_map is a 2D NumPy array with shape (n_slices, n_theta_bins).
    """

    # Create custom colormap (blue -> light grey -> red)
    colors = ['#4B6FA5', '#D3D3D3', '#C10E21']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

    # Mask invalid data: NaNs or zero thickness (adjust condition if needed)
    invalid_mask = np.isnan(thickness_map) | (thickness_map == 0)
    masked_thickness = np.ma.masked_where(invalid_mask, thickness_map)

    mean_thickness = masked_thickness.mean()

    plt.figure(figsize=(12, 8))

    # Plot masked thickness map
    im = plt.imshow(masked_thickness, aspect='auto', cmap=cmap)

    plt.xlabel("Azimuthal Angle ", fontsize=12, fontweight='bold')
    plt.ylabel("Z height (mm)", fontsize=12, fontweight='bold')

    # Set x-ticks to degrees assuming bins cover 0 to 360°
    n_theta_bins = thickness_map.shape[1]
    tick_angles = np.linspace(0, 360, n_theta_bins + 1)
    tick_locs = np.linspace(-0.5, n_theta_bins - 0.5, n_theta_bins + 1)  # align ticks with bin edges
    plt.xticks(tick_locs, [f"{int(angle)}°" for angle in tick_angles], rotation=45, fontsize=11)
    
    # Original y-ticks (indices)
    yticks = plt.yticks()[0]  # get current y tick locations (positions)

    # Create new labels by scaling tick values by 10
    new_labels = [f"{int(tick * 10)}" for tick in yticks[1:-1]]

    plt.yticks(yticks[1:-1], new_labels, fontsize=11)

    # Colorbar
    cbar = plt.colorbar(im, label="Radial Thickness (mm)", shrink=0.8)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Radial Thickness (mm)", fontsize=12, fontweight='bold')

    plt.title("Radial Thickness Map", fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()

    # Add black border around plot
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig("radial_thickness_human.png")
    plt.show()

    # Print summary stats excluding invalid data
    print("Thickness Statistics (excluding invalid data):")
    print(f"Mean: {mean_thickness:.2f}")
    print(f"Min: {masked_thickness.min():.2f}")
    print(f"Max: {masked_thickness.max():.2f}")
    print(f"Std Dev: {masked_thickness.std():.2f}")


if __name__ == "__main__":

    seg_masks = []
    crop_size = 90

    for i in range(9):
        seg_mask = np.load(f"/data.lfpn/ibraun/Code/paper_volume_calculation/Segmentation masks/prepped_masks_t00_z0{i}.npy")
        h, w = seg_mask.shape[:2]  # height, width
        start_x = w // 2 - crop_size // 2
        start_y = h // 2 - crop_size // 2
        seg_mask = seg_mask[start_y:start_y+crop_size, start_x:start_x+crop_size]
        seg_masks.append(seg_mask)

    seg_masks = np.array(seg_masks)

    # fig, axes = plt.subplots(1, 9, figsize=(18, 2))  # wide figure, 9 columns

    # for i, ax in enumerate(axes):
    #     ax.imshow(seg_masks[i], cmap='gray')
    #     ax.axis('off')  # hide axis ticks and labels
    #     ax.set_title(f"Image {i+1}")

    # plt.tight_layout()
    # plt.savefig("slices.png")
    # plt.show()
    # sys.exit()

    thickness_map = compute_thickness_map(seg_masks, n_theta_bins=36, label_of_interest=2)
    plot_thickness_map(thickness_map)

