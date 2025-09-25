import os
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

def seg_mask_plot_thickness_map(thickness_map,time, output_folder):
    """
    Plot the radial thickness map with a custom colormap and legend for invalid points.
    Assumes thickness_map is a 2D NumPy array with shape (n_slices, n_theta_bins).
    """

    # Create custom diverging colormap (blue -> light grey -> red)
    colors = ['#4B6FA5', '#D3D3D3', '#C10E21']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

    # Mask invalid data: NaNs or zero thickness (adjust condition if needed)
    invalid_mask = np.isnan(thickness_map) | (thickness_map == 0)
    masked_thickness = np.ma.masked_where(invalid_mask, thickness_map)

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

    plt.title(f"Radial Thickness Map - Time Step {time}", fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()

    # Add black border around plot
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"thickness_map_{time}.pdf"),dpi=300)
    plt.close()


def translate_mesh_to_origin(mesh, points, threshold=0):
    """
    Translate the mesh so that the centroid of points with z >= threshold is at the origin.
    """
    # Select points above the given z-threshold (e.g., basal region of LV)
    basal_points = points[points[:, 2] >= threshold]

    # Compute centroid of selected points
    centroid = np.mean(basal_points, axis=0)

    # Translate mesh so that this centroid is placed at the origin
    mesh.translate(-centroid, inplace=True)

    return centroid


def cartesian_to_cylindrical(points):
    """
    Convert cartesian coordinates to cylindrical (r, theta, z).
    Theta wrapped to [0, 2pi].
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Radial distance
    r = np.sqrt(x ** 2 + y ** 2)

    # Azimuthal angle
    theta = np.arctan2(y, x)

    # Wrap theta into [0, 2π]
    theta = (theta + 2 * np.pi) % (2 * np.pi)

    return r, theta, z


def find_ring_slices(points, n_z_bins=15):
    """
    Find z-bin slices that have a ring shape based on a heuristic:
    inner radius > 0.2 * outer radius.
    Returns a list of tuples (slice_index, z_low, z_high).
    """
    z = points[:, 2]

    # Define bin edges along z-axis
    z_bins = np.linspace(z.min(), z.max(), n_z_bins + 1)
    ring_slices = []

    for i in range(n_z_bins):
        z_low, z_high = z_bins[i], z_bins[i + 1]

        # Extract points within the current z-slice
        mask = (z >= z_low) & (z < z_high)
        slice_points = points[mask]

        if slice_points.size == 0:
            continue

        # Compute radial distances in slice
        xy = slice_points[:, :2]
        r_slice = np.linalg.norm(xy, axis=1)

        r_min, r_max = r_slice.min(), r_slice.max()

        # Heuristic: consider slice a "ring" if inner radius is not too small
        if r_min > 0.2 * r_max:
            ring_slices.append((i, z_low, z_high))

    return ring_slices, z_bins


def compute_thickness_map(r, theta, z, ring_slices, z_bins, n_theta_bins=36):
    """
    Compute the radial thickness map for filtered z-bins and theta bins.
    """
    # Define angular bins
    theta_bins = np.linspace(0, 2 * np.pi, n_theta_bins + 1)

    # Extract z-bin indices for ring slices
    filtered_zbins = np.array(ring_slices)[:, 0]
    n_filtered_zbins = len(filtered_zbins)

    # Digitize coordinates into bin indices
    theta_idx = np.digitize(theta, theta_bins) - 1
    z_idx = np.digitize(z, z_bins) - 1

    # Initialize thickness map (NaN means no data)
    thickness_map = np.full((n_filtered_zbins, n_theta_bins), np.nan)

    for i, zi in enumerate(filtered_zbins):
        zi = int(zi)
        for ti in range(n_theta_bins):
            # Select points belonging to current (z, theta) bin
            mask = (z_idx == zi) & (theta_idx == ti)
            if np.any(mask):
                r_values = r[mask]
                # Thickness = outer radius - inner radius
                thickness_map[i, ti] = r_values.max() - r_values.min()

    # Compute z-coordinates corresponding to slice centers
    z_bin_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    filtered_z_coords = z_bin_centers[filtered_zbins.astype(int)]

    # Normalize z to start from zero
    z_shift = z_bins.min()
    filtered_z_coords = filtered_z_coords - z_shift

    return thickness_map, filtered_z_coords


def meshes_plot_thickness_map(thickness_map, filtered_z_coords, time, output_folder):
    """
    Plot the radial thickness map with a custom colormap and legend for invalid points.
    """
    # Compute mean and std for filtering outliers
    mean_thick = np.nanmean(thickness_map)
    std_thick = np.nanstd(thickness_map)

    # Mask invalid or outlier values
    invalid_mask = (thickness_map < mean_thick - 5 * std_thick) | np.isnan(thickness_map)
    valid_thickness = np.ma.masked_where(invalid_mask, thickness_map)

    # Custom colormap: blue (thin) → grey (normal) → red (thick)
    colors = ['#4B6FA5', '#D3D3D3', '#C10E21']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    cmap.set_bad(color='black', alpha=0.8)

    plt.figure(figsize=(12, 8))

    # Show map: x=angle, y=z-position, color=thickness
    im = plt.imshow(valid_thickness, aspect='auto', cmap=cmap,
                    extent=[0, 360, filtered_z_coords[-1], filtered_z_coords[0]])

    # Axis labels
    plt.xlabel("Azimuthal Angle", fontsize=12, fontweight='bold')
    plt.ylabel("Z Height (mm)", fontsize=12, fontweight='bold')

    # Format ticks
    ticks = np.arange(0, 370, 40)
    plt.xticks(ticks, fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val)}°"))

    # Colorbar with label
    cbar = plt.colorbar(im, label="Radial Thickness (mm)", shrink=0.8)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Radial Thickness (mm)", fontsize=12, fontweight='bold')

    # Legend for invalid values
    if np.any(invalid_mask):
        invalid_patch = mpatches.Patch(color='black', label='Invalid (Small or NaN)')
        plt.legend(handles=[invalid_patch], loc='upper right', fontsize=12, framealpha=0.8)

    # Title and styling
    plt.title("Radial Thickness Map", fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()  # Flip so apex is at bottom

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    plt.tight_layout()

    # Save figure as PDF
    plt.savefig(os.path.join(output_folder, f"thickness_map_{time}.pdf"), dpi=300)
    plt.close()
