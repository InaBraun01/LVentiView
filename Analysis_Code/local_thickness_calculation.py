import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker


def translate_mesh_to_origin(mesh, points, threshold=0):
    """
    Translate the mesh so that the centroid of points with z >= threshold is at the origin.
    """
    basal_points = points[points[:, 2] >= threshold]
    centroid = np.mean(basal_points, axis=0)
    mesh.translate(-centroid, inplace=True)
    return centroid


def cartesian_to_cylindrical(points):
    """
    Convert cartesian coordinates to cylindrical (r, theta, z).
    Theta wrapped to [0, 2pi].
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    theta = (theta + 2 * np.pi) % (2 * np.pi)
    return r, theta, z


def find_ring_slices(points, n_z_bins=15):
    """
    Find z-bin slices that have a ring shape based on a heuristic:
    inner radius > 0.2 * outer radius.
    Returns a list of tuples (slice_index, z_low, z_high).
    """
    z = points[:, 2]
    z_bins = np.linspace(z.min(), z.max(), n_z_bins + 1)
    ring_slices = []

    for i in range(n_z_bins):
        z_low, z_high = z_bins[i], z_bins[i + 1]
        mask = (z >= z_low) & (z < z_high)
        slice_points = points[mask]

        if slice_points.size == 0:
            continue

        xy = slice_points[:, :2]
        r_slice = np.linalg.norm(xy, axis=1)
        r_min, r_max = r_slice.min(), r_slice.max()

        if r_min > 0.2 * r_max:
            ring_slices.append((i, z_low, z_high))

    return ring_slices, z_bins


def plot_ring_slices(points, ring_slices):
    """
    Plot ring-shaped slices with scatter plots.
    """
    n_rows = int(np.ceil(len(ring_slices) / 5))
    fig, axes = plt.subplots(n_rows, 5, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    for idx, (zi, z_low, z_high) in enumerate(ring_slices):
        ax = axes[idx]
        mask = (points[:, 2] >= z_low) & (points[:, 2] < z_high)
        slice_points = points[mask]
        ax.scatter(slice_points[:, 0], slice_points[:, 1], s=1)
        ax.set_title(f"Z: {z_low:.2f}–{z_high:.2f}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for ax in axes[len(ring_slices):]:
        ax.axis('off')

    plt.suptitle("Ring-Shaped Slices of Mesh", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("sliced_mesh.png")
    plt.close(fig)


def compute_thickness_map(r, theta, z, ring_slices, z_bins, n_theta_bins=36):
    """
    Compute the radial thickness map for filtered z-bins and theta bins.
    """
    theta_bins = np.linspace(0, 2 * np.pi, n_theta_bins + 1)
    filtered_zbins = np.array(ring_slices)[:, 0]
    n_filtered_zbins = len(filtered_zbins)

    theta_idx = np.digitize(theta, theta_bins) - 1
    z_idx = np.digitize(z, z_bins) - 1

    thickness_map = np.full((n_filtered_zbins, n_theta_bins), np.nan)

    for i, zi in enumerate(filtered_zbins):
        zi = int(zi)
        for ti in range(n_theta_bins):
            mask = (z_idx == zi) & (theta_idx == ti)
            if np.any(mask):
                r_values = r[mask]
                thickness_map[i, ti] = r_values.max() - r_values.min()

    # Calculate filtered z bin centers
    z_bin_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    filtered_z_coords = z_bin_centers[filtered_zbins.astype(int)]
    z_shift = z_bins.min()
    filtered_z_coords = filtered_z_coords - z_shift

    return thickness_map, filtered_z_coords


def plot_thickness_map(thickness_map, filtered_z_coords):
    """
    Plot the radial thickness map with a custom colormap and legend for invalid points.
    """
    mean_thick = np.nanmean(thickness_map)
    std_thick = np.nanstd(thickness_map)

    invalid_mask = (thickness_map < mean_thick - 5 * std_thick) | np.isnan(thickness_map)
    valid_thickness = np.ma.masked_where(invalid_mask, thickness_map)

    colors = ['#4B6FA5', '#D3D3D3', '#C10E21']  # Blue, Light Grey, Red
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    cmap.set_bad(color='black', alpha=0.8)

    mean_thickness = np.mean(valid_thickness.compressed())

    plt.figure(figsize=(12, 8))
    im = plt.imshow(valid_thickness, aspect='auto', cmap=cmap,
                    extent=[0, 360, filtered_z_coords[-1], filtered_z_coords[0]])

    plt.xlabel("Azimuthal Angle", fontsize=12, fontweight='bold')
    plt.ylabel("Z Height (mm)", fontsize=12, fontweight='bold')

    ticks = np.arange(0, 370, 40)
    plt.xticks(ticks, fontsize=11)
    plt.yticks(fontsize=11)

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val)}°"))

    cbar = plt.colorbar(im, label="Radial Thickness (mm)", shrink=0.8)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Radial Thickness (mm)", fontsize=12, fontweight='bold')

    if np.any(invalid_mask):
        invalid_patch = mpatches.Patch(color='black', label='Invalid (Small or NaN)')
        plt.legend(handles=[invalid_patch], loc='upper right', fontsize=12, framealpha=0.8)

    plt.title("Radial Thickness Map", fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig("radial_thickness_human.png")
    plt.show()

    # Print summary stats
    print("Thickness Statistics (excluding invalid data):")
    print(f"Mean: {mean_thickness:.2f} mm")
    print(f"Min: {np.min(valid_thickness.compressed()):.2f} mm")
    print(f"Max: {np.max(valid_thickness.compressed()):.2f} mm")
    print(f"Std Dev: {np.std(valid_thickness.compressed()):.2f} mm")
    print(f"Invalid points: {np.sum(invalid_mask)} of {thickness_map.size}")


if __name__ == "__main__":
    # mesh = pv.read("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/idealized_bowl.vtu")  
    mesh = pv.read("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/ES_human.vtk")
    points = mesh.points

    # Translate mesh to origin based on basal points threshold
    translate_mesh_to_origin(mesh, points, threshold=0)

    # Convert points to cylindrical coordinates
    r, theta, z = cartesian_to_cylindrical(points)

    # Find ring-shaped z slices
    ring_slices, z_bins = find_ring_slices(points, n_z_bins=15)

    # Plot the ring-shaped slices
    plot_ring_slices(points, ring_slices)

    # Compute thickness map and filtered z coords
    thickness_map, filtered_z_coords = compute_thickness_map(r, theta, z, ring_slices, z_bins)

    # Plot the thickness map
    plot_thickness_map(thickness_map, filtered_z_coords)
