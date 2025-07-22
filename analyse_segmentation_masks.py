import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import sys, os
from matplotlib.colors import ListedColormap, BoundaryNorm

indices = ["0160","0140", "0120","0100", "0080", "0060", "0040", "0020"]


# Define RGBA colors for labels 0–3
colors = [
    (0, 0, 0, 0),     # 0 → transparent
    (0, 0, 0, 0),     # 1 → red
    (0, 1, 0, 1),     # 2 → green
    (0, 0, 1, 1)      # 3 → blue
]

# Create colormap and normalization
cmap = ListedColormap(colors)
norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=len(colors))


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


def create_ring_mask_with_hole( outer_boundary, inner_boundary, grid_size=None):
    """
    Creates a mask with different labels for various regions:
    - 0: Outside the outer boundary
    - 2: Between outer and inner boundaries (the ring region)
    - 3: Inside the inner boundary
    
    Parameters:
    -----------
    outer_x, outer_y : array-like
        X and Y coordinates of the outer boundary
    inner_x, inner_y : array-like
        X and Y coordinates of the inner boundary
    grid_size : tuple, optional
        Size of the output grid (height, width). If None, uses the actual coordinate range.
    plot_result : bool
        Whether to plot the result
    
    Returns:
    --------
    mask : numpy.ndarray
        Labeled mask with:
        - 0 outside the outer boundary
        - 2 in the ring region
        - 3 inside the inner boundary
    x, y : numpy.ndarray
        Coordinate arrays used for the mask
    """
    
    # Create Path objects from the boundaries
    outer_path = Path(outer_boundary)
    inner_path = Path(inner_boundary)

    outer_x = outer_boundary[:,0]
    outer_y = outer_boundary[:,1]
    inner_x = inner_boundary[:,0]
    inner_y = inner_boundary[:,1]
    
    # Determine grid size
    if grid_size is None:

        # Bounds from the boundaries
        x_min_data = 0
        x_max_data = 256
        y_min_data = 0
        y_max_data = 256

        # Compute data width/height
        x_range = x_max_data - x_min_data
        y_range = y_max_data - y_min_data

        # Resolution for 47 pixels in data range
        x_res = x_range / 256
        y_res = y_range / 256

        # Add 10 pixel margin on each side using same resolution
        x_margin = 0
        y_margin = 0

        # Final x and y coordinate arrays
        x = np.linspace(x_min_data - x_margin, x_max_data + x_margin, 256)
        y = np.linspace(y_min_data - y_margin, y_max_data + y_margin, 256)
    else:
        # If grid_size is provided, use it
        x = np.linspace(0, grid_size[1]-1, grid_size[1])
        y = np.linspace(0, grid_size[0]-1, grid_size[0])
    
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    mask = np.zeros_like(X, dtype=int)

    # Test which points are inside each path
    mask_outer = outer_path.contains_points(points)
    mask_inner = inner_path.contains_points(points)

    inner_mask = mask_inner.reshape(X.shape)
    outer_mask = mask_outer.reshape(X.shape)

    #label the blood pool
    mask[mask_inner.reshape(X.shape)] = 3
    
    # Label the ring region (between outer and inner boundaries) with 2
    mask[(mask_outer.reshape(X.shape)) & (~mask_inner.reshape(X.shape))] = 2
    
    return mask, x, y, outer_boundary, inner_boundary

def dice_score(y_true, y_pred, epsilon=0):
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    volume_sum = np.sum(y_true == 1) + np.sum(y_pred == 1)
    return (2. * intersection + epsilon) / (volume_sum + epsilon)

# Folder where masks are saved
input_dir = "/data.lfpn/ibraun/Code/paper_volume_calculation/Segmentation masks"

# Load all masks into a list
mri_images = [np.load(os.path.join(input_dir, f"data_t0_z{i}.npy")) for i in range(1,9)]
mri_images = mri_images[::-1]

prepped_mri_images = [np.load(os.path.join(input_dir, f"prepped_data_t00_z0{i}.npy")) for i in range(9)]

#fig, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 2 * 4))  # 2 rows, 8 columns

# for i, index in enumerate(indices):
#     #----- Top row: Generated mask -----
#     icontour_file = f"/data.lfpn/ibraun/Code/paper_volume_calculation/Patient_Segmentations/SCD_ManualContours/Healthy/SC-N-02/contours-manual/IRCCI-expert/IM-0001-{index}-icontour-manual.txt"
#     ocontour_file = f"/data.lfpn/ibraun/Code/paper_volume_calculation/Patient_Segmentations/SCD_ManualContours/Healthy/SC-N-02/contours-manual/IRCCI-expert/IM-0001-{index}-ocontour-manual.txt"

#     ix, iy = read_contour(icontour_file)
#     ox, oy = read_contour(ocontour_file)

#     inner_boundary = np.stack([ix, iy], axis=1)
#     outer_boundary = np.stack([ox, oy], axis=1)

#     mask_gen, x, y, outer_boundary, inner_boundary = create_ring_mask_with_hole(outer_boundary, inner_boundary)

#     # Plot MRI + contours
#     ax = axes[0,i]
#     ax.imshow(mri_images[i], cmap='gray')
#     ax.imshow(mask_gen, cmap=cmap, norm=norm, alpha = 0.5)
#     ax.plot(ix, iy, 'black', linewidth=1.5, label='Inner')
#     ax.plot(ox, oy, 'black', linewidth=1.5, label='Outer')
#     ax.axis('off')

#     mask_loaded = np.load(f"/data.lfpn/ibraun/Code/paper_volume_calculation/Segmentation masks/prepped_masks_t00_z0{i+1}.npy")
#     # Flip prepped MRI in both x and y before plotting
#     rotated = np.flipud(np.rot90(prepped_mri_images[i], k=1))
#     rotated_mask =  np.flipud(np.rot90(mask_loaded, k=1))

#     ax2 = axes[1, i]
#     ax2.imshow(rotated, cmap='gray')
#     ax2.imshow(rotated_mask, cmap=cmap, norm=norm, alpha = 0.5)
#     ax2.axis('off')

#     axes[0, len(indices)//2 - 1].set_title("Manual Segmentations", fontsize=14)
#     axes[1, len(indices)//2 - 1].set_title("Automatic Segmentation", fontsize=14)

fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 2 * 4))  # 2 rows, 8 columns

manual_masks_bp = []
automated_masks_bp = []

manual_masks_myo = []
automated_masks_myo = []

for i, index in enumerate(indices):
    #----- Top row: Generated mask -----
    icontour_file = f"/data.lfpn/ibraun/Code/paper_volume_calculation/Patient_Segmentations/SCD_ManualContours/Healthy/SC-N-02/contours-manual/IRCCI-expert/IM-0001-{index}-icontour-manual.txt"
    ocontour_file = f"/data.lfpn/ibraun/Code/paper_volume_calculation/Patient_Segmentations/SCD_ManualContours/Healthy/SC-N-02/contours-manual/IRCCI-expert/IM-0001-{index}-ocontour-manual.txt"

    ix, iy = read_contour(icontour_file)
    ox, oy = read_contour(ocontour_file)

    inner_boundary = np.stack([ix, iy], axis=1)
    outer_boundary = np.stack([ox, oy], axis=1)

    mask_gen, x, y, outer_boundary, inner_boundary = create_ring_mask_with_hole(outer_boundary, inner_boundary)

    manual_masks_bp.append((mask_gen == 3).astype(np.uint8))
    manual_masks_myo.append((mask_gen == 2).astype(np.uint8))

    mask_loaded = np.load(f"/data.lfpn/ibraun/Code/paper_volume_calculation/Segmentation masks/prepped_masks_t00_z0{i+1}.npy")
    rotated_mask =  np.flipud(np.rot90(mask_loaded, k=1))
    automated_masks_bp.append((rotated_mask == 3).astype(np.uint8))
    automated_masks_myo.append((rotated_mask == 2).astype(np.uint8))

automated_masks_bp = np.array(automated_masks_bp)
automated_masks_myo = np.array(automated_masks_myo)

#Calculate dice scores
dice_scores_bp = []
dice_scores_myo = []
for i in range(automated_masks_bp.shape[0]):
    bp_auto = automated_masks_bp[i]
    bp_manu = manual_masks_bp[i] #I assume the manual segmentation to be the true segmentation and the automatic segmentation as the prediction
    score_bp= dice_score(bp_manu, bp_auto)
    dice_scores_bp.append(round(score_bp, 3))

    myo_auto = automated_masks_myo[i]
    myo_manu = manual_masks_myo[i] #I assume the manual segmentation to be the true segmentation and the automatic segmentation as the prediction
    score_myo= dice_score(myo_manu, myo_auto)
    dice_scores_myo.append(round(score_myo, 3))

print(dice_scores_bp)
print(dice_scores_myo)

print(f"Dice Score Bloop Pool Mean: {np.mean(dice_scores_bp):.3f}")
print(f"Dice Score Bloop Pool Standard Deviation: {np.std(dice_scores_bp):.3f}")

print(f"Dice Score Myocardium Mean: {np.mean(dice_scores_myo):.3f}")
print(f"Dice Score Myocardium Standard Deviation: {np.std(dice_scores_myo):.3f}")

#     # Plot MRI + contours
#     ax = axes[i]
#     # ax.imshow(mask_gen, cmap=cmap, norm=norm, alpha = 0.5)
#     ax.imshow(rotated_mask, cmap=cmap, norm=norm, alpha = 0.5)
#     ax.plot(ix, iy, 'black', linewidth=1.5, label='Inner')
#     ax.plot(ox, oy, 'black', linewidth=1.5, label='Outer')
#     ax.axis('off')

#     axes[len(indices)//2 - 1].set_title("Compare Segmentations", fontsize=14)


# plt.tight_layout()
# plt.savefig("auto_manu_segmentation_masks_no_MRI.png")
# plt.show()





