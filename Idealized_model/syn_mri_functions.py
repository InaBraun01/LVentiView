import sys
import numpy as np
import vtk
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import binary_dilation

#in this folder I save all of the functions, which I need to generate the synthetic MRI images. 
np.random.seed(42)

def generate_ellipsoidal_mask(height, width, n_regions=5, region_size=20, value_range=(20, 160)):
    """
    Creates background  for synthetic MRI images. Creates background out of ellipsoidal of varying size of the 
    long and short axis of the ellipse. Pixels in each ellipses have solid color values + noise. 
    ---------------------------------------------------------------------------------------------
    height: height of background
    width: width of background
    n_regions: number of ellipsoidal regions 
    region_size: Defines range of size of ellipses
    value_range: diffences range of values/colors of ellipses
    ----------------------------------------------------------------------------------------------
    Return: numpy array of backgroun
    """
    mask = np.zeros((height, width), dtype=int)

    for _ in range(n_regions):
        # Random value between value_range (e.g., 0 to 140)
        value = np.random.randint(*value_range)

        # Random starting point for the center of the ellipse
        # Ensure the region fits inside the mask
        x_center = np.random.randint(0, width)
        y_center = np.random.randint(0, height)

        # Semi-major and semi-minor axes for the ellipse
        a = region_size + np.random.randint(-region_size, region_size) # Semi-major axis (horizontal radius)
        b = region_size + np.random.randint(-region_size, region_size) # Semi-minor axis (vertical radius)

        # Generate the elliptical region
        for y in range(y_center - b, y_center + b):
            for x in range(x_center - a, x_center + a):

                color_jitter = np.random.uniform(-10,10)
                # Check if the point (x, y) lies inside the ellipse
                if (x - x_center)**2 / a**2 + (y - y_center)**2 / b**2 <= 1:
                    if 0 <= x < width and 0 <= y < height:
                        mask[y, x] = value + color_jitter

    return mask

def load_vtu_mesh(filename):
    """
    Loads vtu file containing mesh
    -----------------------------
    filename: file name of vtu file
    ----------------------------------
    Retrun: mesh
    """
    #reader = vtk.vtkXMLUnstructuredGridReader()
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def extract_slice(mesh, z_value):
    """
    Slices mesh parallel to x and y plane
    ----------------------------------------
    mesh: mesh object to be slices
    z_value: z value at which to slice the mesh
    ------------------------------------------
    Return: slice of mesh
    """
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, z_value)
    plane.SetNormal(0, 0, 1)  # Z-axis

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(mesh)
    cutter.Update()
    return cutter.GetOutput()

def polydata_to_numpy(polydata):
    """
    Extract points from polydata object
    ------------------------------------
    polydata: polydata object
    -------------------------------------
    numpy array of points
    """
    points = polydata.GetPoints()
    if points is None:
        return np.empty((0, 3))
    return np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])

def find_ring_boundaries(points_3d, delta=0.001):
    """
    Identifies inner and outer boundary points of a 2D ring.
    
    Args:
        points_3d (Nx3 array): 3D point cloud.
        delta (float): Thickness tolerance for detecting boundaries.

    Returns:
        inner_boundary (array of points), outer_boundary (array of points), center
    """
    # Project points to 2D
    points_2d = points_3d[:, :2]
    
    # Compute centroid
    center = np.mean(points_2d, axis=0)
    
    # Compute distances to center
    distances = np.linalg.norm(points_2d - center, axis=1)
    
    # Identify inner and outer boundary points
    r_min, r_max = distances.min(), distances.max()

    inner_mask = np.abs(distances - r_min) < delta
    outer_mask = np.abs(distances - r_max) < delta
    
    inner_boundary = points_3d[inner_mask]
    outer_boundary = points_3d[outer_mask]

    return inner_boundary, outer_boundary, center

def find_ring_boundaries_polar(points_3d, num_segments=36):
    """
    Identifies inner and outer boundary points of a 2D ring using polar coordinates.
    
    Args:
        points_3d (Nx3 array): 3D point cloud.
        num_segments (int): Number of angular segments to divide the ring into.
    
    Returns:
        inner_boundary (array of points), outer_boundary (array of points), center
    """
    import numpy as np
    
    # Project points to 2D
    points_2d = points_3d[:, :2]
    
    # Compute centroid
    center = np.mean(points_2d, axis=0)
    
    # Convert to polar coordinates
    points_centered = points_2d - center
    r = np.linalg.norm(points_centered, axis=1)
    theta = np.arctan2(points_centered[:, 1], points_centered[:, 0])
    
    # Divide into angular bins
    bin_size = 2 * np.pi / num_segments
    inner_boundary_indices = []
    outer_boundary_indices = []
    
    for i in range(num_segments):
        bin_start = -np.pi + i * bin_size
        bin_end = bin_start + bin_size
        
        # Find points in this angular bin
        bin_mask = (theta >= bin_start) & (theta < bin_end)
        if not np.any(bin_mask):
            continue
            
        # Find min and max radius in this bin
        bin_points = np.where(bin_mask)[0]
        bin_r = r[bin_mask]
        
        if len(bin_r) > 0:
            inner_idx = bin_points[np.argmin(bin_r)] #points with min r are part of the inner boundary
            outer_idx = bin_points[np.argmax(bin_r)] #points with max r are part of the outer boundary
            
            inner_boundary_indices.append(inner_idx)
            outer_boundary_indices.append(outer_idx)
    
    inner_boundary = points_3d[inner_boundary_indices]
    outer_boundary = points_3d[outer_boundary_indices]
    
    return inner_boundary, outer_boundary, center

def order_points_clockwise(points):
    """
    Order points clockwise, so that in the order of the points the points form a circle
    ----------------------------------------------------------------------------------
    points: numpy array of points
    ----------------------------------------------------------------------------------
    return: numpy array of the sorted points
    """
    # Compute center
    center = np.mean(points, axis=0)

    # Compute angles
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    # Sort points by angle
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]

def create_ring_mask_with_hole(background, outer_boundary, inner_boundary, i, grid_size=None, vol_calculation = False):
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
        x_min_data = -0.016529157519445517
        x_max_data = 0.023470432133575435
        y_min_data = -0.015502749694884922
        y_max_data = 0.024496838965704114

        # Compute data width/height
        x_range = x_max_data - x_min_data
        y_range = y_max_data - y_min_data

        # Resolution for 47 pixels in data range
        x_res = x_range / 47
        y_res = y_range / 47

        # Add 10 pixel margin on each side using same resolution
        x_margin = 60 * x_res
        y_margin = 60 * y_res

        # Final x and y coordinate arrays
        x = np.linspace(x_min_data - x_margin, x_max_data + x_margin, 167)
        y = np.linspace(y_min_data - y_margin, y_max_data + y_margin, 167)
    else:
        # If grid_size is provided, use it
        x = np.linspace(0, grid_size[1]-1, grid_size[1])
        y = np.linspace(0, grid_size[0]-1, grid_size[0])
    
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Create the mask with different labels
    mask_irreg = background
    mask = np.zeros_like(X, dtype=int)

    # Test which points are inside each path
    mask_outer = outer_path.contains_points(points)
    mask_inner = inner_path.contains_points(points)

    inner_mask = mask_inner.reshape(X.shape)
    outer_mask = mask_outer.reshape(X.shape)


    if not vol_calculation:
        #label the blood pool
        random_vals = np.random.uniform(140, 160, size=inner_mask.shape)
        mask_irreg[inner_mask] = random_vals[inner_mask]

        #label the myocardium
        ring_mask = (outer_mask) & (~inner_mask)

        # Generate random values (either int or float) of the same shape
        random_vals_ring = np.random.uniform(50, 100, size=ring_mask.shape)

        # Assign values only where the mask is True
        mask_irreg[ring_mask] = random_vals_ring[ring_mask]

        return mask_irreg, x, y, outer_boundary, inner_boundary

    else:
        #label the blood pool
        mask[mask_inner.reshape(X.shape)] = 3
        
        # Label the ring region (between outer and inner boundaries) with 2
        mask[(mask_outer.reshape(X.shape)) & (~mask_inner.reshape(X.shape))] = 2

        # fig, ax = plt.subplots(figsize=(6, 6))
    
        # extent = [x.min(), x.max(), y.min(), y.max()]
        # im = ax.imshow(mask, cmap='gray', origin='lower', extent=extent)
        # cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cbar.set_label("Mask Label")

        # # Plot boundaries
        # ax.plot(outer_boundary[:, 0], outer_boundary[:, 1], 'r-', label='Outer boundary')
        # ax.plot(inner_boundary[:, 0], inner_boundary[:, 1], 'b-', label='Inner boundary')

        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_aspect('equal')
        # ax.legend()
        # plt.tight_layout()
        # plt.show()
        # sys.exit()
    
        return mask, x, y, outer_boundary, inner_boundary

def calculate_vol(mask, i, distance, offset_base, offset_apex, num_slices):
    """
    Volume calculation from segmentation masks. Not taking distance between z planes account in the top layer
    and taking offsets into account
    -----------------------------------------------------
    """
    x_reso = 0.8523
    y_reso = 0.8523
    # x_reso = 0.4
    # y_reso = 0.4

    pixel_myo = np.sum(mask == 2)
    pixel_bp = np.sum(mask == 3)

    #for base slice take only the offest as the z height
    if i == (num_slices-1):
        #z_reso_base = offset_base
        #z_reso_base = 0.0
        z_reso_base = distance
        vol_myo = z_reso_base*x_reso*y_reso*pixel_myo
        vol_bp = z_reso_base*x_reso*y_reso*pixel_bp

    #for apex take offset plus thickness as the z height
    elif i == 0:
        #z_reso_apex = distance + offset_apex
        z_reso_apex = distance 
        vol_myo = z_reso_apex*x_reso*y_reso*pixel_myo
        vol_bp = z_reso_apex*x_reso*y_reso*pixel_bp

    else:
        vol_myo = distance*x_reso*y_reso*pixel_myo
        vol_bp = distance*x_reso*y_reso*pixel_bp

    return vol_myo, vol_bp

def jitter_mask(mask, jitter_strength=0.05, max_jitter=0.01):
    """
    Adds jitter to the position of True values in a binary mask.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        A binary mask with True and False values (or 1 and 0 values).
    jitter_strength : float, optional (default=0.05)
        The fraction of `True` pixels to be jittered (e.g., 0.05 means 5% of the `True` pixels will be moved).
    max_jitter : int, optional (default=1)
        The maximum distance (in pixels) by which the `True` values can be shifted.

    Returns:
    --------
    jittered_mask : numpy.ndarray
        The mask with jitter added to the positions of the True values.
    """
    # Get the coordinates of the True values
    true_indices = np.argwhere(mask)
    
    # Number of True values to jitter
    n_jitter = int(jitter_strength * len(true_indices))
    
    # Randomly select which True values will be jittered
    jitter_indices = np.random.choice(len(true_indices), size=n_jitter, replace=False)
    
    # Apply jitter to the selected indices
    for idx in jitter_indices:
        # Get the current position of the True value
        y, x = true_indices[idx]
        
        # Random jitter offset (within [-max_jitter, max_jitter])
        jitter_y = np.random.uniform(-max_jitter, max_jitter)
        jitter_x = np.random.uniform(-max_jitter, max_jitter)
        
        # New position after jitter
        new_y = int(np.clip(y + jitter_y, 0, mask.shape[0] - 1))
        new_x = int(np.clip(x + jitter_x, 0, mask.shape[1] - 1))

        if new_y != y:
            print(max_jitter)
            print(jitter_y)
            print(y + jitter_y)
            print(y)

        
        # Set the jittered position to True in the mask
        mask[new_y, new_x] = True
        
        # Set the original position to False
        mask[y, x] = False

    return mask

def modify_seg_mask_increa_bp(mask):

    # Define labels
    inside_label = 3       # Label for inside of the bowl (the sphere)
    myocardium_label = 2   # Label for the ring (the bowl)

    # Create binary mask of the inside of the bowl
    inside_mask = mask == inside_label

    # Dilate the inside mask to get its border
    dilated_inside = binary_dilation(inside_mask)

    # The outer ring of the inside (i.e., the boundary) is where dilation added new pixels
    outer_ring_of_inside = dilated_inside & (~inside_mask)

    # Add these pixels to the myocardium (ring)
    mask[outer_ring_of_inside] = inside_label

    return mask

def modify_seg_mask_decrea_bp(mask):

    # Step 1: Create boolean masks
    inside_mask = (mask == 3)
    outside_mask = (mask == 2)

    # Step 2: Dilate the inside mask by 1 pixel
    dilated_inside = binary_dilation(outside_mask)

    # Step 3: Find the 1-pixel thick ring around inside_mask
    inner_boundary_ring = dilated_inside & ~outside_mask

    # Step 4: Keep only pixels in the ring that also touch the myocardium
    touching_myocardium = inner_boundary_ring & inside_mask

    # Step 5: Add those to the inside mask (or assign to a new label)
    mask[touching_myocardium] = 2  # or another label if you prefer

    return mask

def plot_modi_masks(mask):
    # Define custom colormap for values 0, 2, 3
    cmap = ListedColormap([
        "#000000",  # 0 → black
        "#00CC66",  # 2 → vibrant green
        "#007FFF"   # 3 → vibrant blue
    ])

    # Define boundaries and normalization to match values to colors
    bounds = [-0.5, 1, 2.5, 3.5]  # So 0→[−0.5,1), 2→[1,2.5), 3→[2.5,3.5)
    norm = BoundaryNorm(bounds, cmap.N)

    # # Assume modi_mask is a 2D NumPy array
    # h, w = mask.shape

    # # Compute center coordinates
    # center_y = h // 2
    # center_x = w // 2

    # # Half-size of crop
    # half_crop = 40  # since 80 // 2 = 40

    # # Crop the region: [start_y:end_y, start_x:end_x]
    # cropped = mask[
    #     center_y - half_crop : center_y + half_crop,
    #     center_x - half_crop : center_x + half_crop
    # ]
    # mask = cropped

    # Plot image
    plt.imshow(mask, cmap=cmap, norm=norm)
    cbar = plt.colorbar(ticks=[0, 2, 3])
    cbar.ax.set_yticklabels(['0', '2', '3'])  # Ensure correct labels
    plt.title("Custom Color Mapping")
    plt.axis('off')
    # plt.savefig("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Segmentation_masks/ground_truth_blood_pool.pdf")
    plt.show()

    return


def create_mask_for_DICO(outer_boundary, inner_boundary, i, grid_size=None):
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
        x_min_data = -0.016529157519445517
        x_max_data = 0.023470432133575435
        y_min_data = -0.015502749694884922
        y_max_data = 0.024496838965704114

        # Compute data width/height
        x_range = x_max_data - x_min_data
        y_range = y_max_data - y_min_data

        # Resolution for 47 pixels in data range
        x_res = x_range / 47
        y_res = y_range / 47

        # Add 10 pixel margin on each side using same resolution
        x_margin = 10 * x_res
        y_margin = 10 * y_res

        # Final x and y coordinate arrays
        x = np.linspace(x_min_data - x_margin, x_max_data + x_margin, 67)
        y = np.linspace(y_min_data - y_margin, y_max_data + y_margin, 67)
    else:
        # If grid_size is provided, use it
        x = np.linspace(0, grid_size[1]-1, grid_size[1])
        y = np.linspace(0, grid_size[0]-1, grid_size[0])
    
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Create the mask with different labels
    mask_irreg = np.loadtxt("irreg_mask.csv", delimiter=',')
    #mask = np.zeros_like(X, dtype=int)

    # Test which points are inside each path
    mask_outer = outer_path.contains_points(points)
    mask_inner = inner_path.contains_points(points)

    inner_mask = mask_inner.reshape(X.shape)
    outer_mask = mask_outer.reshape(X.shape)
    
    #Jitter the points in the masks a bit
    jit_mask_outer = jitter_mask(outer_mask)
    jit_mask_inner = jitter_mask(inner_mask)

    outer_mask = jit_mask_outer
    inner_mask = jit_mask_inner

    #label the blood pool
    random_vals = np.random.uniform(140, 160, size=inner_mask.shape)
    mask_irreg[inner_mask] = random_vals[inner_mask]

    #label the myocardium
    ring_mask = (outer_mask) & (~inner_mask)

    # Generate random values (either int or float) of the same shape
    random_vals_ring = np.random.uniform(50, 100, size=ring_mask.shape)

    # Assign values only where the mask is True
    mask_irreg[ring_mask] = random_vals_ring[ring_mask]

    # #label the blood pool
    # mask[mask_inner.reshape(X.shape)] = 3
    
    # # Label the ring region (between outer and inner boundaries) with 2
    # mask[(mask_outer.reshape(X.shape)) & (~mask_inner.reshape(X.shape))] = 2
    
    return mask_irreg, x, y, outer_boundary, inner_boundary
