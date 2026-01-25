"""
Cardiac Mesh Analysis and Fitting Pipeline

This module provides functionality for loading, manipulating, and fitting 3D cardiac mesh models
to medical imaging data. It includes tools for:
- Loading statistical shape models with PCA modes
- Mesh voxelization and rendering
- Neural network-based mesh fitting
- Evaluation and visualization of results

Dependencies: PyVista, PyTorch, SciPy, NumPy, MeshIO
"""

import os
import sys
from copy import deepcopy
import csv
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import meshio
import pickle
import time
import random
from scipy.ndimage.morphology import binary_fill_holes as bfh

# Set device for PyTorch computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_ShapeModel(num_modes, sz, cp_frequency, model_dir):
    """
    Load a statistical shape model with PCA modes for cardiac mesh analysis.

    Args:
        num_modes (int): Number of PCA modes to load
        sz (int): Size parameter for mesh scaling (size of cropped image)
        cp_frequency (int): Control point sampling frequency
        model_dir (str): Directory containing the shape model files
        
    Returns:
        tuple: Contains mesh, control points, projection matrices, mode bounds,
                mean modes, mesh offset, and mesh axes
                
    Note:
        - Uses half-scaled model for sz < 96 (typically for monkey studies)
        - Uses full-scaled model for sz >= 96 (typically for human studies)
    """
    # Load the mean/average mesh model
    if sz < 50:
        # Use half-scaled model for smaller left ventricles (e.g., monkey studies)
        mean_mesh_file = os.path.join(model_dir, 'Mean/LV_mean_half_scaled.vtk')
    else:
        # Use full-scaled model for larger left ventricles (e.g., human studies)
        mean_mesh_file = os.path.join(model_dir, 'Mean/LV_mean_scaled.vtk')


    mesh_1 = meshio.read(mean_mesh_file)

    # Load the PCA projection matrix (PHI)
    proj_matrix_path = os.path.join(model_dir, 'ML_augmentation/Projection_matrix.npy')
    PHI = np.asmatrix(np.load(proj_matrix_path))

    # Truncate to requested number of modes
    PHI = PHI[:, :num_modes]
    n_points = PHI.shape[0] // 3

    # Load mode boundaries for parameter constraints
    mode_bounds = np.loadtxt(os.path.join(model_dir, 'boundary_coeffs.txt'))
    mode_bounds = mode_bounds[:num_modes, :]

    # Load individual mode shapes
    modes = []
    for i in range(num_modes):
        modes.append(meshio.read(os.path.join(model_dir, 'Modes/Mode%d.vtk' % i)))

    # Load mesh boundary node indices
    epi_pts = np.load(os.path.join(model_dir, 'Boundary_nodes/EPI_points.npy'))  # Epicardial points
    endo_pts = np.load(os.path.join(model_dir, 'Boundary_nodes/ENDO_points.npy'))  # Endocardial points
    exterior_points_index = np.concatenate([epi_pts, endo_pts])

    # Subsample control points based on frequency parameter
    exterior_points_index = exterior_points_index[::cp_frequency]

    # Create reduced PCA matrix for control points
    PHI3 = np.reshape(np.array(PHI), (n_points, 3, num_modes), order='F')
    PHI3 = PHI3[exterior_points_index, :, :num_modes]
    PHI3 = np.reshape(PHI3, (-1, num_modes), order='F')

    # Calculate mean mode coefficients
    mean_modes = np.dot(meshio.read(mean_mesh_file).points.reshape((-1,), order='F'), np.array(PHI))

    # Define normalized mesh coordinate system axes
    mesh_vpc = np.array([0.9, 0., 0.])  # Ventricular-pulmonary commissure
    mesh_sax_normal = np.array([1., 0., 0.])  # Short-axis normal vector
    mesh_rv_direction = np.array([0., -0.75, -1.])  # Right ventricle direction
    mesh_axes = [mesh_vpc, mesh_sax_normal, mesh_rv_direction]

    return (mesh_1, mesh_1.points[exterior_points_index], PHI3, PHI, mode_bounds, mean_modes, mesh_axes)



def meshio_tetra_to_unstructured_grid(mesh):
    """
    Convert a meshio mesh with tetrahedral cells and associated data to a PyVista UnstructuredGrid.
    """
    # Check if the mesh contains tetrahedral cells - required for this conversion
    if "tetra" not in mesh.cells_dict:
        raise ValueError("Mesh must contain tetrahedral cells (type: 'tetra').")
    
    # Extract mesh points (vertices) and ensure proper data type for PyVista
    points = np.array(mesh.points, dtype=np.float32) # ensure float32 or float64
    
    # Extract tetrahedral cell connectivity data as integer array
    tets = np.array(mesh.cells_dict["tetra"], dtype=np.int64)
    
    # Get the total number of tetrahedral cells in the mesh
    n_cells = tets.shape[0]
    
    # Flatten the cell array and prepend with number of nodes per cell (always 4 for tets)
    # PyVista format requires: [n_nodes, node1, node2, node3, node4, n_nodes, node1, ...]
    cells = np.hstack([np.full((n_cells, 1), 4), tets]).flatten()
    
    # Create cell type array - all cells are tetrahedra (VTK cell type 10)
    celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    
    # Create PyVista UnstructuredGrid from cells, cell types, and points
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    
    # Optionally: add point data if available
    # Point data is associated with mesh vertices
    if mesh.point_data:
        # Transfer all point data arrays to the PyVista grid
        for key, val in mesh.point_data.items():
            grid.point_data[key] = val
    
    # Transfer cell data if available
    # Cell data is associated with mesh elements (tetrahedra)
    if mesh.cell_data:
        for key, val in mesh.cell_data.items():
            # mesh.cell_data[key] could be a list if multiple cell types
            # Handle case where cell data is stored as list (multiple cell types)
            if isinstance(val, list):
                # Select the one for tetrahedra
                # Find index of tetrahedral cells in the cells list
                tet_data = val[mesh.cells.index(("tetra", tets))]
                grid.cell_data[key] = tet_data
            else:
                # Direct assignment for single cell type data
                grid.cell_data[key] = val
    
    # Return the fully populated PyVista UnstructuredGrid
    return grid

def voxelizeUniform(mesh, z_res, gridsize=None, bp_channel=False):
    """
    Convert a 3D mesh to a voxelized representation on a uniform grid.
    
    Args:
        mesh: Input mesh object to voxelize
        sz (int): Grid resolution (creates sz x sz x sz grid) (size of cropped image)
        gridsize (float, optional): Physical size of the grid. Defaults to 2*sz
        offset (array, optional): Offset for grid positioning. Defaults to [sz, sz, 2*sz]
        bp_channel (bool): If True, also compute blood pool channel using hole filling
        
    Returns:
        numpy.ndarray or tuple: Voxelized mesh as boolean array.
                               If bp_channel=True, returns (myocardium, blood_pool) tuple
    """

    #set resolution, gridsize and spacing
    resolution = (z_res, z_res, z_res)

    if gridsize is None:
        gridsize =  z_res

    spacing = (gridsize / z_res, gridsize / z_res, gridsize / z_res)

    #transform the meshio mesh to a pyvista unstructured grid
    model = meshio_tetra_to_unstructured_grid(mesh)

    # Determine origin based on mesh bounds
    bounds = np.array(model.bounds)
    center = np.array([
        (bounds[1] + bounds[0]) / 2,
        (bounds[3] + bounds[2]) / 2,
        (bounds[5] + bounds[4]) / 2,
    ])
    origin = center - 0.5 * np.array(resolution) * spacing

    # Create grid
    grid = pv.ImageData()
    grid.dimensions = resolution
    grid.spacing = spacing
    grid.origin = origin

    # Sample mesh onto grid
    sampled = grid.sample(model)
    array = sampled.get_array('vtkValidPointMask')
    binary = np.reshape(np.round(array), sampled.dimensions, order='F').astype(bool)

    if bp_channel:
        # Generate blood pool channel using hole filling
        filled = []
        
        for k in range(binary.shape[-1]):
            # Process each slice along the long axis (from apex to base)
            # Expected pattern: empty -> myocardium only -> myocardium + blood pool -> empty
            
            # Fill holes to get blood pool
            filled.append(np.logical_xor(bfh(binary[..., k]), binary[..., k]))

            # Handle special case: non-closed myocardium at base
            if k > 0 and np.sum(filled[k]) == 0 and np.sum(filled[k-1]) != 0:
                # Use previous slice as template for blood pool estimation
                f = filled[k-1] + 0
                #calculates region that was blood pool in the previous slice, but is now missing
                small_bp = np.logical_xor(f, np.logical_and(f, binary[..., k]))
                #final filled blood pool
                filled[k] = np.logical_xor(bfh(np.logical_or(small_bp, binary[..., k])), binary[..., k])

            # Zero out blood pool for empty myocardium slices
            if np.sum(binary[..., k]) == 0:
                filled[k] = filled[k] * 0

        filled = np.moveaxis(np.array(filled), 0, -1)
        return binary, filled, origin

    return binary, origin


class evalLearnedInputs():
    """
    Evaluates learned neural network parameters to generate mesh geometry.
    
    This class takes learned parameters from a neural network and converts them
    back into 3D mesh coordinates using the PCA projection matrix.
    """
    
    def __init__(self, learned_inputs, mode_bounds, mode_means, mesh, PHI,dicom_exam):
        """
        Initialize the evaluator.
        
        Args:
            learned_inputs: Neural network that outputs PCA mode coefficients
            mode_bounds (array): Min/max bounds for each PCA mode
            mode_means (array): Mean values for each PCA mode
            mesh: Template mesh object
            PHI (array): PCA projection matrix
        """
        from copy import deepcopy

        self.learned_inputs = learned_inputs
        self.num_modes = learned_inputs.num_modes
        self.ones_input = torch.Tensor(np.ones((1, 1))).to(device)
        self.mode_bounds = mode_bounds
        self.mode_means = mode_means
        self.mesh = deepcopy(mesh)
        self.PHI = PHI + 0  # Create copy
        self.dicom_exam = dicom_exam

    def __call__(self, just_mesh=True):
        """
        Generate mesh from learned parameters.
        
        Args:
            just_mesh (bool): If True, return only mesh. If False, return additional info.
            
        Returns:
            mesh or tuple: Generated mesh, optionally with mode coefficients
        """
        # Get predictions from neural network
        with torch.no_grad():
            modes_output, volume_shift, x_shifts, y_shifts, volume_rotations = self.learned_inputs(self.ones_input)
            modes_output_reshape = modes_output.view(1,self.num_modes,self.learned_inputs.num_time_steps)
            modes_output = modes_output_reshape.cpu().numpy()
            volume_shift = volume_shift.cpu().numpy()[0]
            x_shifts = x_shifts.cpu().numpy()
            y_shifts = y_shifts.cpu().numpy()
            volume_rotations = volume_rotations.cpu().numpy()[0, 0]

        meshes = []
        rescaled_modes = []

        for time_step in range(len(self.dicom_exam.time_frames_to_fit)):

            # Convert normalized modes to actual PCA coefficients
            normed_modes = np.concatenate([modes_output[0,:,time_step]])
            
            rescaled_mode = (normed_modes * (self.mode_bounds[:, 1] - self.mode_bounds[:, 0]) / 2 + 
                            self.mode_means)
            
            # Generate 3D points from PCA coefficients
            gt_cp = ampsToPoints(rescaled_mode, self.PHI)
            #set 3D points to node positions of template mesh
            mesh_copy = deepcopy(self.mesh)
            mesh_copy.points = gt_cp
            meshes.append(mesh_copy)
            rescaled_modes.append(rescaled_mode)

        return meshes, rescaled_modes


def ampsToPoints(amps, PHI=None):
    """
    Convert PCA mode amplitudes back to 3D point coordinates.
    
    Args:
        amps (array): PCA mode amplitudes
        PHI (array): PCA projection matrix
        
    Returns:
        numpy.ndarray: 3D coordinates as (N, 3) array
    """
    # Project back from PCA space to coordinate space
    proj_back = PHI.dot(amps)
    n_points = proj_back.shape[1] // 3
    
    # Reshape from flattened format to (N, 3) coordinates
    Rec_Coords = np.zeros((n_points, 3))
    Rec_Coords[:, 0] = np.array(proj_back)[0, :][0:n_points]              # X coordinates
    Rec_Coords[:, 1] = np.array(proj_back)[0, :][n_points:2*n_points]     # Y coordinates  
    Rec_Coords[:, 2] = np.array(proj_back)[0, :][2*n_points:3*n_points]   # Z coordinates

    # Verify reshaping is correct
    assert np.array_equal(Rec_Coords, proj_back.reshape((3, -1)).T)
    return Rec_Coords


def set_initial_mesh_alignment(dicom_exam, mesh_axes, warp_and_slice_model, se):
    """
    Align the initial mesh orientation with the DICOM image coordinate system.

    Args:
        dicom_exam: DICOM examination data containing orientation information
        mesh_axes: Mesh coordinate system axes
        warp_and_slice_model: Model for mesh warping and slicing
        se: Slice extraction model
    """
    # Get initial mesh coordinate system
    initial_mesh_vpc, initial_mesh_sax_normal, initial_mesh_rv_direction = transformMeshAxes(
        mesh_axes, dicom_exam.sz, 0, np.eye(3))

    # Calculate rotation to align mesh SAX normal with DICOM SAX normal
    rotM = getRotationMatrix(initial_mesh_sax_normal, dicom_exam.sax_normal)
    new_mesh_rv_direction = np.dot(rotM, initial_mesh_rv_direction)
    print(new_mesh_rv_direction)

    # Determine valve direction from DICOM data
    valve_direction = dicom_exam.rv_direction if dicom_exam.valve_center is None else dicom_exam.aortic_valve_direction

    print(valve_direction)
    

    # # Calculate additional rotation for RV direction alignment
    rotM2 = getRotationMatrix(new_mesh_rv_direction, valve_direction)
    rotM = np.dot(rotM2, rotM)

    # Convert to Euler angles
    euler_rot = np.array(Rotation.from_matrix(rotM).as_euler('xyz'))
    print(euler_rot)

    # Apply rotation to models
    with torch.no_grad():
        warp_and_slice_model.initial_alignment_rotation += torch.Tensor(euler_rot).to(device)
        se.initial_alignment_rotation += torch.Tensor(euler_rot).to(device)


def getRotationMatrix(A, B):
    """
    Calculate rotation matrix that rotates vector A to align with vector B.
    
    Args:
        A (array): Source vector
        B (array): Target vector
        
    Returns:
        numpy.ndarray: 3x3 rotation matrix
        
    Uses Rodrigues' rotation formula for efficient computation.
    """
    # Normalize input vectors
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    
    # Calculate rotation parameters
    c = np.dot(A, B)  # Cosine of angle
    v = np.cross(A, B)  # Rotation axis
    
    # Skew-symmetric matrix for cross product
    v_x = np.array([[0, -v[2], v[1]], 
                    [v[2], 0, -v[0]], 
                    [-v[1], v[0], 0]])
    
    # Rodrigues' rotation formula
    v_x_squared = np.dot(v_x, v_x)
    rotM = np.eye(3) + v_x + v_x_squared * (1 / (1 + c))
    return rotM


def transformMeshAxes(mesh_axes, sz, vol_shifts_out, myR):
	"""
	Transform mesh coordinate system axes using rotation and translation.

	Args:
		mesh_axes (list): List of mesh axis vectors [vpc, sax_normal, rv_direction]
		sz (float): Scaling factor to convert from physical space to voxel space
		vol_shifts_out: Volume shift parameters
		myR (array): Rotation matrix
		
	Returns:
		list: Transformed mesh axes
	"""
	mesh_vpc, mesh_sax_normal, mesh_rv_direction = mesh_axes

	# Apply translation, rotation and scaling
	mesh_vpc_transformed = np.squeeze(np.dot(mesh_vpc - vol_shifts_out, myR.T) * sz)
	mesh_sax_normal_transformed = np.squeeze(np.dot(mesh_sax_normal, myR.T) * sz)
	mesh_rv_direction_transformed = np.squeeze(np.dot(mesh_rv_direction, myR.T) * sz)

	# Normalize direction vectors
	mesh_sax_normal_transformed = (mesh_sax_normal_transformed / 
									np.linalg.norm(mesh_sax_normal_transformed))
	mesh_rv_direction_transformed = (mesh_rv_direction_transformed / 
									np.linalg.norm(mesh_rv_direction_transformed))

	return [mesh_vpc_transformed, mesh_sax_normal_transformed, mesh_rv_direction_transformed]


def resetModel(learned_inputs, eli, use_bp_channel, sz, mesh_offset, ones_input, 
               pcaD, warp_and_slice_model, random_starting_mesh=True):
    """
    Reset neural network model parameters and initialize mesh state.
    
    Args:
        learned_inputs: Neural network model to reset
        eli: Mesh evaluator object
        use_bp_channel (bool): Whether to use blood pool channel
        sz (int): Image size parameter
        mesh_offset (array): Mesh positioning offset
        ones_input: Input tensor of ones
        pcaD: PCA decoder
        warp_and_slice_model: Mesh warping model
        random_starting_mesh (bool): Use random initial mesh instead of mean
    """
    print('Resetting model...')

    # Reset all network layers to zero
    with torch.no_grad():
        layers_to_reset = [
            learned_inputs.modes_output_layer,
            learned_inputs.volume_shift_layer, 
            learned_inputs.x_shift_layer,
            learned_inputs.y_shift_layer,
            learned_inputs.volume_rotations_layer
        ]
        
        for layer in layers_to_reset:
            layer.weight.fill_(0.)
            layer.bias.fill_(0.)

        if random_starting_mesh:
            # Use the dedicated function for random initialization
            init_random_start_mesh(learned_inputs, device)

        # Generate initial mesh
        msh = eli()

        # Use the dedicated function for voxelization
        mean_arr_batch, origin = prepare_voxelized_mean_array(msh, sz, use_bp_channel, device)

        # Update model with initial parameters
        modes_output, _, _, _, _ = learned_inputs(ones_input)
        predicted_cp = pcaD(modes_output)
        warp_and_slice_model.control_points = predicted_cp


def getTensorLabelsAndInputImage(dicom_exam):
    """
    Extract segmentation labels from DICOM exam for a specific time frame.
    
    Args:
        dicom_exam: DICOM examination object containing series data
        time_frame (int): Time frame index to extract
        
    Returns:
        torch.Tensor: Concatenated tensor labels for all series and slices
        
    Note:
        - Label 2 = myocardium, Label 3 = blood pool
        - Excludes series listed in dicom_exam.series_to_exclude
    """
    tensor_labels = []

    
    for series in dicom_exam:
        # Skip excluded series
        if series.name in dicom_exam.series_to_exclude:
            continue

        for slice_idx in range(series.slices):
            # Extract myocardium and blood pool masks
            myo = series.prepped_seg[:, slice_idx][None, None, ..., None] == 2
            bp = series.prepped_seg[:, slice_idx][None, None, ..., None] == 3
            
            # Combine channels
            tensor_labels.append(np.concatenate([myo, bp], axis=1))

    # Concatenate all labels and convert to tensor
    tensor_labels = np.concatenate(tensor_labels, axis=-1)
    tensor_labels = torch.Tensor(tensor_labels).to(device).permute(0, 1, 3, 4, 5, 2)

    return tensor_labels


def prepare_voxelized_mean_array(mesh, sz, use_bp_channel, device):
    """
    Prepare voxelized representation of mesh for neural network processing.
    
    Args:
        mesh: Input mesh to voxelize
        sz (int): Voxel grid size
        use_bp_channel (bool): Include blood pool channel
        mesh_offset (array): Mesh positioning offset
        device: PyTorch device for tensor allocation
        
    Returns:
        torch.Tensor: Voxelized mesh ready for network input
    """
    if use_bp_channel:
        mean_arr, mean_bp_arr, origin = voxelizeUniform(mesh, sz, bp_channel=use_bp_channel)
        mean_arr = mean_arr.astype('float')
        mean_arr_batch = torch.Tensor(np.concatenate([mean_arr[None, None], 
                                                     mean_bp_arr[None, None]], axis=1)).to(device)
    else:
        mean_arr, origin = voxelizeUniform(mesh, sz, bp_channel=use_bp_channel)
        mean_arr = mean_arr.astype('float')
        mean_arr_batch = torch.Tensor(np.tile(mean_arr[None, None], (1, 1, 1, 1, 1))).to(device)

    return mean_arr_batch, origin


def init_random_start_mesh(learned_inputs, device):
    """
    Initialize neural network with random starting mesh parameters.
    
    Args:
        learned_inputs: Neural network model
        device: PyTorch device
    """
    bias_n = len(learned_inputs.modes_output_layer.bias)
    random_bias = torch.Tensor(np.random.random((bias_n,)) * 2 - 1).to(device)
    learned_inputs.modes_output_layer.bias = torch.nn.Parameter(random_bias)


def get_time_frames_to_fit(dicom_exam, time_frames_to_fit, burn_in_length, num_cycle):
    """
    Determine which time frames to use for mesh fitting.
    
    Args:
        dicom_exam: DICOM examination object
        time_frames_to_fit: Specification of frames ('all', 'all_loop', or list)
        burn_in_length (int): Number of burn-in frames before actual data
        num_cycle (int): Number of cycles for looped fitting
        
    Returns:
        list: Time frame indices to process
    """
    if time_frames_to_fit == 'all':
        return list(range(-burn_in_length, 0)) + list(range(dicom_exam.time_frames))
    elif time_frames_to_fit == 'all_loop':
        return (list(range(-burn_in_length, 0)) + 
                list(range(dicom_exam.time_frames)) * int(num_cycle))
    else:
        return time_frames_to_fit


def save_results_post_training(dicom_exam, outputs, eli, se, sz, use_bp_channel, 
                              mesh_offset, learned_inputs, tensor_labels, save_mesh=True):
    """
    Save mesh fitting results after training completion.
    
    Args:
        dicom_exam: DICOM examination object
        outputs: Network outputs
        time_frame (int): Current time frame
        eli: Mesh evaluator
        se: Slice extractor
        sz (int): Image size
        use_bp_channel (bool): Use blood pool channel
        mesh_offset (array): Mesh offset
        learned_inputs: Neural network
        tensor_labels: Ground truth labels
        save_mesh (bool): Whether to save mesh files
        
    Returns:
        float: Final Dice coefficient
    """
    # Generate final mesh and rendered results
    msh, rescale_modes = eli()

    d0_values = []
    d1_values = []

    for index, time_step in enumerate(dicom_exam.time_frames_to_fit):
    
        d0 = dice_loss(outputs[index][:,:1], tensor_labels[:,:1,:,:,:,time_step])  # Myocardium
        d1 = dice_loss(outputs[index][:,1:], tensor_labels[:,1:,:,:,:,time_step])  # Blood pool

        d0_values.append(d0.item())
        d1_values.append(d1.item())

        # Store rendered and sliced mesh in DICOM exam object for visulization
        ones_input = torch.Tensor(np.ones((1, 1))).to(device)
        mesh_render = getSlices(se, msh[index], sz, use_bp_channel, mesh_offset, learned_inputs, ones_input, index)

        dicom_exam.fitted_meshes[time_step] = {}
        dicom_exam.fitted_meshes[time_step].setdefault('rendered_and_sliced', []).append(
            np.transpose(outputs[index].cpu().numpy()[0], (3, 1, 2, 0)))
        

        if save_mesh:
            # Create output directory
            output_folder = dicom_exam.folder['meshes']
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            output_folder_csv = dicom_exam.folder['meshes_csv']
            if not os.path.exists(output_folder_csv):
                os.makedirs(output_folder_csv)

            # Save mesh as VTK file
            mesh_filename = os.path.join(output_folder, "mesh_t=%d.vtk" % time_step) #these two meshes are exactly the same 
            meshio.write(mesh_filename, msh[index])

            # Save PCA mode coefficients as CSV
            csv_filename = os.path.join(output_folder_csv, "mesh_t=%d.csv" % time_step)
            with open(csv_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                for coefficient in rescale_modes[index]:
                    writer.writerow([coefficient])
    
    return d0_values ,d1_values


def getSlices(se, mesh, sz, use_bp_channel, mesh_offset, learned_inputs, ones_input, index):
    """
    Extract 2D slices from 3D mesh using the slice extraction model.
    
    Args:
        se: Slice extraction model
        mesh: Input 3D mesh
        sz (int): Image size
        use_bp_channel (bool): Include blood pool channel
        mesh_offset (array): Mesh positioning offset
        learned_inputs: Outputing PCA coefficients of meshes
        ones_input: Input tensor of ones
        
    Returns:
        torch.Tensor: Extracted 2D slices
    """
    # Get transformation parameters from network
    modes_output, global_offsets, x_shifts, y_shifts, global_rotations = learned_inputs(ones_input)
    per_slice_offsets = torch.cat([y_shifts[:, :, :,index] * 0, y_shifts[:, :, :,index], x_shifts[:, :, :,index]], dim=-1)

    # Use the dedicated function for voxelization
    mean_arr_batch, origin = prepare_voxelized_mean_array(mesh, sz, use_bp_channel, device)

    # Extract slices using the slice extraction model
    result = se([mean_arr_batch, global_offsets[:,:,:,index], per_slice_offsets, global_rotations[:,:,:,index]])

    return result


def slicewiseDice(arr1, arr2):
    """
    Calculate Dice coefficient for each slice between two binary arrays.
    
    Args:
        arr1 (array): First binary array (predictions)
        arr2 (array): Second binary array (ground truth)
        
    Returns:
        tuple: (slice_dice, has_target) arrays
               - slice_dice: Dice coefficient for each slice
               - has_target: Whether each slice has ground truth data
               
    The Dice coefficient measures overlap: 2 * |A ∩ B| / (|A| + |B|)
    """
    arr1 = np.squeeze(arr1)
    arr2 = np.squeeze(arr2)

    slice_dice = []
    has_target = []

    for i in range(len(arr1)):
        # Calculate Dice coefficient with small epsilon to avoid division by zero
        intersection = 2 * np.sum(arr1[i] * arr2[i])
        union = np.sum(arr1[i]) + np.sum(arr2[i]) + 0.00001
        dice = intersection / union
        
        slice_dice.append(dice)
        has_target.append((np.sum(arr2[i]) > 0) * 1)  # Binary indicator for target presence

    return np.array(slice_dice), np.array(has_target)


def dice_loss(pred, target, slice_weights=1):
    """
    Compute Dice loss for segmentation evaluation.
    
    The Dice loss is computed as 1 - Dice coefficient, where the Dice coefficient
    measures the overlap between predicted and target segmentation masks.
    
    Args:
        pred (torch.Tensor): Predicted segmentation probabilities/masks
                            Shape: (batch, channels, height, width, depth)
        target (torch.Tensor): Ground truth segmentation masks  
                              Shape: (batch, channels, height, width, depth)
        slice_weights (float or torch.Tensor): Weighting factor for different slices
    
    Returns:
        torch.Tensor: Mean Dice loss across all samples and channels
        
    Note:
        - Small epsilon (0.00001) added to numerator and denominator for numerical stability
        - Loss is averaged across all spatial dimensions (0,1,2,3)
        - Final loss is weighted by slice_weights parameter
    """
    # Calculate intersection (numerator of Dice coefficient)
    # Sum over batch, channel, height, width dimensions
    numerator = 2 * torch.sum(pred * target, axis=(0,1,2,3))
    
    # Calculate union (denominator of Dice coefficient) 
    # Sum of predicted and target masks
    denominator = torch.sum(pred + target, axis=(0,1,2,3))
    
    # Compute Dice coefficient with numerical stability
    # Add small epsilon to avoid division by zero
    dloss = (numerator + 0.00001) / (denominator + 0.00001)
    
    # Return mean loss across all computed values
    return torch.mean(dloss)