import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from Python_Code.Utilis.fit_mesh_models_utils import makeSliceCoordinateSystems, rotation_tensor
from Python_Code.Utilis.interpolate_spline import interpolate_spline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_not_trainable(m):
    """
    Sets all parameters in a model to not require gradients, 
    making them untrainable.
    """
    for param in m.parameters():
        param.requires_grad = False


class SliceExtractor(torch.nn.Module):
    """
    Module for slice extraction from 3D volumes.
    Allows configurable global and slice-level translation and rotation.
    """

    def __init__(
        self, 
        input_volume_shape, 
        dicom_exam, 
        allow_global_shift_xy=False, 
        allow_global_shift_z=False, 
        allow_slice_shift=False, 
        allow_rotations=False, 
        series_to_exclude=[]
    ):
        """
        Args:
            input_volume_shape (tuple): Volume shape (depth, height, width).
            dicom_exam (object): DICOM exam information containing metadata and point grids.
            allow_global_shift_xy (bool): Allow XY global translation.
            allow_global_shift_z (bool): Allow Z global translation.
            allow_slice_shift (bool): Allow per-slice translation.
            allow_rotations (bool): Allow global rotations.
            series_to_exclude (list): Series name strings to exclude.
        """
        super().__init__()
        self.vol_shape = input_volume_shape

        # Set flags for allowed types of shifts and rotations
        self.allow_shifts = allow_global_shift_xy or allow_global_shift_z
        self.allow_global_shift_xy = allow_global_shift_xy
        self.allow_global_shift_z = allow_global_shift_z
        self.allow_slice_shifts = allow_slice_shift
        self.allow_rotations = allow_rotations

        # Default initial rotation (can be useful for registration)
        self.initial_alignment_rotation = torch.zeros(3, device=device)

        # Gather world coordinates for relevant slices, skipping excluded series
        slices = []
        for s in dicom_exam:
            if s.name not in series_to_exclude:
                slices.extend(s.XYZs)   # Each s.XYZs is an array of world coordinates per slice
                # Concatenate all slice coordinates into one big array
                all_xyz_coords = np.concatenate(s.XYZs, axis=0)  # Shape: (total_voxels, 3)

                # Compute the minimum coordinate along each axis (x, y, z)
                volume_origin = np.min(all_xyz_coords, axis=0)
        
        # Total number of 2D slices across all included series
        self.num_slices = len(slices)

        # Stack and shape: (1, N_points_total, 3) where N_points_total = num_slices * slice_height * slice_width
        self.grid = np.concatenate(slices)[None]

        # Create an orientation (local coordinate system) for each slice
        self.coordinate_system = torch.Tensor(makeSliceCoordinateSystems(dicom_exam)).to(device)
        
        # Normalize grid to be centered and isotropic in DICOM space
        center = dicom_exam.center

        # self.grid = (self.grid - center) / dicom_exam.sz

        self.grid = (self.grid - volume_origin)

        self.grid = 2 * (self.grid / (self.vol_shape)) - 1
        self.grid = torch.Tensor(self.grid).to(device)

    def forward(self, args):
        """
        Generates extracted slices from the input volume, applying global and/or per-slice
        geometric transformations as configured. Applies global and/or per-slice
        geometric transformations to a grid and then samples the mesh onto the grid.

        Args:
            args (tuple): 
              - volume (Tensor): The input 3D volume tensor to sample from [batch_size, num_channels, D, H, W].
              - global_offsets (Tensor): Offsets to apply (batch, point, 3).
              - per_slice_offsets (Tensor): Offsets per slice (batch, slice, 3).
              - global_rotations (Tensor): Rotation angles (batch, 1, 3).
        Returns:
            torch.Tensor: The set of resulting sampled slices, batched.
        """
        vol, global_offsets, per_slice_offsets, global_rotations = args

        # Duplicate the grid for each item in the batch
        batched_grid = self.grid.repeat(global_offsets.shape[0], 1, 1)

        # Determine if we should actually rotate (r_weight)
        r_weight = 1 if self.allow_rotations else 0

        # Create rotation tensor for the batch, optionally adding initial alignment
        R = rotation_tensor(
            global_rotations[...,:1]*r_weight + self.initial_alignment_rotation[0],
            global_rotations[...,1:2]*r_weight + self.initial_alignment_rotation[1],
            global_rotations[...,2:]*r_weight + self.initial_alignment_rotation[2],
            1, device
        )

        # Calculate the orientations for all slices post-rotation
        rotated_coords = torch.bmm(self.coordinate_system, R.repeat(self.coordinate_system.shape[0], 1, 1))
        
        # Apply the rotation to every point in batched_grid
        batched_grid = torch.bmm(batched_grid, R)

        # Apply global translation offsets, respecting allowed axes
        if self.allow_shifts:
            if not self.allow_global_shift_xy:
                global_offsets[:,:,1] = 0  # No shift in X
                global_offsets[:,:,2] = 0  # No shift in Y
            if not self.allow_global_shift_z:
                global_offsets[:,:,0] = 0  # No shift in Z
            batched_grid += global_offsets

        # Apply per-slice offsets, using each slice's rotated coordinate system
        if self.allow_slice_shifts:
            # Reshape so that each slice can have its own offset
            batched_grid = batched_grid.view(1, self.num_slices, -1, 3)
            # Offset each slice's grid along its own primary axes
            batched_grid += torch.bmm(per_slice_offsets[0,:,None], rotated_coords)[None, :]
            # Collapse back to original grid shape
            batched_grid = batched_grid.view(1, -1, 3)

        # Reshape to [batch, num_slices, H, W, 3] for grid_sample; then permute to match expected order
        batched_grid = batched_grid.view(-1, self.num_slices, self.vol_shape[0], self.vol_shape[1], 3)
        batched_grid = batched_grid.permute(0, 2, 3, 1, 4)  # [batch, H, W, num_slices, 3]

        # Sample from the volume using the computed transformed sampling points
        # grid_sample will extract the (batch of) slices as specified by batched_grid
        res = F.grid_sample(vol, batched_grid, align_corners=True, mode='bilinear')

        return res


class GivenPointSliceSamplingSplineWarpSSM(torch.nn.Module):
    """
    Module for extracting slices after applying spline-based warping
    to given control points, incorporating translations and rotations.
    """

    def __init__(
        self, 
        input_volume_shape, 
        control_points, 
        dicom_exam, 
        allow_global_shift_xy=False,
        allow_global_shift_z=False,
        allow_slice_shift=False,
        allow_rotations=False,
        series_to_exclude=[]
    ):
        """
        Args:
            input_volume_shape (tuple): Shape of the input volume.
            control_points (array): Control points for spline warps.
            dicom_exam (object): DICOM exam with metadata.
            allow_global_shift_xy (bool): Allow XY translation.
            allow_global_shift_z (bool): Allow Z translation.
            allow_slice_shift (bool): Allow per slice translation.
            allow_rotations (bool): Allow global rotation.
            series_to_exclude (list): List of series names to exclude.
        """
        super().__init__()
        self.vol_shape = input_volume_shape
        self.control_points = torch.Tensor(control_points).to(device)
        self.allow_shifts = allow_global_shift_xy or allow_global_shift_z
        self.allow_global_shift_xy = allow_global_shift_xy
        self.allow_global_shift_z = allow_global_shift_z
        self.allow_slice_shifts = allow_slice_shift
        self.allow_rotations = allow_rotations
        self.de_sax_normal = torch.Tensor(dicom_exam.sax_normal).to(device)
        self.initial_alignment_rotation = torch.zeros(3, device=device)
        self.sz = self.vol_shape[0]

        slices = []
        #only take into consideration slices that should not be excluded
        for s in dicom_exam:
            if s.name not in series_to_exclude:
                slices.extend(s.XYZs) # Each s.XYZs is an array of world coordinates per slices

                # Concatenate all slice coordinates into one big array
                all_xyz_coords = np.concatenate(s.XYZs, axis=0)  # Shape: (total_voxels, 3)

                # Compute the minimum coordinate along each axis (x, y, z)
                volume_origin = np.min(all_xyz_coords, axis=0)


        self.num_slices = len(slices)
        self.grid = np.concatenate(slices)[None]
        self.grid = (self.grid - volume_origin)

        self.grid = 2 * (self.grid / (self.vol_shape)) - 1

        self.grid = torch.Tensor(self.grid).to(device)

        # Create an orientation (local coordinate system) for each slice
        self.coordinate_system = torch.Tensor(makeSliceCoordinateSystems(dicom_exam)).to(device)


    def forward(self, args):
        """
        Args:
            args (tuple): A tuple containing:
                - warped_control_points: Tensor of warped control point coordinates.
                - volume (Tensor): The input 3D volume tensor to sample from [batch_size, num_channels, D, H, W]
                - global_offsets: Global translation vectors for each sample.
                - per_slice_offsets: Per-slice shift vectors, allowing slice-wise deformation.
                - global_rotations: Global rotation angles (yaw, pitch, roll) per sample.
        Returns:
            torch.Tensor: Output slices after spline-based warping and resampling from the volume.
        """
        warped_control_points, vol, global_offsets, per_slice_offsets, global_rotations = args

        # Repeat static control grid and control points to match batch size
        # warped_control_points.shape[0] = batch size
        batched_grid = self.grid.repeat(warped_control_points.shape[0], 1, 1)
        batched_control_points = self.control_points.repeat(warped_control_points.shape[0], 1, 1)

        # --- Global Rotation ---
        # If rotation is allowed, use global rotations + initial alignment;
        # otherwise set rotation weight to zero, disabling it.

        r_weight = 1 if self.allow_rotations else 0
        R = rotation_tensor(
            global_rotations[..., :1] * r_weight + self.initial_alignment_rotation[0],  # yaw
            global_rotations[..., 1:2] * r_weight + self.initial_alignment_rotation[1],  # pitch
            global_rotations[..., 2:]  * r_weight + self.initial_alignment_rotation[2],  # roll
            1, 
            device=device
        )

        #Apply rotation to the sampling grid
        batched_grid = torch.bmm(batched_grid, R)  # (B, N, 3) x (B, 3, 3) -> (B, N, 3)

        # Rotate the coordinate system vectors (used for per-slice offset)
        rotated_coords = torch.bmm(
            self.coordinate_system,               # (num_slices, 3, 3)
            R.repeat(self.coordinate_system.shape[0], 1, 1)  # repeat R for each slice
        )

        # --- Global Translation ---
        if self.allow_shifts:
            # Optionally zero out shift components depending on configuration
            if not self.allow_global_shift_xy:
                global_offsets[:, :, 1] = 0  # y
                global_offsets[:, :, 2] = 0  # z
            if not self.allow_global_shift_z:
                global_offsets[:, :, 0] = 0  # x

            # Apply global translation to the grid
            batched_grid += global_offsets

        # --- Per-Slice Translation ---
        if self.allow_slice_shifts:
            # Reshape grid to [1, num_slices, num_points_per_slice, 3]
            batched_grid = batched_grid.view(1, self.num_slices, -1, 3)

            # Apply slice-wise translation using matrix multiplication of offsets and directions
            batched_grid += torch.bmm(per_slice_offsets[0, :, None], rotated_coords)[None, :, :, :]

            # Flatten back to shape [1, N, 3] for interpolation
            batched_grid = batched_grid.view(1, -1, 3)


        # --- Spline Interpolation ---
        #Interpolates how each point in batched_grid is defomed based on control points using B-spline
        #for warp points I know deformed position as well as undeformed position
        #Interpolate where each voxel in the deformed (new) volume originally was in the undeformed input volume.
        interpolated_sample_locations = interpolate_spline(
            train_points=warped_control_points.flip(-1) * 2 - 1,  # Normalize to [-1, 1]
            train_values=batched_control_points.flip(-1) * 2 - 1,
            query_points=batched_grid,
            order=1  # Linear interpolation
        ).view(-1, self.num_slices, self.vol_shape[0], self.vol_shape[1], 3)


        # Rearrange dimensions to [B, D, H, W, 3] for grid_sample
        interpolated_sample_locations = interpolated_sample_locations.permute(0, 2, 3, 1, 4)

        # Flatten all dimensions except the last one (which should be the 3D coordinates)
        flat_coords = interpolated_sample_locations.reshape(-1, 3)

        # Compute min and max per axis (X, Y, Z)
        min_vals = flat_coords.min(dim=0).values
        max_vals = flat_coords.max(dim=0).values

        z_min = min_vals[0]
        z_max = max_vals[0]

        # File to append to
        csv_file = "z_bounds_log.csv"

        # Append mode
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([z_min.item(), z_max.item()])  # Just Z bounds

        # --- Sample the Volume ---
        # you sample the original volume (vol) at the undeformed positions that correspond to the deformed voxel locations.
        # Thus, it tells you for each voxel in the deformed location wether it was 
        # inside or outside the original segmentation mask.
        res = F.grid_sample(vol, interpolated_sample_locations, align_corners=True, mode='bilinear')
        return res




class LearnableInputPPModel(torch.nn.Module):
    """
    End-to-end model that uses a set of learnable input parameters
    to predict mesh modes and transformation parameters. 
    Useful for direct optimization of parameters for a single image/subject.
    """
    def __init__(self, learned_inputs, pcaD, warp_and_slice_model):
        super().__init__()
        self.learned_inputs = learned_inputs                 # Network that outputs parameters from a constant input
        self.pcaD = pcaD                                     # PCA decoder to map modes to mesh control points
        self.warp_and_slice_model = warp_and_slice_model     # Warping and slicing module (takes mesh and outputs slices)
        self.num_modes = learned_inputs.num_modes            # Number of PCA modes

    def forward(self, args):
        """
        Forward method to predict mesh, perform slicing, and return all intermediate parameters.
        
        Args:
            args (tuple): (voxelized_mean_mesh, ones_input[, temp])
                voxelized_mean_mesh: Mean/template mesh as a 3D volume (for grid sampling)
                ones_input: 1-vector (dummy input for learned_inputs)
                temp: Optional - extra value passed on (not commonly used)
        Returns:
            tuple: (predicted_slices, modes_output, volume_shift, global_rotations, predicted_cp, slice_shifts)
        """
        # Unpack arguments and handle optional temp
        voxelized_mean_mesh, ones_input = args

        # Use the learnableInputs module to get all mesh and transformation parameters
        modes_output, volume_shift, x_shifts, y_shifts, global_rotations = self.learned_inputs(ones_input)
        
        # Concatenate per-slice shift parameters to form a [N, num_slices, 3] offset tensor
        # Here, the last dimension: [x_shift, y_shift, z_shift=0]
        slice_shifts = torch.cat([x_shifts, y_shifts, y_shifts*0], dim=-1)

        # Decode the low-dimensional modes into a full set of mesh control points
        predicted_cp = self.pcaD(modes_output)
        
        # Apply the warping and slicing model to get predicted 2D slices
        predicted_slices = self.warp_and_slice_model([
            predicted_cp, voxelized_mean_mesh, volume_shift, slice_shifts, global_rotations])

        
        # Return all outputs for further loss computation, analysis, or optimization
        return predicted_slices, modes_output, volume_shift, global_rotations, predicted_cp, slice_shifts


class PCADecoder(torch.nn.Module):
    """
    Decodes low-dimensional PCA shape modes into full mesh control points.
    Handles de-normalization, scaling, and translation of the mesh.
    """
    def __init__(self, num_modes, num_points, mode_bounds, mode_means, offset, mesh_origin, scale=64):
        super().__init__()
        # Compute the scaling (span) and mean for each mode for de-normalization
        self.mode_spans = torch.Tensor((mode_bounds[:,1] - mode_bounds[:,0]) / 2).to(device)
        self.mode_means = torch.Tensor(mode_means).to(device)
        self.offset = torch.Tensor(offset).to(device)
		#correspond to the scale used when you generated the meshes that PCA components were derived from
        #meshes were generated on a 128x128 grid
        self.scale = scale 
        self.mesh_origin = mesh_origin
        self.num_points = num_points
        # Linear transformation from modes to all mesh coordinates
        # Weights of linear projection are set as PCA components
        self.fc1 = nn.Linear(num_modes, 3*num_points)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Modes input, shape [batch_size, num_modes]
        Returns:
            torch.Tensor: Mesh control points [batch_size, num_points, 3]
        """
        batch_size = x.shape[0]
        # De-normalize: unscale and add the mean for each mode
        x = x * self.mode_spans + self.mode_means
        # Linear projection to mesh points
        mesh_points = self.fc1(x)                                              # [batch_size, 3*num_points]
        mesh_points = mesh_points.view(batch_size, 3, self.num_points)         # [batch_size, 3, num_points]
        mesh_points = torch.transpose(mesh_points, 1, 2)                       # [batch_size, num_points, 3]

        # Ensure origin is a tensor on the same device
        origin_tensor = torch.tensor(self.mesh_origin, dtype=mesh_points.dtype, device=mesh_points.device)

        # Convert scale to tensor if needed (broadcastable)
        scale_tensor = torch.tensor(self.scale, dtype=mesh_points.dtype, device=mesh_points.device)

        # Apply normalization
        mesh_points = (mesh_points - origin_tensor) / scale_tensor

        return mesh_points


class learnableInputs(torch.nn.Module):
    """
    Module for producing mesh and transformation parameters as learnable outputs
    from a constant input vector ("dummy" input).
    Allows fitting parameters directly via gradient descent for a specific subject/image.
    """
    def __init__(self, num_modes=12, num_slices=10):
        super().__init__()
        # Each layer projects a 1-vector to the desired number of variables
        self.modes_output_layer      = nn.Linear(1, num_modes)
        self.volume_shift_layer      = nn.Linear(1, 3)
        self.x_shift_layer           = nn.Linear(1, num_slices)
        self.y_shift_layer           = nn.Linear(1, num_slices)
        self.volume_rotations_layer  = nn.Linear(1, 3)
        self.num_slices = num_slices
        self.num_modes = num_modes

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Batch input of shape [batch_size, 1]
        Returns:
            tuple: (modes_output, volume_shift, x_shifts, y_shifts, volume_rotations)
                - modes_output: [batch_size, num_modes]
                - volume_shift: [batch_size, 1, 3]
                - x_shifts:     [batch_size, num_slices, 1]
                - y_shifts:     [batch_size, num_slices, 1]
                - volume_rotations: [batch_size, 1, 3]
        """
        batch_size = x.shape[0]
        # Project 1-vector to each parameter type, then reshape for correct broadcasting
        modes_output     = self.modes_output_layer(x)                               # Shape: [B, num_modes]
        volume_shift     = self.volume_shift_layer(x).view(batch_size, 1, 3)        # Shape: [B, 1, 3]
        x_shifts         = self.x_shift_layer(x).view(batch_size, self.num_slices, 1)
        y_shifts         = self.y_shift_layer(x).view(batch_size, self.num_slices, 1)
        volume_rotations = self.volume_rotations_layer(x).view(batch_size, 1, 3)    # [B, 1, 3]
        return (modes_output, volume_shift, x_shifts, y_shifts, volume_rotations)
    



def makeFullPPModelFromDicom(
    sz, num_modes, starting_cp, dicom_exam, 
    mode_bounds, mode_means, PHI3, offset, mesh_origin,
    allow_global_shift_xy=True, allow_global_shift_z=True,
    allow_slice_shift=False, allow_rotations=False, 
    series_to_exclude=[]
):
    """
    Utility function for composing the full pipeline model: warp/slice, PCA decoder, and supporting models.

    Returns:
        Tuple of (pcaD, warp_and_slice_model, learned_inputs, li_model)
    """
    warp_and_slice_model = GivenPointSliceSamplingSplineWarpSSM(
        (sz,sz,sz), starting_cp[None], dicom_exam,
        allow_global_shift_xy=allow_global_shift_xy,
        allow_global_shift_z=allow_global_shift_z,
        allow_slice_shift=allow_slice_shift,
        allow_rotations=allow_rotations,
        series_to_exclude=series_to_exclude
    )
    warp_and_slice_model.to(device)

    num_points = int(PHI3.shape[0] / 3)
    num_slices = warp_and_slice_model.num_slices

    #initalize PCADecoder weights with PCA componetes mapping output PCA coefficients to meshes 
    #initalize biases with 0
    pcaD = PCADecoder(num_modes, num_points, mode_bounds[:num_modes], mode_means[:num_modes], offset=offset, scale = sz, mesh_origin = mesh_origin)
    with torch.no_grad():
        pcaD.fc1.weight = nn.Parameter(torch.Tensor(PHI3))
        pcaD.fc1.bias.fill_(0.)
    pcaD.apply(make_not_trainable)
    pcaD.to(device)

    learned_inputs = learnableInputs(num_modes=num_modes, num_slices=num_slices)
    #initialize weights and biases with 0
    with torch.no_grad():
        for m in [learned_inputs.modes_output_layer, learned_inputs.volume_shift_layer, learned_inputs.x_shift_layer, learned_inputs.y_shift_layer, learned_inputs.volume_rotations_layer]:
            m.weight.fill_(0.)
            m.bias.fill_(0.)

    li_model = LearnableInputPPModel(learned_inputs, pcaD, warp_and_slice_model,)
    li_model.to(device)

    return pcaD, warp_and_slice_model, learned_inputs, li_model