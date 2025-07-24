import sys
import numpy as np
import torch

def makeSliceCoordinateSystems(dicom_exam):
    """
    Create coordinate systems for each slice in a DICOM examination.
    
    This function processes DICOM data to generate coordinate systems for medical imaging
    slices. Each coordinate system is defined by three orthogonal vectors: two in-plane
    vectors and one out-of-plane vector (normal to the slice).
    
    Args:
        dicom_exam (list): List of DICOM slice objects, where each object has:
            - orientation (array-like): 6-element array containing two 3D vectors
              that define the in-plane orientation [x1, y1, z1, x2, y2, z2]
            - slices (int): Number of slices in this series
    
    Returns:
        numpy.ndarray: Array of shape (total_slices, 3, 3) where each slice contains
                      a 3x3 coordinate system matrix with columns representing:
                      [in_plane_1, in_plane_2, out_of_plane] vectors
    """
    slice_coordinate_systems = []
    
    # Iterate through each DICOM series in the examination
    for s in dicom_exam:
        # Extract the first in-plane orientation vector (first 3 elements)
        # This typically represents the row direction in the image
        in_plane_1 = s.orientation[:3]
        
        # Extract the second in-plane orientation vector (last 3 elements)  
        # This typically represents the column direction in the image
        in_plane_2 = s.orientation[3:]
        
        # Calculate the out-of-plane vector (slice normal) using cross product
        # This vector is perpendicular to both in-plane vectors
        out_of_plane = np.cross(in_plane_1, in_plane_2)
        
        # Create coordinate system for each slice in this series
        # All slices in a series typically share the same orientation
        for sl in range(s.slices):
            slice_coordinate_systems.append([in_plane_1, in_plane_2, out_of_plane])
    
    # Convert list to numpy array for efficient numerical operations
    slice_coordinate_systems = np.array(slice_coordinate_systems)
    return slice_coordinate_systems

def rotation_tensor(theta, phi, psi, n_comps, device):
    """
    Generate 3D rotation matrices using Euler angles (XYZ convention).
    
    This function creates batched 3D rotation matrices by composing rotations around
    the X, Y, and Z axes in that order. The rotations are applied as: R = Rz * Ry * Rx.
    
    Args:
        theta (torch.Tensor): Rotation angles around X-axis (in radians)
        phi (torch.Tensor): Rotation angles around Y-axis (in radians)  
        psi (torch.Tensor): Rotation angles around Z-axis (in radians)
        n_comps (int): Number of rotation matrices to generate (batch size)
        device (torch.device): Device to place tensors on (CPU or GPU)
    
    Returns:
        torch.Tensor: Batch of 3D rotation matrices with shape (n_comps, 3, 3)
                     Each matrix represents the combined rotation Rz * Ry * Rx
    
    Reference:
        Implementation based on PyTorch discussion:
        https://discuss.pytorch.org/t/constructing-a-matrix-variable-from-other-variables/1529/3
    """
    
    # Create constant tensors for batch operations
    # Shape: (n_comps, 1, 1) for broadcasting in matrix construction
    one = torch.ones(n_comps, 1, 1).to(device)
    zero = torch.zeros(n_comps, 1, 1).to(device)
    
    # Construct rotation matrix around X-axis
    # Rx = [[1,    0,      0   ],
    #       [0, cos(θ), sin(θ)],
    #       [0,-sin(θ), cos(θ)]]
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
     ), 2)
    
    # Construct rotation matrix around Y-axis  
    # Ry = [[cos(φ),  0, -sin(φ)],
    #       [   0,    1,     0   ],
    #       [sin(φ),  0,  cos(φ)]]
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
     ), 2)
    
    # Construct rotation matrix around Z-axis
    # Rz = [[cos(ψ), sin(ψ), 0],
    #       [-sin(ψ),cos(ψ), 0], 
    #       [   0,      0,   1]]
    rot_z = torch.cat((
        torch.cat((psi.cos(), psi.sin(), zero), 1),
        torch.cat((-psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
     ), 2)
    
    # Combine rotations using batch matrix multiplication: R = Rz * Ry * Rx
    # The order matters - this applies X rotation first, then Y, then Z
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))