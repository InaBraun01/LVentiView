"""
Medical Image Coordinate and Visualization Utilities

This module provides utilities for:
1. Converting image pixel coordinates to 3D world coordinates using DICOM spatial information
2. Converting segmentation masks to RGB format for visualization
3. Processing mesh-based segmentation results for visualization and analysis
"""

import sys
import numpy as np


def planeToXYZ(img_size, position=np.array([0, 0, 0]), orientation=[np.array([1, 0, 0, 1, 0, 0])], 
				pixel_spacing=[1, 1]):
    """
    Convert image pixel coordinates to 3D world coordinates using DICOM spatial parameters.
    
    This function creates coordinate grids that map each pixel in a 2D image to its
    corresponding 3D world coordinate using DICOM orientation and position information.
    
    Args:
        img_size (tuple): Image dimensions (height, width)
        position (np.ndarray): 3D position of image origin [x, y, z] in world coordinates
        orientation (list): DICOM orientation vectors [row_direction + col_direction] (6 values)
        pixel_spacing (list): Pixel spacing [row_spacing, col_spacing] in mm
        
    Returns:
        tuple: (X, Y, Z) coordinate grids, each with shape (height, width)
               where each element contains the world coordinate for that pixel
    """
    # Split orientation into row and column direction vectors
    Yxyz = orientation[:3]   # Row direction vector (Y-axis in image space)
    Xxyz = orientation[3:]   # Column direction vector (X-axis in image space)
    
    # Extract position components
    Sx, Sy, Sz = position
    
    # Extract direction vector components
    Xx, Xy, Xz = Xxyz
    Yx, Yy, Yz = Yxyz
    
    # Extract pixel spacing (last two values)
    Di, Dj = pixel_spacing[-2:]
    
    # Create transformation matrix from image coordinates to world coordinates
    # This follows the DICOM standard for patient coordinate system transformation
    M = np.array([
        [Xx*Di, Yx*Dj, 0, Sx],  # X world coordinate
        [Xy*Di, Yy*Dj, 0, Sy],  # Y world coordinate  
        [Xz*Di, Yz*Dj, 0, Sz],  # Z world coordinate
        [0,     0,     0, 1],   # Homogeneous coordinate
    ])
    
    # Create pixel coordinate grids
    xv, yv = np.meshgrid(np.linspace(0, img_size[1], img_size[1]), 
                         np.linspace(0, img_size[0], img_size[0]))
    
    # Stack coordinates into homogeneous format (x, y, 0, 1)
    pts = np.concatenate([
        xv.reshape((1, -1)),      # x coordinates
        yv.reshape((1, -1)),      # y coordinates  
        xv.reshape((1, -1)) * 0,  # z coordinates (0 for 2D plane)
        xv.reshape((1, -1)) * 0 + 1  # homogeneous coordinate (1)
    ], axis=0)
    
    # Apply transformation to get 3D world coordinates
    X, Y, Z = np.dot(M, pts)[:3]
    
    # Reshape back to image dimensions
    X = X.reshape(img_size)
    Y = Y.reshape(img_size)
    Z = Z.reshape(img_size)
    
    return X, Y, Z


def to3Ch(img):
    """
    Convert image to 3-channel RGB format for visualization.
    
    Handles different input formats:
    - Grayscale (H,W): Converts to RGB by replicating channels
    - Segmentation masks with values [0,1,2,3]: Maps to colors (black, red, green, blue)
    - RGB images (H,W,3): Normalizes to [0,1] range
    
    Args:
        img (np.ndarray): Input image with shape (H,W) or (H,W,3)
        
    Returns:
        np.ndarray: 3-channel image with shape (H,W,3) and values in [0,1]
    """
    if len(img.shape) == 2:
        # Handle 2D input
        img_vals = np.unique(img)
        
        # Special case: segmentation mask with classes 0,1,2,3
        if set(img_vals) <= set([0, 1, 2, 3]):
            # Create color-coded segmentation: 
			# 0=black, 1=red (RV), 2=green (LV), 3=blue (blood pool)
            cimg = np.zeros(img.shape + (3,))
            for i in [1, 2, 3]:
                cimg[img == i, i-1] = 1  # Map class i to channel i-1
            return cimg
        else:
            # Regular grayscale image: normalize and replicate across 3 channels
            img = img / img.max()
            return np.tile(img[..., None], (1, 1, 3))
            
    elif len(img.shape) == 3:
        # Handle 3D input (already RGB)
        img = img / img.max()
        return img
        
    else:
        print(f'Error: input to to3Ch() should have shape (H,W) or (H,W,3), '
              f'but received input with shape: {img.shape}')


def prepMeshMasks(dicom_exam):
    """
    Process fitted mesh segmentations into numpy arrays for analysis and visualization.
    
    For each series in the DICOM exam, this function creates:
    - mesh_seg: Mean segmentation mask from fitted meshes
    - mesh_seg_std: Standard deviation of segmentation masks (if multiple meshes)
    
    Both arrays have shape (timesteps, slices, width, height, channels).
    
    Args:
        dicom_exam: DICOM examination object containing series and fitted mesh data
    """
    num_chans = 2  # Expected number of channels in mesh segmentation
    
    # Track slice positions across all series
    slice_positions = [0]
    
    for s in dicom_exam:
        # Skip excluded series
        if s.name in dicom_exam.series_to_exclude:
            continue
            
        # Update slice position tracking
        slice_positions.append(s.slices + slice_positions[-1])
        sind, eind = slice_positions[-2], slice_positions[-1]  # Start and end indices
        sz = s.prepped_seg.shape[2]  # Image size
        
        seg_means, seg_stds = [], []
        
        # Process each time frame
        for t in range(dicom_exam.time_frames):
            # Count fitted meshes for this time frame
            if t not in dicom_exam.fitted_meshes:
                num_fitted_meshes = 0
            else:
                num_fitted_meshes = len(dicom_exam.fitted_meshes[t]['rendered_and_sliced'])
            
            if num_fitted_meshes == 0:
                # No meshes available: create zero-filled segmentation
                seg_means.append(np.zeros((1, s.slices, sz, sz, num_chans)))
                seg_stds = None
                
            elif num_fitted_meshes == 1:
                # Single mesh: use directly
                mesh_data = dicom_exam.fitted_meshes[t]['rendered_and_sliced'][0]
                seg_means.append(mesh_data[sind:eind][None])
                seg_stds = None
                
            else:
                # Multiple meshes: compute mean and standard deviation
                mesh_list = [dicom_exam.fitted_meshes[t]['rendered_and_sliced'][k][sind:eind][None] 
                           for k in range(num_fitted_meshes)]
                seg_means.append(np.mean(mesh_list, axis=0))
                seg_stds.append(np.std(mesh_list, axis=0))
        
        # Concatenate results across time frames
        s.mesh_seg = np.concatenate(seg_means, axis=0)
        if seg_stds is not None:
            s.mesh_seg_std = np.concatenate(seg_stds, axis=0)
        else:
            s.mesh_seg_std = None
        
        # Ensure 3-channel format for downstream analysis
        if s.mesh_seg.shape[-1] == 2:
            # Add zero-filled first channel: [0, ch1, ch2] -> 3 channels
            zero_channel = s.mesh_seg[..., :1] * 0
            s.mesh_seg = np.concatenate([zero_channel, s.mesh_seg], axis=-1)
            
            if s.mesh_seg_std is not None:
                zero_channel_std = s.mesh_seg_std[..., :1] * 0
                s.mesh_seg_std = np.concatenate([zero_channel_std, s.mesh_seg_std], axis=-1)
                
        elif s.mesh_seg.shape[-1] == 1:
            # Add two zero-filled channels: [0, ch1, 0] -> 3 channels
            zero_channel = s.mesh_seg * 0
            s.mesh_seg = np.concatenate([zero_channel, s.mesh_seg, zero_channel], axis=-1)
            
            if s.mesh_seg_std is not None:
                zero_channel_std = s.mesh_seg_std * 0
                s.mesh_seg_std = np.concatenate([zero_channel_std, s.mesh_seg_std, zero_channel_std], axis=-1)