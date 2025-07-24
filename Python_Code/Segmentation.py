"""
Cardiac MRI Segmentation Module

This module provides functions for segmenting cardiac MRI DICOM series,
calculating optimal crop sizes based on myocardium segmentation, and
processing the results for further analysis.
"""

import sys
import time
import numpy as np
import math
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Local imports
from Python_Code.Utilis.pytorch_segmentation_utils import (
    produce_segmentation_at_required_resolution, simple_shape_correction
)
from Python_Code.Utilis.visualizeDICOM import planeToXYZ, to3Ch


def segment(dicom_exam):
    """
    Perform cardiac segmentation on all series in a DicomExam object.
    
    This function processes each DICOM series by:
    1. Running deep learning segmentation at optimal resolution
    2. Resampling results back to original resolution
    3. Calculating optimal crop size based on myocardium
    4. Computing real-world 3D coordinates for each pixel
    5. Cropping and preparing data for further analysis
    
    Args:
        dicom_exam (DicomExam): DicomExam object containing series to segment
        
    Returns:
        None: Modifies the DicomExam object in-place, adding segmentation
              results to each series including:
              - seg: Full-resolution segmentation masks
              - prepped_seg: Cropped segmentation for analysis
              - prepped_data: Cropped image data for analysis
              - XYZs: Real-world 3D coordinates for each pixel
              - sz: Size of the cropped region
              - c1, c2: Center coordinates of the crop
    """

    crop_sizes = []
    
    # Process each series in the exam
    for series_idx, series in enumerate(dicom_exam.series):
        
        # Determine if this is a short-axis view
        is_sax = series.view in ['SAX', 'unknown']
        
        # Run segmentation at optimal resolution
        segmented_data, segmentation_mask, center_x, center_y = produce_segmentation_at_required_resolution(
            series.prepped_data, series.pixel_spacing, is_sax
        )
        # Resample segmentation back to original resolution
        # Order=0 ensures label preservation (nearest neighbor interpolation)
        zoom_factors = (1, 1, 1 / series.pixel_spacing[1], 1 / series.pixel_spacing[2])
        series.seg = zoom(segmentation_mask, zoom_factors, order=0)
        
        # Calculate optimal crop size based on myocardium segmentation
        crop_size = _calculate_optimal_crop_size(series.seg, center_x, center_y,margin_factor=2.5)
        crop_sizes.append(crop_size)

        # Ensure crop region stays within image bounds
        max_offset = min(
            segmentation_mask.shape[2] - crop_size, 
            segmentation_mask.shape[3] - crop_size
        )
        
        # Adjust center coordinates to keep crop within bounds
        center_x = np.clip(center_x - crop_size // 2, 0, max_offset)
        center_y = np.clip(center_y - crop_size // 2, 0, max_offset)
        series.c1, series.c2 = center_x, center_y


        # Generate 3D world coordinates for each pixel in each slice
        series.XYZs = []
        for slice_idx in range(series.slices):
            # Get 3D coordinate grids for the full slice
            X, Y, Z = planeToXYZ(
                segmentation_mask.shape[2:],  # Image dimensions
                series.image_positions[slice_idx],  # DICOM image position
                series.orientation,  # DICOM orientation vectors
                [1, 1]  # Pixel spacing in plane
            )
            
            # Crop coordinate grids to match segmentation crop
            X_cropped = X[center_y:center_y + crop_size, center_x:center_x + crop_size]
            Y_cropped = Y[center_y:center_y + crop_size, center_x:center_x + crop_size]
            Z_cropped = Z[center_y:center_y + crop_size, center_x:center_x + crop_size]
            
            # Stack coordinates and flatten for easy access
            xyz_coords = np.stack([X_cropped.ravel(), Y_cropped.ravel(), Z_cropped.ravel()], axis=1)

            # print(np.concatenate([X.reshape((sz**2,1)), Y.reshape((sz**2,1)), Z.reshape((sz**2,1))]).sum())
            series.XYZs.append(xyz_coords)

        # Crop and transpose data to match expected format (time, slice, y, x)
        crop_slice_x = slice(center_x, center_x + crop_size)
        crop_slice_y = slice(center_y, center_y + crop_size) 
        
        series.prepped_seg = np.transpose(
            segmentation_mask[:, :, crop_slice_x, crop_slice_y], (0, 1, 3, 2)
        )

        series.prepped_data = np.transpose(
            segmented_data[:, :, crop_slice_x, crop_slice_y], (0, 1, 3, 2)
        )

        # Apply shape correction for short-axis views
        if is_sax:
            print("  Applying shape correction for SAX view")
            series.prepped_seg = simple_shape_correction(series.prepped_seg)
            

    #calculate crop size for all series in dicom exam
    dicom_exam.sz = int(np.mean(crop_sizes))


def _calculate_optimal_crop_size(segmentation, center_x, center_y, margin_factor=2):
    """
    Calculate optimal crop size based on myocardium segmentation extent.
    
    This function analyzes the myocardium segmentation (label=2) to determine
    the minimum crop size needed to encompass the cardiac structures with
    appropriate margin.
    
    Args:
        segmentation (np.ndarray): 4D segmentation array (time, slice, height, width)
        center_x (int): X-coordinate of the crop center
        center_y (int): Y-coordinate of the crop center
        margin_factor (float): Multiplier for adding margin around detected structure
        
    Returns:
        int: Optimal crop size (always even number for compatibility)
        
    Algorithm:
        1. Find a representative time frame and slice with myocardium
        2. Measure structure extent in 4 directions (horizontal, vertical, 2 diagonals)
        3. Average the measurements to get representative diameter
        4. Apply margin factor and round to even number
    """
    time_steps, z_stacks, height, width = segmentation.shape
    
    # Start with middle time frame and slice
    mid_time = time_steps // 2
    mid_slice = z_stacks // 2
    
    # Find a slice with myocardium segmentation (label = 2)
    myocardium_mask = None
    slice_idx = mid_slice
    
    # Search backwards from middle slice
    while slice_idx >= 0:
        candidate_mask = segmentation[mid_time, slice_idx, :, :]
        if np.sum(candidate_mask == 2) > 0:  # Found myocardium
            myocardium_mask = candidate_mask
            break
        slice_idx -= 1
    
    # If no myocardium found in backwards search, try other time frames
    if myocardium_mask is None:
        for time_idx in [mid_time + 1, mid_time - 1]:
            if 0 <= time_idx < time_steps:
                candidate_mask = segmentation[time_idx, mid_slice, :, :]
                if np.sum(candidate_mask == 2) > 0:
                    myocardium_mask = candidate_mask
                    break
    
    # Fallback: use any available myocardium mask
    if myocardium_mask is None:
        myocardium_mask = np.zeros((height, width))
        print("Warning: No myocardium segmentation found, using default crop size")
    
    # Measure structure extent in multiple directions
    diameters = []
    
    # Horizontal diameter through center
    horizontal_coords = np.where(myocardium_mask[center_x, :] == 2)[0]
    if len(horizontal_coords) > 0:
        horizontal_diameter = horizontal_coords[-1] - horizontal_coords[0]
        diameters.append(horizontal_diameter)
    
    # Vertical diameter through center  
    vertical_coords = np.where(myocardium_mask[:, center_y] == 2)[0]
    if len(vertical_coords) > 0:
        vertical_diameter = vertical_coords[-1] - vertical_coords[0]
        diameters.append(vertical_diameter)
    
    # Main diagonal diameter
    main_diag_coords = np.where(np.diag(myocardium_mask) == 2)[0]
    if len(main_diag_coords) > 0:
        # Scale by sqrt(2) to account for diagonal distance
        main_diag_diameter = (main_diag_coords[-1] - main_diag_coords[0]) * math.sqrt(2)
        diameters.append(main_diag_diameter)
    
    # Anti-diagonal diameter
    anti_diag_coords = np.where(np.diag(np.rot90(myocardium_mask)) == 2)[0]
    if len(anti_diag_coords) > 0:
        anti_diag_diameter = (anti_diag_coords[-1] - anti_diag_coords[0]) * math.sqrt(2)
        diameters.append(anti_diag_diameter)
    
    # Calculate average diameter or use default
    if diameters:
        average_diameter = np.mean(diameters)
    else:
        average_diameter = 64  # Default size if no measurements available
        print("Warning: Could not measure myocardium extent, using default diameter")
    
    # Apply margin and ensure even number
    crop_size_with_margin = average_diameter * margin_factor
    final_crop_size = _round_up_to_even(crop_size_with_margin)
    
    return final_crop_size


def _round_up_to_even(value):
    """
    Round a number up to the nearest even integer.
    
    Args:
        value (float): Number to round up
        
    Returns:
        int: Nearest even integer greater than or equal to input
        
    """
    return int(math.ceil(value / 2.0)) * 2
