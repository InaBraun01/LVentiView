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
    produce_segmentation_at_required_resolution, simple_shape_correction,
)
from Python_Code.Utilis.visualizeDICOM import planeToXYZ, to3Ch


def segment(dicom_exam,crop_size=None, margin_factor = 2):
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

        if crop_size:
            crop_size = crop_size
        else:
            crop_size = _calculate_optimal_crop_size(series.seg, center_x, center_y)

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

            center_y = int(center_y)
            center_x = int(center_x)
            crop_size = int(crop_size)
            
            # Crop coordinate grids to match segmentation crop
            X_cropped = X[center_y:center_y + crop_size, center_x:center_x + crop_size]
            Y_cropped = Y[center_y:center_y + crop_size, center_x:center_x + crop_size]
            Z_cropped = Z[center_y:center_y + crop_size, center_x:center_x + crop_size]

            
            # Stack coordinates and flatten for easy access
            xyz_coords = np.stack([X_cropped.ravel(), Y_cropped.ravel(), Z_cropped.ravel()], axis=1)

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
            series.prepped_seg = simple_shape_correction(series.prepped_seg)
            

    #calculate crop size for all series in dicom exam
    dicom_exam.sz = int(np.mean(crop_sizes))

def _calculate_optimal_crop_size(segmentation, center_x, center_y, margin_factor=2,
                                  myocardium_label=2):
    """
    Calculate optimal crop size based on myocardium segmentation extent.

    Uses the bounding box of the myocardium mask rather than line-intersection
    measurements through the center, since bounding box extent is robust to
    shape — works equally well for ring-shaped myocardium (SAX) and
    horseshoe-shaped myocardium (LAX), where a center-symmetric line-based
    measurement can badly under- or over-estimate the true extent.

    Rather than relying on a single slice, this computes the bounding-box
    diameter at every (time, z) combination in the volume and takes the
    median across all combinations that contain myocardium. This is more
    robust than picking one representative slice/frame, since it averages
    out cardiac-phase effects (e.g. systole vs diastole) as well as
    slice-position effects (e.g. apex/base having a smaller footprint).

    Args:
        segmentation (np.ndarray): 4D segmentation array (time, slice, height, width)
        center_x (int): X-coordinate of the crop center (kept for signature
            compatibility / potential future use; not required by the
            bounding-box approach itself)
        center_y (int): Y-coordinate of the crop center (see above)
        margin_factor (float): Multiplier for adding margin around detected structure
        myocardium_label (int): label value corresponding to myocardium in
            this segmentation's class mapping — VERIFY this against your
            actual class indices (SAX and LAX checkpoints may use different
            label orderings; don't assume 2 for both).

    Returns:
        int: Optimal crop size (always even number for compatibility)
    """
    time_steps, z_stacks, height, width = segmentation.shape

    diameters = []
    for time_idx in range(time_steps):
        for slice_idx in range(z_stacks):
            candidate_mask = segmentation[time_idx, slice_idx, :, :]
            rows, cols = np.where(candidate_mask == myocardium_label)

            if len(rows) == 0:
                continue

            height_extent = rows.max() - rows.min()
            width_extent = cols.max() - cols.min()
            diameter = max(height_extent, width_extent)
            diameters.append(diameter.item())

    if not diameters:
        print("Warning: No myocardium segmentation found, using default crop size")
        return _round_up_to_even(64 * margin_factor)

    diameter = np.median(diameters)

    crop_size_with_margin = diameter * margin_factor
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