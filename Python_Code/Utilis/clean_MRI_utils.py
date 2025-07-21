import sys
import numpy as np
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass

from Python_Code.Utilis.pytorch_segmentation_utils import (
    produce_segmentation_at_required_resolution, simple_shape_correction
)
from Python_Code.Utilis.visualizeDICOM import planeToXYZ

def clean_slices_base(dicom_series, percentage = 0.3):
    """
    Remove z-slices that are above the valve plane based on valve plane detection.
    
    This function identifies and removes slices that are consistently above the valve plane
    across time frames. It uses a threshold-based approach where if a slice is above the
    valve plane in at least 30% of time frames, it's considered for removal.
    
    Args:
        dicom_series: DICOM series object containing:
            - slice_above_valveplane: 2D array (time_frames x z_stacks) indicating 
              which slices are above valve plane (0 = above, 1 = below)
            - prepped_seg: 4D segmentation array (time x z height x height x width)
            - prepped_data: 4D image data array (time x z height x height x width)
            - cleaned_data: 4D image data array (time x z height x height x width)
            - slices: number of z-slices
            - XYZs: list of spatial coordinates for each slice

        percentage: percentage of time-frames used as threshold (default 0.3 = 30%)
    Returns:
        list: Indices of z-slices that were removed
    
    Modifies:
        dicom_series.prepped_seg: Updated with slices removed
        dicom_series.prepped_data: Updated with slices removed  
        dicom_series.slices: Updated slice count
        dicom_series.XYZs: Updated coordinate list
    """
    number_time_frames = dicom_series.slice_above_valveplane.shape[0]
    number_z_stacks = dicom_series.slice_above_valveplane.shape[1]
    
    # Use 30% of time frames as threshold for determining if slice should be removed
    threshold = int(percentage * number_time_frames)
    z_height_remove = []

    # Check each z-slice across all time frames
    for row in range(number_z_stacks):
        number_above_base = 0
        
        # Count how many time frames this slice is above the valve plane
        for col in range(number_time_frames):
            if dicom_series.slice_above_valveplane[col, row] == 0:  # 0 indicates above valve plane
                number_above_base += 1

            # If this slice is above valve plane in enough time frames, mark for removal
            if number_above_base >= threshold:
                z_height_remove.append(row)
                break
    
    #this is hard because I do not
    # Handle edge case: add missing top layers if lower neighbors are already marked for removal
    # This ensures we don't leave isolated slices at the top or the bottom, depending on how the MRI orientation

    z_height_remove = set(z_height_remove)

    # Top padding logic 
    if (number_z_stacks - 3 in z_height_remove and 
        number_z_stacks - 2 not in z_height_remove and 
        number_z_stacks - 1 not in z_height_remove):
        z_height_remove.update([number_z_stacks - 2, number_z_stacks - 1])
    elif (number_z_stacks - 2 in z_height_remove and 
        number_z_stacks - 1 not in z_height_remove):
        z_height_remove.add(number_z_stacks - 1)

    # Bottom padding logic
    if (0 in z_height_remove and 
        1 not in z_height_remove and 
        2 not in z_height_remove):
        z_height_remove.update([1, 2])
    elif (0 in z_height_remove and 
        1 not in z_height_remove):
        z_height_remove.add(1)

    # Convert back to sorted list if needed
    z_height_remove = sorted(z_height_remove)
        
    # Remove identified slices from all data structures
    dicom_series.prepped_seg = np.delete(dicom_series.prepped_seg, z_height_remove, axis=1)
    dicom_series.prepped_data = np.delete(dicom_series.prepped_data, z_height_remove, axis=1)
    dicom_series.cleaned_data = np.delete(dicom_series.cleaned_data, z_height_remove, axis=1)
    dicom_series.slices = dicom_series.slices - len(z_height_remove)

    dicom_series.image_positions = [item for i, item in enumerate(dicom_series.image_positions) if i not in z_height_remove]
    dicom_series.slice_locations = [item for i, item in enumerate(dicom_series.slice_locations) if i not in z_height_remove]

    print(dicom_series.prepped_data.shape)

    return z_height_remove


def estimateValvePlanePosition(dicom_exam):
    """
    Estimate valve plane position in SAX (Short Axis) slices using morphological analysis.
    
    The segmentation network may make myocardium predictions that extend above the valve plane
    (e.g., by segmenting atrium wall), which can cause LV mesh fitting issues. This function
    identifies slices/frames that are above the valve plane so their masks can be ignored
    during LV mesh fitting.
    
    Method:
        Uses morphological analysis to detect where LV myocardium transitions from a closed 
        circle to a C-shape, indicating the approach to the valve plane. The heuristic 
        measures how completely the myocardium surrounds the blood pool.
    
    Args:
        dicom_exam: List of DICOM series objects, each containing:
            - view: string indicating view type ('SAX' for short axis)
            - frames: number of time frames
            - slices: number of z-slices  
            - prepped_seg: 4D segmentation array where:
                - value 2 = myocardium
                - value 3 = blood pool
    
    Modifies:
        For each SAX series in dicom_exam:
            - VP_heuristic2: 2D array (frames x slices) with valve plane estimates
            - slice_above_valveplane: binary array indicating slices above valve plane
    
    Note:
        Currently only uses SAX slices. Could be enhanced with LAX (Long Axis) slice 
        comparison for more accurate valve plane detection.
    """
    for series in dicom_exam:
        if series.view == 'SAX':
            # Initialize valve plane heuristic array
            series.VP_heuristic2 = np.zeros((series.frames, series.slices))
            
            # Analyze each time frame and z-slice
            for time_frame in range(series.frames):
                for z_slice in range(series.slices):
                    # Extract myocardium and blood pool masks
                    myocardium_mask = series.prepped_seg[time_frame, z_slice] == 2
                    blood_pool_mask = series.prepped_seg[time_frame, z_slice] == 3
                    
                    # Create 1-pixel outer boundary of the blood pool using morphological dilation
                    blood_pool_boundary = binary_dilation(blood_pool_mask) * (1 - blood_pool_mask)
                    
                    # Calculate fraction of blood pool boundary that has myocardium
                    # If myocardium completely surrounds blood pool, this approaches 1.0
                    # Add small epsilon values to avoid division by zero
                    myocardium_coverage = ((np.sum(myocardium_mask * blood_pool_boundary) + 0.00001) / 
                                         (np.sum(blood_pool_boundary) + 0.00001))
                    
                    series.VP_heuristic2[time_frame, z_slice] = myocardium_coverage
            
            # Convert to binary: slices with >98% myocardium coverage are considered below valve plane
            # Use 0.98 threshold rather than 1.0 to allow for occasional missing pixels
            series.VP_heuristic2 = series.VP_heuristic2 > 0.98
            
            # Store binary result indicating slices above valve plane (inverted logic)
            series.slice_above_valveplane = series.VP_heuristic2.astype(int)


def clean_time_frames(dicom_series, slice_threshold = 2):
    """
    Remove time frames that have too many missing or empty slices.
    
    This function identifies and removes time frames where more than 2 z-slices 
    contain no image data (sum of pixel values = 0). Including such frames decreases the 
    accuracy with which the meshes can be fitted to the segmentation masks.
    
    Args:
        dicom_series: DICOM series object containing:
            - prepped_data: 4D image data array (time x z x height x width)
            - prepped_seg: 4D segmentation array (time x z x height x width)  
            - cleaned_data: 4D image data array (time x z height x height x width)
            - frames: number of time frames

        slice_threshold: threshold of missing slices in order for the time frame 
                        to be removed (default = 2)
    
    Returns:
        list: Indices of time frames that were removed
        
    Modifies:
        dicom_series.prepped_seg: Updated with incomplete frames removed
        dicom_series.prepped_data: Updated with incomplete frames removed
        dicom_series.frames: Updated frame count
    """
    incomplete_frames = []
    
    # Check each time frame for missing data
    for time_frame in range(dicom_series.prepped_data.shape[0]):
        missing_slices_count = 0
        
        # Count empty slices in this time frame
        for z_slice in range(dicom_series.prepped_data.shape[1]):
            # Check if slice contains any non-zero data
            if dicom_series.prepped_data[time_frame, z_slice, :, :].sum() == 0:
                missing_slices_count += 1

            # If more than 2 slices are missing, mark this time frame for removal
            if missing_slices_count > slice_threshold:
                incomplete_frames.append(time_frame)
                break
    
    # Remove incomplete time frames from both segmentation and image data
    dicom_series.prepped_seg = np.delete(dicom_series.prepped_seg, incomplete_frames, axis=0)
    dicom_series.prepped_data = np.delete(dicom_series.prepped_data, incomplete_frames, axis=0)
    dicom_series.cleaned_data = np.delete(dicom_series.cleaned_data, incomplete_frames, axis=0)
    dicom_series.frames = dicom_series.frames - len(incomplete_frames)

    keep_indices = [i for i in range(dicom_series.prepped_data.shape[0]) if i not in incomplete_frames]
    dicom_series.image_ids = dicom_series.image_ids[keep_indices]


    return incomplete_frames


def clean_slices_apex(dicom_series, percentage = 0.2):
    """
    Remove z-slices at the apex that lack sufficient LV or blood pool segmentation.
    
    This function removes z-slices where either the LV myocardium or blood pool 
    segmentation is missing in more than 20% of time frames. Such slices are 
    typically at the apex where segmentation becomes unreliable or where 
    anatomical structures are no longer present.
    
    Args:
        dicom_series: DICOM series object containing:
            - prepped_seg: 4D segmentation array (time x z height x height x width) where:
                - value 2 = LV myocardium  
                - value 3 = blood pool
            - prepped_data: 4D image data array (time x z height x height x width)
            - cleaned_data: 4D image data array (time x z height x height x width)
            - slices: number of z-slices
            - XYZs: list of spatial coordinates for each slice

        percentage: threshold for number of missing segmentation in order for z slice 
                    to be removed (default 0.2 = 20%)
    
    Returns:
        list: Indices of z-slices that were removed
        
    Modifies:
        dicom_series.prepped_seg: Updated with problematic slices removed
        dicom_series.prepped_data: Updated with problematic slices removed
        dicom_series.slices: Updated slice count  
        dicom_series.XYZs: Updated coordinate list
    """
    number_time_frames = dicom_series.prepped_seg.shape[0]
    
    # Use percentage (default 20%) of time frames as threshold for determining problematic slices
    threshold = int(percentage * number_time_frames)
    slices_to_remove = []
    
    # Check each z-slice across all time frames
    for z_slice in range(dicom_series.prepped_seg.shape[1]):
        missing_lv_count = 0
        missing_blood_pool_count = 0
        
        # Count missing segmentations across time frames for this slice
        for time_frame in range(number_time_frames):
            # Extract LV myocardium and blood pool masks
            lv_mask = (dicom_series.prepped_seg[time_frame, z_slice, :, :] == 2).astype(np.uint8)
            blood_pool_mask = (dicom_series.prepped_seg[time_frame, z_slice, :, :] == 3).astype(np.uint8)

            # Count frames where LV myocardium segmentation is absent
            if lv_mask.sum() == 0:
                missing_lv_count += 1
            
            # Count frames where blood pool segmentation is absent  
            if blood_pool_mask.sum() == 0:
                missing_blood_pool_count += 1

            # If either structure is missing in too many frames, mark slice for removal
            if missing_lv_count >= threshold or missing_blood_pool_count >= threshold:
                slices_to_remove.append(z_slice)
                break

    # Remove problematic slices from all data structures
    dicom_series.prepped_seg = np.delete(dicom_series.prepped_seg, slices_to_remove, axis=1)
    dicom_series.prepped_data = np.delete(dicom_series.prepped_data, slices_to_remove, axis=1)
    dicom_series.cleaned_data = np.delete(dicom_series.cleaned_data, slices_to_remove, axis=1)
    dicom_series.slices = dicom_series.slices - len(slices_to_remove)

    dicom_series.image_positions = [item for i, item in enumerate(dicom_series.image_positions) if i not in slices_to_remove]
    dicom_series.slice_locations = [item for i, item in enumerate(dicom_series.slice_locations) if i not in slices_to_remove]
    
    return slices_to_remove


def postprocess_cleaned_data(dicom_exam, dicom_series: list) -> None:
    """
    Post-process cleaned DICOM data for each series:
    - Segments the cleaned data at the required resolution.
    - Computes crop centers and applies cropping.
    - Generates 3D world coordinates (X, Y, Z) for each pixel in each slice.
    - Prepares segmented data and mask in the expected format.
    - Applies shape correction for short-axis (SAX) views.

    Parameters:
    -----------
    dicom_series : list 
        A list of DICOM series objects, each expected to have attributes like
        `cleaned_data`, `pixel_spacing`, `view`, `image_positions`, and `orientation`.
    """

    for series_idx, series in enumerate(dicom_exam.series):
        print(series.cleaned_data.shape)

        is_sax = series.view in ['SAX', 'unknown']
        crop_size = dicom_exam.sz

        # Perform segmentation and determine cropping center
        segmented_data, segmentation_mask, center_x, center_y = produce_segmentation_at_required_resolution(
            series.cleaned_data, 
            series.pixel_spacing, 
            is_sax
        )

        # Compute max offset for valid cropping range
        max_offset = min(
            segmentation_mask.shape[2] - crop_size, 
            segmentation_mask.shape[3] - crop_size
        )

        # Clip center coordinates to keep crop within bounds
        center_x = np.clip(center_x - crop_size // 2, 0, max_offset)
        center_y = np.clip(center_y - crop_size // 2, 0, max_offset)
        series.c1, series.c2 = center_x, center_y

        # Generate 3D world coordinates for each slice in the series
        series.XYZs = []
        for slice_idx in range(series.slices):
            height, width = series.cleaned_data.shape[-2:]  # Dynamic dimensions

            X, Y, Z = planeToXYZ(
                (height, width),
                series.image_positions[slice_idx],
                series.orientation,
                [1, 1]  # Assuming isotropic in-plane resolution; adjust if needed
            )

            # Crop the coordinate grids
            X_cropped = X[center_y:center_y + crop_size, center_x:center_x + crop_size]
            Y_cropped = Y[center_y:center_y + crop_size, center_x:center_x + crop_size]
            Z_cropped = Z[center_y:center_y + crop_size, center_x:center_x + crop_size]

            # Stack cropped coordinates and flatten to (N, 3)
            xyz_coords = np.stack([X_cropped.ravel(), Y_cropped.ravel(), Z_cropped.ravel()], axis=1)
            series.XYZs.append(xyz_coords)

        # Define slices for cropping
        crop_slice_x = slice(center_x, center_x + crop_size)
        crop_slice_y = slice(center_y, center_y + crop_size)

        # Crop and transpose data to shape (time, slice, y, x)
        series.prepped_seg = np.transpose(
            segmentation_mask[:, :, crop_slice_x, crop_slice_y], 
            (0, 1, 3, 2)
        )
        series.prepped_data = np.transpose(
            segmented_data[:, :, crop_slice_x, crop_slice_y], 
            (0, 1, 3, 2)
        )

        # Optional shape correction for short-axis view
        if is_sax:
            print("  Applying shape correction for SAX view")
            series.prepped_seg = simple_shape_correction(series.prepped_seg)


def estimate_MRI_orientation(dicom_exam):
    slice_diameter = []
    for i in range(dicom_exam.series[0].prepped_seg.shape[1]):
        slice = dicom_exam.series[0].prepped_seg[0,i,:,:]
        myo_mask = np.where(slice == 2, 1, 0)

        center_y, center_x = center_of_mass(myo_mask)
        cy, cx = int(np.round(center_y)), int(np.round(center_x))

        diameters = []

        # Horizontal diameter
        h_pixels = np.where(myo_mask[cy, :] == 1)[0]
        if len(h_pixels) > 0:
            diameters.append(h_pixels[-1] - h_pixels[0])
        
        # Vertical diameter  
        v_pixels = np.where(myo_mask[:, cx] == 1)[0]
        if len(v_pixels) > 0:
            diameters.append(v_pixels[-1] - v_pixels[0])
        
        # Diagonal diameters
        d1_pixels = np.where(np.diag(myo_mask) == 1)[0]
        if len(d1_pixels) > 0:
            diameters.append((d1_pixels[-1] - d1_pixels[0]) * np.sqrt(2))
        
        d2_pixels = np.where(np.diag(np.rot90(myo_mask)) == 1)[0]
        if len(d2_pixels) > 0:
            diameters.append((d2_pixels[-1] - d2_pixels[0]) * np.sqrt(2))
        
        slice_diameter.append(np.mean(diameters))
    

    n = dicom_exam.series[0].prepped_seg.shape[1]
    mid = n // 2
    start_avg = sum(slice_diameter[:mid]) / mid
    end_avg = sum(slice_diameter[mid:]) / (n - mid)

    if start_avg > end_avg:
        dicom_exam.MRI_orientation = "base_top"
    elif start_avg < end_avg:
        dicom_exam.MRI_orientation = "apex_top"
