"""
Cardiac Parameter Analysis Module

This module provides functions for computing cardiac parameters from DICOM cardiac MRI data,
including left ventricular blood pool volume, myocardium volume and thickness calculation.
"""

import os
import sys
import shutil
from typing import Tuple, Optional, List
import ast

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend suitable for script and threads
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import pyvista as pv
from scipy.ndimage.measurements import center_of_mass
from skimage.morphology import skeletonize
from matplotlib.colors import LinearSegmentedColormap

from Python_Code.Utilis.volume_cal_functions import cal_bp_volume
from Python_Code.Utilis.thickness_functions import (
    seg_mask_plot_thickness_map,
    translate_mesh_to_origin,
    cartesian_to_cylindrical,
    find_ring_slices,
    compute_thickness_map,
    meshes_plot_thickness_map
)



def compute_cardiac_parameters(dicom_exam, mask_type: str = 'seg') -> None:
    """
    Compute various cardiac parameters using voxelized masks.

    Parameters:
    -----------
    dicom_exam : DicomExam
        DICOM examination object containing cardiac MRI data
    mask_type : str, default 'seg'
        Type of mask to use:
        - 'seg': Use segmentation masks
        - 'mesh': Use masks extracted from fitted mesh

    Returns:
    --------
    None
        Results are saved to CSV files and plots are generated

    Notes:
    ------
    Computes the following cardiac parameters:
    - LV blood pool volume over time
    - LV myocardium volume, thickness, radius, and length over time
    - Ejection fraction (EF), stroke volume (SV)
    - End-diastolic volume (EDV), end-systolic volume (ESV)
    """
    if dicom_exam.time_frames == 1:
        print('Function requires exams with more than 1 time frame')
        return


    for series in dicom_exam:

        # Initialize measurement containers
        measurements = {
            'myo_volumes': [],
            'bp_volumes': [],
            'lengths': []
        }

        if series.name in dicom_exam.series_to_exclude:
            continue
        masks, output_folders = _prepare_masks_and_folders(
            series, dicom_exam, mask_type
        )


        if series.view == 'SAX':
            _process_sax_series(series, masks, measurements)
        else:  # LAX view
            continue

        # Generate outputs
        _generate_outputs(measurements, series.view, output_folders, mask_type, series.name)


def _prepare_masks_and_folders(series, dicom_exam, mask_type: str) -> Tuple:
    """Prepare masks and output folders based on mask type."""
    if mask_type == 'mesh':
        # Combine multi-channel class masks into single-channel with integer labels
        masks = np.round(series.mesh_seg)

        masks = np.sum(masks * [[[[[1, 2, 3]]]]], axis=-1)
        output_folders = {
            'plots': dicom_exam.folder['mesh_plots'],
            'analysis': dicom_exam.folder['seg_output_mesh']
        }
    else:
        masks = series.prepped_seg
        output_folders = {
            'plots': dicom_exam.folder['seg_plots'],
            'analysis': dicom_exam.folder['seg_output_seg']
        }

    # Create output directories
    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)

    return masks, output_folders


def _process_sax_series(series, masks, measurements) -> None:
    """Process short-axis (SAX) series data."""

    measurements['bp_volumes'].append([])
    measurements['myo_volumes'].append([])

    pixel_volume = _calculate_pixel_volume(series)
    print(pixel_volume)

    for time_frame in range(series.frames):
        # Calculate volumes
        myo_vol, bp_vol = _calculate_volumes_sax(
            masks, time_frame, series.slices, pixel_volume
        )

        measurements['myo_volumes'][-1].append(myo_vol)
        measurements['bp_volumes'][-1].append(bp_vol)

    # Average across all series
    for key in ['myo_volumes', 'bp_volumes']:
        if measurements[key]:
            measurements[key] = np.nanmean(np.array(measurements[key]), axis=0)



def _calculate_volumes_sax(masks, time_frame: int, num_slices: int, 
                          pixel_volume: float) -> Tuple[float, float]:
    """Calculate myocardium and blood pool volumes for SAX view."""
    myo_vol = 0
    bp_vol = 0

    for slice_idx in range(num_slices):
        # Apply pixel_volume directly to the mask (as in original code)
        myo = (masks[time_frame, slice_idx] == 2).astype(int) * pixel_volume
        bp = (masks[time_frame, slice_idx] == 3).astype(int) * pixel_volume

        myo_vol += np.sum(myo)
        bp_vol += np.sum(bp)

    return myo_vol, bp_vol


def _calculate_pixel_volume(series) -> float:
    """Calculate volume of a single pixel in ml."""
    return (series.pixel_spacing[0] * 
            series.pixel_spacing[1] * 
            series.pixel_spacing[2] * 1e-3)


def _generate_outputs(measurements, view: str, output_folders: dict, 
                     mask_type: str, series_name) -> None:
    """Generate CSV outputs and plots from measurements."""
    if view == 'SAX':
        df = pd.DataFrame({
            'myo_volume': measurements['myo_volumes'],
            'bp_volume': measurements['bp_volumes']
        })
        titles = [
            "Volume of Myocardium [ml]",
            "Volume of Blood Pool [ml]"
        ]
    
    filename = f'{series_name}_simpson_volumes.csv'

    # Generate plots
    for i, col in enumerate(df.columns):
        plot_time_series(
            df, 
            file_name=col,
            output_folder=output_folders['plots'],
            series_name=series_name,
            title=titles[i],
            y_value=col,
            y_label=titles[i],
        )

    # Save measurements to CSV
    df.to_csv(os.path.join(output_folders['analysis'], filename), index=False)

    # Calculate and save cardiac parameters
    cardiac_params = _calculate_cardiac_parameters(measurements['bp_volumes'])
    param_df = pd.DataFrame([
        {'Parameter': 'EDV', 'Value': cardiac_params['EDV'], 
         'Time_step': cardiac_params['EDV_time']},
        {'Parameter': 'ESV', 'Value': cardiac_params['ESV'], 
         'Time_step': cardiac_params['ESV_time']},
        {'Parameter': 'EF', 'Value': cardiac_params['EF'], 'Time_step': None},
        {'Parameter': 'SV', 'Value': cardiac_params['SV'], 'Time_step': None}
    ])
    param_df.to_csv(
        os.path.join(output_folders['analysis'], f'{series_name}_ED_ES_state.csv'), 
        index=False
    )


def _calculate_cardiac_parameters(bp_volumes: np.ndarray) -> dict:
    """Calculate cardiac parameters from blood pool volumes."""
    edv = np.max(bp_volumes)
    edv_time = np.argmax(bp_volumes)
    esv = np.min(bp_volumes)
    esv_time = np.argmin(bp_volumes)
    
    return {
        'EDV': edv,
        'EDV_time': edv_time,
        'ESV': esv,
        'ESV_time': esv_time,
        'EF': (edv - esv) / edv * 100,
        'SV': edv - esv
    }


def plot_time_series(data: pd.DataFrame, y_value: str, file_name: str, 
                    output_folder: str, series_name = None, title: str = "Time Series",
                    y_label: str = "Value", x_label: str = "Time frame",
                    line_color: str = "#3498db", scatter_color: str = "#ec6564",
                    alpha: float = 0.7) -> None:
    """
    Plot time series with both line and scatter points.

    Parameters:
    -----------
    data : pd.DataFrame
        Data containing the time series
    y_value : str
        Column name for y-axis values
    file_name : str  
        Base name for output files
    output_folder : str
        Directory to save plots
    title : str
        Plot title
    y_label : str
        Y-axis label
    x_label : str
        X-axis label
    line_color : str
        Color for the line plot
    scatter_color : str
        Color for scatter points
    alpha : float
        Transparency level
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot line and scatter
    sns.lineplot(
        data=data, x=data.index, y=y_value,
        color=line_color, linewidth=2, alpha=alpha, label="Trend"
    )
    
    sns.scatterplot(
        data=data, x=data.index, y=y_value,
        color=scatter_color, s=50, alpha=alpha, label="Data Points"
    )
    
    # Formatting
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()

    if series_name:
        filename =f"{series_name}_{file_name}"
    else:
        filename = file_name

    # Save plots
    plt.savefig(
        os.path.join(output_folder, f'{filename}.pdf'), 
        dpi=300, bbox_inches='tight'
    )
    plt.savefig(
        os.path.join(output_folder, f'{filename}.svg'), 
        format='svg'
    )
    plt.close(fig)


def estimate_thickness_and_radius(myo_mask: np.ndarray) -> Tuple[float, float]:
    """
    Estimate myocardium thickness and radius from binary mask.
    
    Assumes the myocardium is approximately circular and calculates:
    - Average myocardium thickness
    - Average myocardium radius
    
    Parameters:
    -----------
    myo_mask : np.ndarray
        Binary myocardium mask with shape (H, W)
        
    Returns:
    --------
    Tuple[float, float]
        (thickness, radius) in pixels
        
    Notes:
    ------
    This simple estimate may produce poor results for C-shaped masks.
    """
    if np.sum(myo_mask) == 0:
        return 0.0, 0.0

    # Get center of mass
    center_y, center_x = center_of_mass(myo_mask)
    cy, cx = int(np.round(center_y)), int(np.round(center_x))

    # Calculate thickness from intersections through center
    horizontal_thickness = np.sum(myo_mask[cy, :])
    vertical_thickness = np.sum(myo_mask[:, cx])
    
    # Diagonal intersections (scaled by sqrt(2) for pixel spacing)
    diag1_thickness = np.sum(np.diag(myo_mask)) * np.sqrt(2)
    diag2_thickness = np.sum(np.diag(np.rot90(myo_mask))) * np.sqrt(2)
    
    thickness = (horizontal_thickness + vertical_thickness + 
                diag1_thickness + diag2_thickness) / 8

    # Calculate radius from diameters
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

    radius = np.mean(diameters) / 2 if diameters else 0.0

    return thickness, radius



def analyze_mesh_volumes(dicom_exam) -> None:
    """
    Analyze volumes from VTK mesh files and identify ED/ES states.
    
    Parameters:
    -----------
    dicom_exam : DicomExam
        DICOM examination object
    """
    meshes_dir = dicom_exam.folder['meshes']
    os.makedirs(meshes_dir, exist_ok=True)

    # Initialize tracking variables
    volumes_data = []
    max_bp_volume = float('-inf')
    min_bp_volume = float('inf')
    max_bp_file = None
    min_bp_file = None

    # Process all VTK files
    for filename in os.listdir(meshes_dir):
        if not filename.lower().endswith('.vtk'):
            continue
            
        time_frame = int(filename.split('=')[1].split('.')[0])
        vtk_path = os.path.join(meshes_dir, filename)
        
        # Calculate volumes
        bp_volume, myo_volume = cal_bp_volume(vtk_path)
        bp_volume_ml = bp_volume * 1e-3
        myo_volume_ml = myo_volume * 1e-3
        
        volumes_data.append({
            'bp_volume': bp_volume_ml,
            'myo_volume': myo_volume_ml,
            'Time': time_frame
        })
        
        # Track extremes for ED/ES identification
        if bp_volume > max_bp_volume:
            max_bp_volume = bp_volume
            max_bp_file = filename
        if bp_volume < min_bp_volume:
            min_bp_volume = bp_volume
            min_bp_file = filename

    # Create and save results
    df_volumes = pd.DataFrame(volumes_data).sort_values('Time').reset_index(drop=True)
    
    df_ed_es = pd.DataFrame([
        {'state': 'ED', 'volume': max_bp_volume * 1e-3, 'time_frame': max_bp_file},
        {'state': 'ES', 'volume': min_bp_volume * 1e-3, 'time_frame': min_bp_file},
        {'state': 'SV', 'volume': (max_bp_volume - min_bp_volume) * 1e-3,'time_frame': ''},
        {'state': 'EF', 'volume': (max_bp_volume - min_bp_volume)/max_bp_volume,'time_frame': ''}
    ])

    # Copy ED/ES mesh files to output directory
    destination_dir = dicom_exam.folder['meshes_output']
    os.makedirs(destination_dir, exist_ok=True)
    
    shutil.copy2(
        os.path.join(meshes_dir, max_bp_file),
        os.path.join(destination_dir, 'ED_state.vtk')
    )
    shutil.copy2(
        os.path.join(meshes_dir, min_bp_file),
        os.path.join(destination_dir, 'ES_state.vtk')
    )

    # Save results
    df_ed_es.to_csv(os.path.join(destination_dir, 'ED_ES_states.csv'), index=False)
    df_volumes.to_csv(os.path.join(destination_dir, 'mesh_volumes.csv'), index=False)

    # Generate plots
    _plot_mesh_volumes(df_volumes, dicom_exam)

    print(f"ED state: {max_bp_file} (Volume: {max_bp_volume * 1e-3:.2f} ml)")
    print(f"ES state: {min_bp_file} (Volume: {min_bp_volume * 1e-3:.2f} ml)")


def _plot_mesh_volumes(df_volumes: pd.DataFrame, dicom_exam) -> None:
    """Generate volume plots for mesh analysis."""
    output_folder = dicom_exam.folder['mesh_vol_plots']
    os.makedirs(output_folder, exist_ok=True)

    titles = ["Volume of Blood Pool [ml]", "Volume of Myocardium [ml]"]
    
    for i, col in enumerate(['bp_volume', 'myo_volume']):
        plot_time_series(
            df_volumes,
            file_name=col,
            output_folder=output_folder,
            title=titles[i],
            y_value=col,
            y_label=titles[i]
        )


def seg_masks_compute_thickness_map(
    dicom_exam, 
    n_theta_bins: int = 36, 
    label_of_interest: int = 2
) -> None:
    """
    Compute thickness maps from segmentation masks using polar coordinate analysis.
    
    This function analyzes cardiac segmentation masks to compute thickness measurements
    across different angular bins (theta) and axial slices (z). The thickness is 
    calculated as the radial distance between the innermost and outermost points
    of the segmented region in each angular bin.
    
    Args:
        dicom_exam: DICOM examination object containing segmentation data and metadata
        n_theta_bins (int, optional): Number of angular bins for polar analysis. 
                                    Defaults to 36 (10° per bin).
        label_of_interest (int, optional): Segmentation label value to analyze. 
                                         Defaults to myocardium 2.
    
    Returns:
        None: Function saves thickness maps to disk and creates visualizations

    """

    for series in dicom_exam.series:

        if series.view == 'SAX':
            print(series.name)
            print(series.view)
                
            output_folder = dicom_exam.folder['seg_thickness']
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            for time in range(dicom_exam.time_frames):
                seg_stack = dicom_exam.series[0].prepped_seg[time, :, :, :]
                Z, H, W = seg_stack.shape
                
                # Initialize thickness map with NaN values
                thickness_map = np.full((Z, n_theta_bins), np.nan)
                
                # Create azimuthal bins from 0 to 2π
                theta_bins = np.linspace(0, 2 * np.pi, n_theta_bins + 1)

                
                # Calculate center from basal or apical plane based on MRI orientation
                cx, cy = _calculate_centroid_from_reference_slice(
                    dicom_exam, seg_stack, label_of_interest
                )
                
                # Process each axial slice
                for z in range(Z):
                    seg_slice = seg_stack[z]
                    
                    # Extract points of interest from segmentation
                    ys, xs = np.where(seg_slice == label_of_interest)
                    
                    if len(xs) == 0:
                        continue  # Skip slices with no segmented regions
                    
                    # Apply pixel spacing to convert to physical coordinates
                    xs_scaled = xs * dicom_exam.series[0].pixel_spacing[1]
                    ys_scaled = ys * dicom_exam.series[0].pixel_spacing[2]
                    
                    # Convert to polar coordinates relative to centroid
                    x_shifted = xs_scaled - cx
                    y_shifted = ys_scaled - cy
                    
                    r = np.sqrt(x_shifted**2 + y_shifted**2)
                    theta = np.arctan2(y_shifted, x_shifted)  # Range: [-π, π]
                    
                    # Normalize theta to [0, 2π] range
                    theta = (theta + 2 * np.pi) % (2 * np.pi)
                    
                    # Assign theta values to discrete bins
                    theta_idx = np.digitize(theta, theta_bins) - 1  # Convert to zero-based indexing
                    
                    # Calculate thickness for each angular bin
                    for ti in range(n_theta_bins):
                        mask = (theta_idx == ti)
                        if not np.any(mask):
                            continue
                        
                        r_values = r[mask]
                        thickness = r_values.max() - r_values.min()
                        thickness_map[z, ti] = thickness
                
                # Save and visualize results
                seg_mask_plot_thickness_map(thickness_map, time, output_folder)
                np.save(os.path.join(output_folder, f"thickness_map_{time}"), thickness_map)


def _calculate_centroid_from_reference_slice(
    dicom_exam, 
    seg_stack: np.ndarray, 
    label_of_interest: int
) -> Tuple[float, float]:
    """
    Calculate centroid coordinates from reference slice based on MRI orientation.
    
    Args:
        dicom_exam: DICOM examination object with orientation information
        seg_stack (np.ndarray): 3D segmentation array (Z, H, W)
        label_of_interest (int): Segmentation label to analyze
        
    Returns:
        Tuple[float, float]: Centroid coordinates (cx, cy) in physical units
    """

    if dicom_exam.MRI_orientation == "base_top":
        reference_slice = seg_stack[0]  # Basal slice
    elif dicom_exam.MRI_orientation == "apex_top":
        reference_slice = seg_stack[-1]  # Apical slice
    else:
        raise ValueError(f"Unknown MRI orientation: {dicom_exam.MRI_orientation}")
    
    # Extract points of interest
    ys, xs = np.where(reference_slice == label_of_interest)
    
    if len(xs) == 0:
        raise ValueError(f"No pixels found with label {label_of_interest} in reference slice")
    
    # Apply pixel spacing and calculate centroid
    xs_scaled = xs * dicom_exam.series[0].pixel_spacing[1]
    ys_scaled = ys * dicom_exam.series[0].pixel_spacing[2]
    
    cx, cy = xs_scaled.mean(), ys_scaled.mean()
    return cx, cy


def meshes_compute_thickness_map(dicom_exam) -> None:
    """
    Compute thickness maps from 3D mesh data using cylindrical coordinate analysis.
    
    This function processes VTK mesh files to generate thickness measurements
    in cylindrical coordinates. The mesh is first translated to origin, converted
    to cylindrical coordinates, and then analyzed in ring-shaped z-slices to
    compute radial thickness variations.
    
    Args:
        dicom_exam: DICOM examination object containing mesh folder paths and 
                   time frame information
    
    Returns:
        None: Function saves thickness maps and filtered z-coordinates to disk
    
    """
    mesh_folder = dicom_exam.folder['meshes']
    output_folder = dicom_exam.folder['mesh_thickness']
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for time_step in dicom_exam.time_frames_to_fit:
        # Load mesh data
        mesh_filename = os.path.join(mesh_folder, f"mesh_t={time_step}.vtk")
        
        if not os.path.exists(mesh_filename):
            print(f"Warning: Mesh file not found: {mesh_filename}")
            continue
            
        mesh = pv.read(mesh_filename)
        points = mesh.points
        
        # Preprocess mesh: translate to origin
        translate_mesh_to_origin(mesh, points, threshold=0)
        
        # Convert Cartesian coordinates to cylindrical
        r, theta, z = cartesian_to_cylindrical(points)
        
        # Identify ring-shaped z-slices for analysis
        ring_slices, z_bins = find_ring_slices(points, n_z_bins=15)
    
        
        # Compute thickness measurements
        thickness_map, filtered_z_coords = compute_thickness_map(
            r, theta, z, ring_slices, z_bins
        )
        
        # Save and visualize results
        meshes_plot_thickness_map(thickness_map, filtered_z_coords, time_step, output_folder)
        
        # Save numerical results
        np.save(os.path.join(output_folder, f"thickness_map_{time_step}"), thickness_map)
        np.save(os.path.join(output_folder, f"filtered_z_coords_{time_step}"), filtered_z_coords)