"""
Cardiac Parameter Analysis Module

This module provides functions for computing cardiac parameters from DICOM cardiac MRI data,
including left ventricular blood pool volume, myocardium measurements, and uncertainty analysis.
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
    plot_ring_slices,
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

    # Initialize measurement containers
    measurements = {
        'myo_volumes': [],
        'bp_volumes': [],
        'lengths': []
    }

    for series in dicom_exam:
        if series.name in dicom_exam.series_to_exclude:
            continue
        masks, output_folders = _prepare_masks_and_folders(
            series, dicom_exam, mask_type
        )

        if series.view == 'SAX':
            _process_sax_series(series, masks, measurements)
        else:  # LAX view
            _process_lax_series(series, masks, measurements, dicom_exam.time_frames)

        # Generate outputs
        _generate_outputs(measurements, series.view, output_folders, mask_type)


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


def _process_lax_series(series, masks, measurements, time_frames: int) -> None:
    """Process long-axis (LAX) series data."""
    measurements['myo_volumes'].append([])
    measurements['bp_volumes'].append([])
    measurements['lengths'].append([])

    central_slice = series.slices // 2
    pixel_volume = _calculate_pixel_volume(series)

    for time_frame in range(time_frames):
        # LAX view only uses central slice for length calculation
        myo = (masks[time_frame, central_slice] == 2).astype(int) * pixel_volume
        bp = (masks[time_frame, central_slice] == 3).astype(int) * pixel_volume

        # Calculate volumes
        myo_vol = np.sum(myo)
        bp_vol = np.sum(bp)

        measurements['myo_volumes'][-1].append(myo_vol if myo_vol != 0 else np.nan)
        measurements['bp_volumes'][-1].append(bp_vol if bp_vol != 0 else np.nan)

        # Calculate length using skeletonization
        myo_mask = (masks[time_frame, central_slice] == 2).astype(int)
        myo_length = np.sum(skeletonize(myo_mask))* series.pixel_spacing[2]
        measurements['lengths'][-1].append(myo_length if myo_length != 0 else np.nan)

    # Average across all series
    for key in ['myo_volumes', 'bp_volumes', 'lengths']:
        if measurements[key]:
            measurements[key] = np.nanmean(np.array(measurements[key]), axis=0)


def _measure_thickness_radius(masks, time_frame: int, central_slice: int, 
                             series) -> Tuple[List[float], List[float]]:
    """Measure myocardium thickness and radius for a given time frame."""
    slice_options = [-1, 0, 1] if masks.shape[1] >= 3 else [0]
    thickness_values, radius_values = [], []

    for slice_offset in slice_options:
        myo_mask = (masks[time_frame, central_slice + slice_offset] == 2).astype(int)
        thickness_px, radius_px = estimate_thickness_and_radius(myo_mask)
        
        # Convert from pixels to mm
        thickness_mm = thickness_px * series.pixel_spacing[2]
        radius_mm = radius_px * series.pixel_spacing[2]
        
        thickness_values.append(thickness_mm)
        radius_values.append(radius_mm)

    return thickness_values, radius_values


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
                     mask_type: str) -> None:
    """Generate CSV outputs and plots from measurements."""
    if view == 'SAX':
        df = pd.DataFrame({
            'myo_volumes': measurements['myo_volumes'],
            'bp_volume': measurements['bp_volumes']
        })
        titles = [
            "Volume of Myocardium [ml]",
            "Volume of Blood Pool [ml]"
        ]
    else:  # LAX
        df = pd.DataFrame({
            'bp_volume': measurements['bp_volumes'],
            'myo_volumes': measurements['myo_volumes'],
            'myo_length': measurements['lengths']
        })
        titles = [
            "Volume of Blood Pool [ml]",
            "Volume of Myocardium [ml]",
            "Length of Myocardium [mm]"
        ]
    
    filename = 'seg_volumes_cal.csv'

    # Generate plots
    for i, col in enumerate(df.columns):
        plot_time_series(
            df, 
            file_name=col,
            output_folder=output_folders['plots'],
            title=titles[i],
            y_value=col,
            y_label=titles[i]
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
        os.path.join(output_folders['analysis'], 'ED_ES_state.csv'), 
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
                    output_folder: str, title: str = "Time Series",
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

    # Save plots
    plt.savefig(
        os.path.join(output_folder, f'{file_name}.pdf'), 
        dpi=300, bbox_inches='tight'
    )
    plt.savefig(
        os.path.join(output_folder, f'{file_name}.svg'), 
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


def calculate_segmentation_uncertainty(dicom_exam, masks_type: str = 'mesh') -> None:
    """
    Calculate and visualize segmentation uncertainty from mesh predictions.
    
    Parameters:
    -----------
    dicom_exam : DicomExam
        DICOM examination object
    masks_type : str
        Type of masks to analyze (currently only 'mesh' supported)
    """
    if dicom_exam[0].mesh_seg is None:
        print('Run fitMesh() before calculating uncertainty')
        return

    output_folder = dicom_exam.folder['mesh_seg_uncertainty']
    os.makedirs(output_folder, exist_ok=True)

    for series in dicom_exam:
        ap_mean = series.mesh_seg
        ap_std = series.mesh_seg_std

        # Calculate uncertainty: 1 - (total activation) / (number of active pixels)
        active_pixels = np.sum(ap_mean > 0, axis=(2, 3)) + 1e-6
        total_activation = np.sum(ap_mean, axis=(2, 3))
        uncertainty = 1 - total_activation / active_pixels

        # Save standard deviation image if available
        if ap_std is not None:
            std_img = np.concatenate(np.concatenate(ap_std, axis=2))
            std_img = ((std_img / 0.5) * 255).astype('uint8')
            imageio.imwrite(
                os.path.join(output_folder, 'std_img.png'), 
                std_img
            )

        # Save mean prediction image
        mean_img = np.concatenate(np.concatenate(ap_mean, axis=2))
        mean_img = (mean_img * 255).astype('uint8')
        imageio.imwrite(
            os.path.join(output_folder, 'mean_img.png'), 
            mean_img
        )

        # Plot uncertainty
        uncertainty_avg = np.mean(uncertainty[:, :, [1, 2]], axis=-1)
        plot_uncertainty(
            uncertainty_avg, 
            dicom_exam,
            os.path.join(output_folder, 'per_image_uncertainty.png')
        )


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
    df_volumes.to_csv(os.path.join(destination_dir, 'mesh_volumes_cal.csv'), index=False)

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


def plot_uncertainty(uncertainty_avg: np.ndarray, dicom_exam, 
                    filename: str) -> None:
    """
    Create uncertainty heatmap.
    
    Parameters:
    -----------
    uncertainty_avg : np.ndarray
        2D array of uncertainty values (time_frames x slices)
    dicom_exam : DicomExam
        DICOM examination object
    filename : str
        Output filename for the plot
    """
    plt.style.use('default')
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Create custom colormap
    colors = ['#440154', '#404387', '#2a788e', '#22a884', '#7ad151', '#fde725']
    cmap = LinearSegmentedColormap.from_list('uncertainty', colors, N=256)

    # Create heatmap
    im = ax.imshow(uncertainty_avg, cmap=cmap, aspect='auto', interpolation='bilinear')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Mean Uncertainty', fontsize=14, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    # Labels and title
    ax.set_xlabel('SAX Slice Index', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Time Frame', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(
        f'Spatiotemporal Uncertainty Distribution\nExam ID: {dicom_exam.id_string}',
        fontsize=16, fontweight='bold', pad=20
    )

    # Customize ticks
    n_frames, n_slices = uncertainty_avg.shape
    x_ticks = np.arange(0, n_slices, max(1, n_slices // 10))
    y_ticks = np.arange(0, n_frames, max(1, n_frames // 10))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Add grid and statistics
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    mean_uncert = np.mean(uncertainty_avg)
    std_uncert = np.std(uncertainty_avg)
    ax.text(
        0.02, 0.98, f'μ = {mean_uncert:.3f}\nσ = {std_uncert:.3f}',
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def seg_masks_compute_thickness_map(dicom_exam, n_theta_bins=36, label_of_interest=2):

    output_folder = dicom_exam.folder['seg_thickness']

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for time in range(dicom_exam.time_frames):
        seg_stack = dicom_exam.series[0].prepped_seg[time,:,:,:]
        Z, H, W = seg_stack.shape
        
        thickness_map = np.full((Z, n_theta_bins), np.nan)
        
        # Create azimuthal bins [0, 2pi]
        theta_bins = np.linspace(0, 2 * np.pi, n_theta_bins + 1)

        #calculate center from basal plane
        if dicom_exam.MRI_orientation == "base_top":
            seg_slice = seg_stack[0]
            
            # Extract points of label_of_interest
            ys, xs = np.where(seg_slice == label_of_interest)

            xs = xs* dicom_exam.series[0].pixel_spacing[1] #scale according to MRI resolution
            ys = ys* dicom_exam.series[0].pixel_spacing[2] #scale according to MRI resolution
            
            # Compute centroid (mean of points)
            cx, cy = xs.mean(), ys.mean()


        elif dicom_exam.MRI_orientation == "apex_top":
            seg_slice = seg_stack[-1]
            
            # Extract points of label_of_interest
            ys, xs = np.where(seg_slice == label_of_interest)

            xs = xs* dicom_exam.series[0].pixel_spacing[1] #scale according to MRI resolution
            ys = ys* dicom_exam.series[0].pixel_spacing[2] #scale according to MRI resolution
            
            # Compute centroid (mean of points)
            cx, cy = xs.mean(), ys.mean()

        
        for z in range(Z):
            seg_slice = seg_stack[z]
            
            # Extract points of label_of_interest
            ys, xs = np.where(seg_slice == label_of_interest)

            xs = xs* dicom_exam.series[0].pixel_spacing[1] #scale according to MRI resolution
            ys = ys* dicom_exam.series[0].pixel_spacing[2] #scale according to MRI resolution

            if len(xs) == 0:
                continue  # no object in this slice
            
            # Convert points to polar coordinates relative to centroid
            x_shifted = xs - cx
            y_shifted = ys - cy
            
            r = np.sqrt(x_shifted**2 + y_shifted**2)
            theta = np.arctan2(y_shifted, x_shifted)  # [-pi, pi]
            
            # Wrap theta to [0, 2pi]
            theta = (theta + 2 * np.pi) % (2 * np.pi)
            
            # Digitize theta into bins
            theta_idx = np.digitize(theta, theta_bins) - 1  # zero based
            
            for ti in range(n_theta_bins):
                mask = (theta_idx == ti)
                if not np.any(mask):
                    continue
                
                r_values = r[mask]
                thickness = r_values.max() - r_values.min()
                thickness_map[z, ti] = thickness   #10mm is the distance between slices in z direction
    
        seg_mask_plot_thickness_map(thickness_map, time, output_folder)
        np.save(os.path.join(output_folder, f"thickness_map_{time}"),thickness_map)
    return 


def meshes_compute_thickness_map(dicom_exam):

    mesh_folder = dicom_exam.folder['meshes']

    output_folder = dicom_exam.folder['mesh_thickness']

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for time_step in dicom_exam.time_frames_to_fit:

        mesh_filename = os.path.join(mesh_folder, "mesh_t=%d.vtk" % time_step) #these two meshes are exactly the same 

        mesh = pv.read(mesh_filename)
        points = mesh.points

        # Translate mesh to origin based on basal points threshold
        translate_mesh_to_origin(mesh, points, threshold=0)

        # Convert points to cylindrical coordinates
        r, theta, z = cartesian_to_cylindrical(points)

        # Find ring-shaped z slices
        ring_slices, z_bins = find_ring_slices(points, n_z_bins=15)

        # Plot the ring-shaped slices
        plot_ring_slices(points, ring_slices)

        # Compute thickness map and filtered z coords
        thickness_map, filtered_z_coords = compute_thickness_map(r, theta, z, ring_slices, z_bins)

        # Plot the thickness map
        meshes_plot_thickness_map(thickness_map, filtered_z_coords,time_step, output_folder)
        np.save(os.path.join(output_folder, f"thickness_map_{time_step}"),thickness_map)
        np.save(os.path.join(output_folder, f"filtered_z_coords_{time_step}"),filtered_z_coords)


def extract_auto_seg_compare_manu_seg(dicom_exam):
    df = pd.read_csv('manual_segmentation.csv', header=None)
    df[1] = df[1].apply(ast.literal_eval)
    df[0] = df[0].str.strip("'")

    manual_image_ids = df.loc[df[0] ==  dicom_exam.id_string, 1].values[0]

    # Create a boolean mask where True means the value is in the filter list
    mask = np.isin(dicom_exam.series[0].image_ids, manual_image_ids)

    # Get the indices (row, col) where matches occur
    indices = np.argwhere(mask)

    output_folder = dicom_exam.folder['seg_masks']

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for index in indices:
        seg_mask = dicom_exam.series[0].prepped_seg[index[0],index[1],:,:]
        MRI_data = dicom_exam.series[0].prepped_data[index[0],index[1],:,:]
        image_id = dicom_exam.series[0].image_ids[index[0],index[1]] 
        np.save(os.path.join(output_folder, f"seg_mask_{image_id}"), seg_mask)
        np.save(os.path.join(output_folder, f"mri_data_{image_id}"), MRI_data)