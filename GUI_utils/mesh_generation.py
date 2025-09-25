import os
import sys
from Python_Code.Utilis.load_DicomExam import loadDicomExam
from Python_Code.Mesh_fitting import fit_mesh
from Python_Code.Utilis.analysis_utils import (
    compute_cardiac_parameters,
    analyze_mesh_volumes,
    meshes_compute_thickness_map
)

def mesh_fit_save_images(input_path, output_folder, log_func=print, progress_func=None, **fit_params):
    """Load a pre-segmented DicomExam, fit meshes, and save mesh visualizations."""
    log_func("Loading Segmentation from (already segmented) DicomExam from Folder")
    de = loadDicomExam(input_path, output_folder)  # load DicomExam object
    de.summary()  # print summary info

    de.fitted_meshes = {}  # initialize mesh container
    fit_mesh(de, progress_callback=progress_func, **fit_params)  # fit meshes, forward progress

    log_func("Save Visualization of Generated Meshes ..")
    de.save_images(use_mesh_images=True)  # save mesh visualization images
    de.save()  # save updated DicomExam object

    # Return folder with mesh images and the DicomExam object
    return os.path.join(de.folder['mesh_segs']), de


def compute_cardiac_parameters_step(de, log_func=print):
    """Compute cardiac parameters from fitted mesh."""
    log_func("Analyzing Sliced Mesh ...")
    compute_cardiac_parameters(de, 'mesh')  # compute metrics from mesh slices

    # Return folder with cardiac plot outputs
    return os.path.join(de.folder['mesh_plots'])


def analyze_mesh_volumes_step(de, log_func=print):
    """Analyze volumes of the fitted mesh."""
    log_func("Analyzing Volumetric Meshes ...")
    analyze_mesh_volumes(de)  # compute volumetric measures

    # Return folder with volume plot outputs
    return os.path.join(de.folder['mesh_vol_plots'])


def calculate_segmentation_thickness_step(de, log_func=print):
    """Compute local wall thickness from segmented meshes."""
    log_func("Calculating Segmentation thickness ...")
    meshes_compute_thickness_map(de)  # compute thickness maps

    # Return folder with thickness map outputs
    return os.path.join(de.folder['mesh_thickness'])
