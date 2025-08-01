import os
import sys
from Python_Code.Utilis.load_DicomExam import loadDicomExam
from Python_Code.Mesh_fitting_one_time import fit_mesh
from Python_Code.Utilis.analysis_utils import (
    compute_cardiac_parameters,
    analyze_mesh_volumes,
    meshes_compute_thickness_map
)

def mesh_fit_save_images(input_path, output_folder, log_func=print, progress_func=None, **fit_params):
    log_func("Loading Segmentation from (already segmented) DicomExam from Folder")
    de = loadDicomExam(input_path, output_folder)
    de.summary()
    de.fitted_meshes = {}
    fit_mesh(de, progress_callback=progress_func, **fit_params)  # forward progress_func
    log_func("Save Visualization of Generated Meshes ..")
    de.save_images(use_mesh_images=True)
    de.save()
    return os.path.join(de.folder['mesh_segs']), de

def compute_cardiac_parameters_step(de, log_func=print):
    log_func("Analyzing Sliced Mesh ...")
    
    compute_cardiac_parameters(de, 'mesh')
    return os.path.join(de.folder['mesh_plots']) # or correct folder!

def analyze_mesh_volumes_step(de, log_func=print):
    log_func("Analyzing Volumetric Meshes ...")
    analyze_mesh_volumes(de)
    return os.path.join(de.folder['mesh_vol_plots']) # or correct folder!

def calculate_segmentation_thickness_step(de, log_func=print):
    log_func("Calculating Segmentation thickness ...")
    meshes_compute_thickness_map(de)
    return os.path.join(de.folder['mesh_thickness']) # or correct folder!