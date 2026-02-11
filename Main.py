import sys
import os
import _pickle as pickle
import numpy as np
import pandas as pd
import csv
import ast
import time
import cProfile
import gc
import torch

from Python_Code.DicomExam import DicomExam
from Python_Code.Utilis.analysis_utils import (
    compute_cardiac_parameters,
    analyze_mesh_volumes,
    meshes_compute_thickness_map,
    seg_masks_compute_thickness_map
)
from Python_Code.Segmentation import segment
from Python_Code.Utilis.load_DicomExam import loadDicomExam
from Python_Code.Mesh_fitting import fit_mesh
from Python_Code.Utilis.clean_MRI_utils import estimateValvePlanePosition , estimate_MRI_orientation

#Path to directory contain data to segment 
#data_dir = "ADD PATH TO DIRECTORY HERE"

#Path to output folder
#output_folder = "ADD PATH TO DIRECTORY HERE"

#list of datasets to contained in data_dir that you want to segment and fit a mesh to 
#datasets = ["NAME1", "NAME2", "NAME3", ...] #ADD NAMES OF YOUR DATA SETS

def load_time_frames_removed(csv_path, patient_id):
    result = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["patient_id"] != patient_id:
                continue

            series_dir = row["series_dir"]

            if row["frames"].strip() == "":
                result[series_dir] = None
            else:
                result[series_dir] = [
                    int(x) for x in row["frames"].split(",")
                ]

    return result


#Path to directory contain data to segment 
#data directory should contain one folder per patient with subfolders for the individual MRI series (e.g. one subfolder for SAX and one for LAX)
#data_dir = "ADD PATH TO DIRECTORY HERE" 

#Path to output folder
#output_folder = "ADD PATH TO DIRECTORY HERE"

#list of datasets contained in data_dir that you want to segment and fit a mesh to 
#datasets = ["NAME1", "NAME2", "NAME3", ...] #ADD NAMES OF YOUR DATA SETS


# Define valid stages
valid_stages = {"all", "meshing", "segmentation"}

if len(sys.argv) > 1:
    stage = sys.argv[1].lower()
    if stage in valid_stages:
        if stage == "all":
            print("Running Segmentation and Mesh Fitting...")
        elif stage == "meshing":
            print("Running Meshing Fitting...")
        elif stage == "segmentation":
            print("Running Segmentation...")
    else:
        print(f"Invalid stage: {stage}")
        print(f"Please choose one of: {', '.join(valid_stages)}")
	
else:
    stage = "all"
    print("Running Segmentation and Mesh Fitting...")

for index,dataset_to_use in enumerate(datasets): 

    print(f"Processing Data from: {dataset_to_use}")

    input_path = os.path.join(data_dir, dataset_to_use)


    if stage == "all" or stage == "segmentation":
        print("Loading DICOM data...")

        dict_z_slices_removed = None
        dict_time_frames_removed = None

        #In case of manual cleaning add the path to a csv file which list for each patient which Z HEIGHTS to remove 
        #the format should be the following. Make sure to add the correct name for the series
            # patient_id,series_dir,frames
            # Patient1,Name_OF_LAX_Series,"0,1,2,3,19"
            # Patient1,Name_OF_LAX_Series,"20,21,22,23"

        z_csv = os.path.join(data_dir, "Z_slices_to_remove.csv")

        #In case of manual cleaning add the path to a csv file which list for each patient which TIME FRAMES to remove 
        #the format should be the following. Make sure to add the correct name for the series
            # patient_id,series_dir,frames
            # Patient1,Name_OF_LAX_Series,"0,1,2,3,19"
            # Patient1,Name_OF_LAX_Series,"20,21,22,23"

        t_csv = os.path.join(data_dir, "Time_frames_to_remove.csv")

        if os.path.exists(z_csv):
            dict_z_slices_removed = load_time_frames_removed(
                z_csv,
                dataset_to_use
            )
        else:
            print(f"[INFO] File not found, skipping removal of z slices: {z_csv}")

        if os.path.exists(t_csv):
            dict_time_frames_removed = load_time_frames_removed(
                t_csv,
                dataset_to_use
            )
        else:
            print(f"[INFO] File not found, skipping removal of time frames: {t_csv}")

        
        
        # #Load MRI data as object of type DicomExam
        de = DicomExam(input_path,output_folder,dict_z_slices_removed, dict_time_frames_removed)

        for series in de.series:
            print(f"Series {series.name} has {series.cleaned_data.shape} and {series.prepped_data.shape} time frames after cleaning.")
            print(series.frames)

        #Standardise number of time frames across all series in the DicomExam object
        print("Standardising Number of Time Frames Across Series by Resampling..")
        de.standardiseTimeframes()

        # Segment MRI data
        print("Running segmentation...")
        segment(de)

        print("Save Visualization of Segmentation Results ..")
        #Save a visualisation of the full MRI data with the generated segmentation masks
        de.save_images()

        print("Estimating valve plane position...")
        #Estimate the planes lying above the LV base
        estimateValvePlanePosition(de)

        print("Estimating landmarks...")
        #Calculate landmarks
        de.estimate_landmarks()
        de.save()

        print("Cleaning data...")
        #Clean MRI data based on Segmentations
        de.clean_data()

        print("Estimate MRI orientation...")
        estimate_MRI_orientation(de)

        print("Save Cleaned Visualization of Segmentation Results ..")
        #Save a visualisation of the cleaned MRI data with the generated segmentation masks
        de.save_images(prefix='Clean')
    
        print("Analyse segmentation masks...")
        #compute blood pool volumes, myocardial volumes, ESV, EDV, SV, EF using Simpsons method
        compute_cardiac_parameters(de,'seg')

        print("Saving analysis object...")
        #Save object of type DicomExam as pickel file
        de.save()

    if stage == "meshing":

        print("Loading Segmentation from (already segmented) DicomExam")
        de = loadDicomExam(input_path,output_folder)

    if stage == "all" or stage == "meshing":
        # Remove all meshes previously fitted to the segmentation masks
        de.fitted_meshes = {}
        start_time = time.time()
        print("Running Mesh Fitting...")
        
        # Try progressively smaller batches if GPU memory is insufficient
        max_attempts = 10  # Maximum number of reduction attempts
        divisor = 1
        success = False
        df_end_dice = None

        for attempt in range(max_attempts):
            try:
                # Determine which time frames to fit
                if divisor == 1:
                    time_frames_to_fit = "all"
                    print(f"Attempting to fit all time frames at once...")
                    
                    # Fit 3D Volumetric meshes to the generated Segmentation masks
                    df_end_dice = fit_mesh(
                        de,
                        fitting_steps=5000,
                        slice_shift_penalty_weigth= 100,
                        time_frames_to_fit="all",
                    )
                    success = True
                    break
                    
                else:
                    total_frames = de.series[0].frames

                    # Split into chunks
                    num_chunks = divisor
                    chunk_size = max(1, total_frames // num_chunks)
                    print(f"Attempting to fit in {num_chunks} batches of ~{chunk_size} frames each...")
                    
                    # Process in chunks
                    all_dice_results = []
                    for chunk_idx in range(num_chunks):
                        start_frame = chunk_idx * chunk_size
                        end_frame = min((chunk_idx + 1) * chunk_size, total_frames)
                        
                        if chunk_idx == num_chunks - 1:  # Last chunk gets remaining frames
                            end_frame = total_frames
                        
                        time_frames_chunk = list(range(start_frame, end_frame))
                        print(f"  Processing chunk {chunk_idx + 1}/{num_chunks}: frames {start_frame}-{end_frame-1}")
                        
                        # Fit mesh for this chunk
                        df_chunk_dice = fit_mesh(
                            de,
                            fitting_steps=5000,
                            slice_shift_penalty_weigth= 100,
                            time_frames_to_fit=time_frames_chunk,
                        )
                        all_dice_results.append(df_chunk_dice)
                        
                        # Clear GPU cache between chunks
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Combine results from all chunks
                    import pandas as pd
                    df_end_dice = pd.concat(all_dice_results, ignore_index=True)
                    success = True
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print(f"GPU out of memory on attempt {attempt + 1}. Reducing batch size...")
                    divisor *= 2  # Double the number of chunks (halve the batch size)
                    
                    # Clear GPU memory
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    if attempt == max_attempts - 1:
                        raise RuntimeError(f"Failed to fit meshes even with {divisor} batches. GPU memory insufficient.")
                else:
                    # Re-raise if it's not a memory error
                    raise
        
        if not success:
            raise RuntimeError("Failed to complete mesh fitting after all attempts.")
        
        end_time = time.time()
        
        # Print dice scores and fitting times
        print(f"Myocardium Dice: {df_end_dice['Myocardium Dice'].mean():.3f}")
        print(f"Blood Pool Dice: {df_end_dice['Blood pool dice'].mean():.3f}")
        print(f"Fitting took: {end_time - start_time:.3f}s")
        
        # Save fitting time
        with open(de.folder['base'] + "/fitting_time.csv", "w") as f:
            f.write(f"{end_time - start_time:.3f}\n")
        
        print("Save Visualization of Generated Meshes ..")
        # Save a visualisation of the sliced mesh overlying the cleaned MRI images
        de.save_images(use_mesh_images=True)
        
        print("Analyzing Sliced Mesh ...")
        # Analysis and Plots various cardiac parameters calculated from voxelized and sliced mesh
        compute_cardiac_parameters(de, 'mesh')
        
        print("Analyzing Volumetric Meshes ...")
        # Analysis and Plots various cardiac parameters calculated directly from the volumetric mesh
        analyze_mesh_volumes(de)
        
        try:
            print("Calculate Local Thickness...")
            meshes_compute_thickness_map(de)
        except Exception as e:
            print(f"Skipping thickness calculation due to error: {e}")
            continue
        
        print("Saving Analysis Object...")
        # Save object of type DicomExam as pickle file
        de.save()
        
        del de
        del df_end_dice
        torch.cuda.empty_cache()
        gc.collect()


