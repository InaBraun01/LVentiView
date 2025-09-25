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



#Lines for me to test the code

data_dir = '/data.lfpn/ibraun/Code/paper_volume_calculation/Patient_data/Healthy/'
output_folder='outputs_patient_data/test' 

datasets = [ 'SCD0003701']

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

        #Load MRI data as object of type DicomExam
        de = DicomExam(input_path,output_folder)

        #Segment MRI data
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

        print("Calculate Local Thickness...")
        #calculate local thickness from segmentation masks
        seg_masks_compute_thickness_map(de)

        print("Saving analysis object...")
        #Save object of type DicomExam as pickel file
        de.save()

    if stage == "meshing":

        print("Loading Segmentation from (already segmented) DicomExam")
        de = loadDicomExam(input_path,output_folder)

    if stage == "all" or stage == "meshing":

        #Remove all meshes previously fitted to the segmentation masks
        de.fitted_meshes = {}

        start_time = time.time()
        print("Running Mesh Fitting...")
        #Fit 3D Volumetric meshes to the generated Segmentation masks
        df_end_dice = fit_mesh(
            de,
            fitting_steps=5,    #number of fitting steps per time frame
            time_frames_to_fit=[0], #"all" or list of time frames to fit 
        )
        end_time = time.time()

        # Print dice scores and fitting times
        print(f"Myocardium Dice: {df_end_dice['Myocardium Dice'].mean():.3f}")
        print(f"Blood Pool Dice: {df_end_dice['Blood pool dice'].mean():.3f}")
        print(f"Fitting took: {end_time - start_time:.3f}s")

        #save fitting time
        with open(de.folder['base'] +"/fitting_time.csv", "w") as f:
            f.write(f"{end_time - start_time:.3f}\n")

        print("Save Visualization of Generated Meshes ..")
        #Save a visualisation of the sliced mesh overlying the cleaned MRI images
        de.save_images(use_mesh_images=True)

        print("Analyzing Sliced Mesh ...")
        #Analysis and Plots various cardiac parameters calculated from voxelized and sliced mesh
        compute_cardiac_parameters(de,'mesh')

        print("Analyzing Volumetric Meshes ...")
        #Analysis and Plots various cardiac parameters calculated directly from the volumetric mesh
        analyze_mesh_volumes(de) #need to implement plotting

        print("Calculate Local Thickness...")
        meshes_compute_thickness_map(de)

        print("Saving Analysis Object...")
        #Save object of type DicomExam as pickel file
        de.save()

    del de
    del df_end_dice  
    torch.cuda.empty_cache()
    gc.collect()



