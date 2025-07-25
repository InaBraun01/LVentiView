import sys
import os
import _pickle as pickle
import numpy as np
import pandas as pd
import csv
import ast
import time
import cProfile

from Python_Code.DicomExam import DicomExam
from Python_Code.Utilis.analysis_utils import (
    compute_cardiac_parameters,
    calculate_segmentation_uncertainty,
    analyze_mesh_volumes,
    extract_auto_seg_compare_manu_seg,
    seg_masks_compute_thickness_map,
    meshes_compute_thickness_map
)
from Python_Code.Segmentation import segment
from Python_Code.Utilis.load_DicomExam import loadDicomExam
from Python_Code.Mesh_fitting_one_time import fit_mesh
from Python_Code.Utilis.clean_MRI_utils import estimateValvePlanePosition , estimate_MRI_orientation

local_path    = os.getcwd()

# data_dir = local_path + '/Data_healthy/'
data_dir = '/data.lfpn/ibraun/Code/paper_volume_calculation/Patient_data/Healthy/'

# data_dir = "/data.lfpn/ibraun/Code/paper_volume_calculation/Human_data"
output_folder='test/test_all' 

# output_folder = 'outputs_healthy_GUI'

#monkey data sets
# datasets = ['Baker', 'Blasius', 'Bobo' ,'Borris' , 'Corbinian', 'Gaerry', 'Gandalf', 'Gisbert', 'Jasper', 'Louie',
# 			'Mae', 'Maike', 'Norman', 'Palmiro', 'Prosper', 'Tibor', 'Ulita', 'Ursetta', 'Welle'] #list of all datasets for which the code should be run

# datasets = ['Welle']

# #heart failure infarct datasets
# datasets = ['SCD0000101','SCD0000201','SCD0000301','SCD0000401','SCD0000501','SCD0000601', 'SCD0000701','SCD0000801','SCD0000901','SCD0001001','SCD0001101','SCD0001201']

# #heart failure without infarct datasets
# datasets = ['SCD0001301','SCD0001401','SCD0001501','SCD0001601','SCD0001701','SCD0001801','SCD0001901','SCD0002001','SCD0002101','SCD0002201','SCD0002301','SCD0002401']

# #Lv Hypertrophy datasets
# datasets = ['SCD0002501','SCD0002601','SCD0002701','SCD0002801','SCD0002901','SCD0003001','SCD0003101','SCD0003201','SCD0003301','SCD0003401','SCD0003501','SCD0003601']

# #Healthy datasets
# datasets = ['SCD0003701','SCD0003801','SCD0003901','SCD0004001','SCD0004101','SCD0004201','SCD0004301','SCD0004401','SCD0004501']

datasets = ['SCD0003701']

# datasets = ['Patient077']
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

for dataset_to_use in datasets: 

    print(f"Processing Data from: {dataset_to_use}")
    input_path = os.path.join(data_dir, dataset_to_use)

    if stage == "all" or stage == "segmentation":
        print("Loading DICOM data...")

        #Load MRI data as object of type DicomExam
        de = DicomExam(input_path,output_folder)

        #Segment MRI data
        print("Running segmentation...")
        segment(de)

        # print("Saving Automatic Segmentation Masks to Available Manual Segmentation Masks...")
        # extract_auto_seg_compare_manu_seg(de)

        print("Save Visualization of Segmentation Results ..")
        #Save a visualisation of the full MRI data with the generated segmentation masks
        de.save_images(prefix='full')

        print("Estimating valve plane position...")
        #Estimate the planes lying above the LV base
        estimateValvePlanePosition(de)

        # print("Cleaning data...")
        # #Clean MRI data based on Segmentations
        # de.clean_data()

        print("Estimate MRI orientation...")
        estimate_MRI_orientation(de)

        print("Save Cleaned Visualization of Segmentation Results ..")
        #Save a visualisation of the cleaned MRI data with the generated segmentation masks
        de.save_images(prefix='cleaned')

        print("Estimating landmarks...")
        #Calculate landmarks
        de.estimate_landmarks()
    
        print("Analyse segmentation masks...")
        compute_cardiac_parameters(de,'seg')

        print("Calculate Local Thickness...")
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
        df_end_dice = fit_mesh(de,training_steps=1, time_frames_to_fit="all", burn_in_length=0, train_mode='normal',
		mode_loss_weight = 7.405277111193427e-07, #how strongly to penalise large mode values
		global_shift_penalty_weigth = 0.3, steps_between_progress_update=100,
		lr =  0.095, num_cycle = 1, num_modes = 25) #fits a mesh to every time frame. Check the function definition for a list of its arguments
        end_time = time.time()

        print(f"Myocardium Dice: {df_end_dice['Myocardium Dice'].mean():.3f}")
        print(f"Blood Pool Dice: {df_end_dice['Blood pool dice'].mean():.3f}")
        print(f"Fitting took: {end_time - start_time:.3f}s")

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

        # print("Calculating Segmentation Uncertainty ...")
        # #Calculate uncertainty of generated mesh
        # calculate_segmentation_uncertainty(de,'mesh')

        print("Calculate Local Thickness...")
        meshes_compute_thickness_map(de)

        print("Saving Analysis Object...")
        #Save object of type DicomExam as pickel file
        de.save()



