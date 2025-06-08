import sys
import os
import _pickle as pickle
import numpy as np
import pandas as pd
import csv
import cProfile

from Python_Code.DicomExam import DicomExam
from Python_Code.Utilis.analysis_utils import compute_cardiac_parameters,calculate_segmentation_uncertainty, analyze_mesh_volumes
from Python_Code.Segmentation import segment
from Python_Code.Utilis.load_DicomExam import loadDicomExam
from Python_Code.Mesh_fitting import fit_mesh
from Python_Code.Utilis.clean_MRI_utils import estimateValvePlanePosition

local_path    = os.getcwd()

data_dir = local_path + '/Data_healthy/'

output_folder='outputs_healthy_GUI'

datasets = ['Baker', 'Blasius', 'Bobo' ,'Borris' , 'Corbinian', 'Gaerry', 'Gandalf', 'Gisbert', 'Jasper', 'Louie',
			'Mae', 'Maike', 'Norman', 'Palmiro', 'Prosper', 'Tibor', 'Ulita', 'Ursetta', 'Welle'] #list of all datasets for which the code should be run

datasets = ['Welle']

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

        print("Save Visualization of Segmentation Results ..")
        #Save a visualisation of the full MRI data with the generated segmentation masks
        de.save_images(prefix='full')

        print("Estimating valve plane position...")
        #Estimate the planes lying above the LV base
        estimateValvePlanePosition(de)

        print("Estimating landmarks...")
        #Calculate landmarks
        de.estimate_landmarks()

        print("Cleaning data...")
        #Clean MRI data based on Segmentations
        de.clean_data()

        print("Save Cleaned Visualization of Segmentation Results ..")
        #Save a visualisation of the cleaned MRI data with the generated segmentation masks
        de.save_images(prefix='cleaned')

        print("Analyse segmentation masks...")
        compute_cardiac_parameters(de,'seg')

        print("Saving analysis object...")
        #Save object of type DicomExam as pickel file
        de.save()

    if stage == "meshing":

        print("Loading Segmentation from (already segmented) DicomExam")
        de = loadDicomExam(input_path,output_folder)


    if stage == "all" or stage == "meshing":

        #Remove all meshes previously fitted to the segmentation masks
        de.fitted_meshes = {}

        print("Running Mesh Fitting...")
        #Fit 3D Volumetric meshes to the generated Segmentation masks
        end_dices = fit_mesh(de,training_steps=1, time_frames_to_fit="all", burn_in_length=0, train_mode='normal',
		mode_loss_weight = 7.405277111193427e-07, #how strongly to penalise large mode values
		global_shift_penalty_weigth = 0.3, steps_between_progress_update=400,
		lr = 0.017379164382135315, num_cycle = 1, num_modes = 25) #fits a mesh to every time frame. Check the function definition for a list of its arguments

        print("Save Visualization of Generated Meshes ..")
        #Save a visualisation of the sliced mesh overlying the cleaned MRI images
        de.save_images(use_mesh_images=True)

        print("Analyzing Sliced Mesh ...")
        #Analysis and Plots various cardiac parameters calculated from voxelized and sliced mesh
        compute_cardiac_parameters(de,'mesh')

        print("Analyzing Volumetric Meshes ...")
        #Analysis and Plots various cardiac parameters calculated directly from the volumetric mesh
        analyze_mesh_volumes(de) #need to implement plotting

        print("Calculating Segmentation Uncertainty ...")
        #Calculate uncertainty of generated mesh
        calculate_segmentation_uncertainty(de,'mesh')

        print("Saving Analysis Object...")
        #Save object of type DicomExam as pickel file
        de.save()


"""
Tomorrows to do:

- find nice parameters for mesh fitting 
- implement code to run for more than one series in each folder




-Clean prints in terminal 


Code that is currently not used:
- models_pytorch.py
- models.py
- segmentation_utils.py
"""
