import sys
import os
import _pickle as pickle
import numpy as np
import pandas as pd
import csv
import cProfile
import argparse

from Python_Code.DicomExam import DicomExam
from Python_Code.Utilis.analysis_utils import compute_cardiac_parameters,calculate_segmentation_uncertainty, analyze_mesh_volumes
from Python_Code.Segmentation import segment
from Python_Code.Utilis.load_DicomExam import loadDicomExam
from Python_Code.Mesh_fitting import fit_mesh
from Python_Code.Utilis.clean_MRI_utils import estimateValvePlanePosition

parser = argparse.ArgumentParser()
parser.add_argument("--num_cycle", type=float, default=5, help="Number of Fitting cycles")
parser.add_argument("--training_steps", type=float, default=800, help="Number of training steps")
parser.add_argument("--lr", type=float, default=0.017379164382135315, help="Learning rate")
parser.add_argument("--mode_loss_weight", type=float, default=7.405277111193427e-07, help="Mode loss weight")
parser.add_argument("--slice_shift_penalty_weight", type=float, default=10.0, help="Slice Shift Penalty Weight")
parser.add_argument("--global_shift_penalty_weight", type=float, default=0.3, help="Global Shift Penalty Weight")
parser.add_argument("--rotation_penalty_weight", type=float, default=1.0, help="Global Rotation Penalty Weight")
args = parser.parse_args()

local_path    = os.getcwd()

data_dir = '/data.lfpn/ibraun/Code/paper_volume_calculation/Patient_data/Healthy/'

output_folder='hyper_outputs_patient_data'

datasets = ['SCD0003701']

stage = "meshing"

# Define valid stages
valid_stages = {"all", "meshing", "segmentation"}

# if len(sys.argv) > 1:
#     stage = sys.argv[1].lower()
#     if stage in valid_stages:
#         if stage == "all":
#             print("Running Segmentation and Mesh Fitting...")
#         elif stage == "meshing":
#             print("Running Meshing Fitting...")
#         elif stage == "segmentation":
#             print("Running Segmentation...")
#     else:
#         print(f"Invalid stage: {stage}")
#         print(f"Please choose one of: {', '.join(valid_stages)}")
	
# else:
#     stage = "all"
#     print("Running Segmentation and Mesh Fitting...")

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

        # print("Cleaning data...")
        # #Clean MRI data based on Segmentations
        # de.clean_data()

        print("Estimating landmarks...")
        #Calculate landmarks
        de.estimate_landmarks()
        print(f"Center: {de.center}")


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
        try:
            end_dices = fit_mesh(
                de,
                training_steps=args.training_steps,
                time_frames_to_fit=[6],  #change this here to be the ES state and then run the hyperparameter search. Try 8 first and then try small search and then turn on big search
                burn_in_length=0,
                train_mode='normal',
                mode_loss_weight=args.mode_loss_weight,
                global_shift_penalty_weigth=args.global_shift_penalty_weight,
                steps_between_progress_update=400,
                slice_shift_penalty_weigth=args.slice_shift_penalty_weight,
                rotation_penalty_weigth=args.rotation_penalty_weight,
                lr=args.lr,
                num_cycle=args.num_cycle,
                num_modes=25
            )
        except Exception as e:
            print(f"fit_mesh crashed with exception: {e}")

            
        print(f"Myocardium Dice: {end_dices[0][0]:.3f}")
        print(f"Blood Pool Dice: {end_dices[0][1]:.3f}")

        # print("Save Visualization of Generated Meshes ..")
        # #Save a visualisation of the sliced mesh overlying the cleaned MRI images
        # de.save_images(use_mesh_images=True)

        # print("Analyzing Sliced Mesh ...")
        # #Analysis and Plots various cardiac parameters calculated from voxelized and sliced mesh
        # compute_cardiac_parameters(de,'mesh')

        print("Analyzing Volumetric Meshes ...")
        #Analysis and Plots various cardiac parameters calculated directly from the volumetric mesh
        analyze_mesh_volumes(de) #need to implement plotting


        # print("Calculating Segmentation Uncertainty ...")
        # #Calculate uncertainty of generated mesh
        # calculate_segmentation_uncertainty(de,'mesh')

        # print("Saving Analysis Object...")
        # #Save object of type DicomExam as pickel file
        # de.save()
