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
data_dir = '/data.lfpn/ibraun/Code/paper_volume_calculation/Patient_data/'

# data_dir = "/data.lfpn/ibraun/Code/paper_volume_calculation/Human_data"
output_folder='outputs_patient_data/final_results' 

# output_folder = 'outputs_healthy_GUI'

#monkey data sets
# datasets = ['Baker', 'Blasius', 'Bobo' ,'Borris' , 'Corbinian', 'Gaerry', 'Gandalf', 'Gisbert', 'Jasper', 'Louie',
# 			'Mae', 'Maike', 'Norman', 'Palmiro', 'Prosper', 'Tibor', 'Ulita', 'Ursetta', 'Welle'] #list of all datasets for which the code should be run

# datasets = ['Welle']

#Healthy datasets
datasets_healthy = ['SCD0003701','SCD0003801','SCD0003901','SCD0004001','SCD0004101','SCD0004201','SCD0004301','SCD0004401','SCD0004501']
# datasets_healthy = ['SCD0003901','SCD0004101','SCD0004201','SCD0004501']
pathology_healthy = [ "Healthy/" for i in range(len(datasets_healthy))]


#heart failure infarct datasets #'SCD0000201',#'SCD0000601', 'SCD0001701' must be run again
#'SCD0000801' is too large can not run on this GPU. I either need to run it on the mac or I need to run it on CPU. 
datasets_failure_infarct = ['SCD0000101','SCD0000201','SCD0000301','SCD0000401','SCD0000501','SCD0000601', 'SCD0000701','SCD0000801','SCD0000901','SCD0001001','SCD0001101','SCD0001201']
# datasets_failure_infarct = ['SCD0000901','SCD0001001','SCD0001101','SCD0001201']

pathology_failure_infarct = ['Heart_failure_infarct/' for i in range(len(datasets_failure_infarct))]

#heart failure without infarct datasets
datasets_failure = ['SCD0001301','SCD0001401','SCD0001501','SCD0001601','SCD0001701','SCD0001801','SCD0001901','SCD0002001','SCD0002101','SCD0002201','SCD0002301','SCD0002401']
#datasets_failure = ['SCD0001401','SCD0001501','SCD0001901','SCD0002101','SCD0002401']
pathology_failure = ['Heart_failure/' for i in range(len(datasets_failure))]

#Lv Hypertrophy datasets
datasets_Lv = ['SCD0002501','SCD0002601','SCD0002701','SCD0002801','SCD0002901','SCD0003001','SCD0003101','SCD0003201','SCD0003301','SCD0003401','SCD0003501','SCD0003601']
#datasets_Lv = ['SCD0002701','SCD0002801','SCD0003001','SCD0003601']
pathology_Lv = ['LV_hypertrophy/' for i in range(len(datasets_Lv))]

datasets = datasets_healthy + datasets_failure_infarct+ datasets_failure  + datasets_Lv
pathology = pathology_healthy + pathology_failure_infarct + pathology_failure + pathology_Lv

# datasets = [ 'SCD0001601','SCD0001801','SCD0002001','SCD0002501','SCD0003501' ]
# pathology = ['Heart_failure/','Heart_failure/','Heart_failure/','LV_hypertrophy/','LV_hypertrophy/']

# datasets = ['Welle']

datasets = [ 'SCD0003701']
pathology = ['Healthy/']

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

for index,dataset_to_use in enumerate(datasets): 

    print(f"Processing Data from: {dataset_to_use}")
    # input_path = os.path.join(data_dir, dataset_to_use)

    input_folder = os.path.join(data_dir,pathology[index])
    input_path = os.path.join(input_folder, dataset_to_use)

    if stage == "all" or stage == "segmentation":
        print("Loading DICOM data...")

        #Load MRI data as object of type DicomExam
        de = DicomExam(input_path,output_folder)

        #Segment MRI data
        print("Running segmentation...")
        segment(de)

        print("Saving Automatic Segmentation Masks to Available Manual Segmentation Masks...")
        extract_auto_seg_compare_manu_seg(de)

        # print("Save Visualization of Segmentation Results ..")
        # #Save a visualisation of the full MRI data with the generated segmentation masks
        # de.save_images(prefix='full')

        # print("Estimating valve plane position...")
        # #Estimate the planes lying above the LV base
        # estimateValvePlanePosition(de)

        # print("Estimating landmarks...")
        # #Calculate landmarks
        # de.estimate_landmarks()

        # print("Cleaning data...")
        # #Clean MRI data based on Segmentations
        # de.clean_data()

        # print("Estimate MRI orientation...")
        # estimate_MRI_orientation(de)

        # print("Save Cleaned Visualization of Segmentation Results ..")
        # #Save a visualisation of the cleaned MRI data with the generated segmentation masks
        # de.save_images(prefix='cleaned')
    
        # print("Analyse segmentation masks...")
        # compute_cardiac_parameters(de,'seg')

        # # print("Calculate Local Thickness...")
        # # seg_masks_compute_thickness_map(de)

        # print("Saving analysis object...")
        # #Save object of type DicomExam as pickel file
        # de.save()

    if stage == "meshing":

        print("Loading Segmentation from (already segmented) DicomExam")
        de = loadDicomExam(input_path,output_folder)

    if stage == "all" or stage == "meshing":

        #Remove all meshes previously fitted to the segmentation masks
        de.fitted_meshes = {}

        start_time = time.time()
        print("Running Mesh Fitting...")
        #Fit 3D Volumetric meshes to the generated Segmentation masks
        try:
            # Replace this with your actual function call
            df_end_dice = fit_mesh(
                de,
                training_steps=5000, 
                time_frames_to_fit="all",
                burn_in_length=0,
                train_mode='normal',
                mode_loss_weight=7.405277111193427e-07,
                global_shift_penalty_weigth=0.3,
                steps_between_progress_update=100,
                lr=0.095,
                num_cycle=1,
                num_modes=25
            )
            end_time = time.time()
        except Exception as e:
           print(f"Error in Dataset {dataset_to_use}: {e}")

        # The loop continues with the next i automatically
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

        try:
            print("Calculate Local Thickness...")
            meshes_compute_thickness_map(de)
        except Exception as e:
           print(f"Error in Dataset {dataset_to_use}: {e}")

        print("Saving Analysis Object...")
        #Save object of type DicomExam as pickel file
        de.save()

    # del de
    # del df_end_dice  # if this is a DataFrame stored on GPU (unlikely, but safe)
    torch.cuda.empty_cache()
    gc.collect()



