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

import matplotlib.pyplot as plt
from test_helper_functions import reshape_to_grid, to3Ch

local_path    = os.getcwd()

data_dir = local_path + '/Data_healthy/'
# data_dir = '/data.lfpn/ibraun/Code/paper_volume_calculation/Patient_data/Heart_failure_infarct/'

# output_folder='hyper_outputs_patient_data'

output_folder = 'outputs_healthy_GUI'

#monkey data sets
# datasets = ['Baker', 'Blasius', 'Bobo' ,'Borris' , 'Corbinian', 'Gaerry', 'Gandalf', 'Gisbert', 'Jasper', 'Louie',
# 			'Mae', 'Maike', 'Norman', 'Palmiro', 'Prosper', 'Tibor', 'Ulita', 'Ursetta', 'Welle'] #list of all datasets for which the code should be run

datasets = ['Corbinian']

# #heart failure infarct datasets
# datasets = ['SCD0000101','SCD0000201','SCD0000301','SCD0000401','SCD0000501','SCD0000601', 'SCD0000701','SCD0000801','SCD0000901','SCD0001001','SCD0001101','SCD0001201']

# #heart failure without infarct datasets
# datasets = ['SCD0001301','SCD0001401','SCD0001501','SCD0001601','SCD0001701','SCD0001801','SCD0001901','SCD0002001','SCD0002101','SCD0002201','SCD0002301','SCD0002401']

# #Lv Hypertrophy datasets
# datasets = ['SCD0002501','SCD0002601','SCD0002701','SCD0002801','SCD0002901','SCD0003001','SCD0003101','SCD0003201','SCD0003301','SCD0003401','SCD0003501','SCD0003601']

# #Healthy datasets
# datasets = ['SCD0003701','SCD0003801','SCD0003901','SCD0004001','SCD0004101','SCD0004201','SCD0004301','SCD0004401','SCD0004501']

# datasets = ['SCD0003801']
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

        print("Cleaning data...")
        #Clean MRI data based on Segmentations
        de.clean_data()

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

        ds = 1

        prep_data = de.series[0].prepped_data
        prep_seg = de.series[0].prepped_seg

        lab = np.concatenate(np.concatenate(prep_seg[:,:,::ds,::ds], axis=2))

        img = to3Ch(lab)

        # Define a threshold to consider a pixel as "green-dominant"
        green_threshold = 1.2  # Adjust as needed
        blue_threshold = 1.2  # Adjust as needed

        # Extract RGB channels
        red_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        blue_channel = img[:, :, 2]

        # Condition: Green must be stronger than both Red and Blue
        green_mask = (green_channel > red_channel * green_threshold) & (green_channel > blue_channel * green_threshold)

        blue_mask = (blue_channel > red_channel * blue_threshold) & (blue_channel > green_channel * blue_threshold)

        # Convert the boolean mask to an integer matrix (1 for green, 0 for others)
        # binary_matrix = green_mask.astype(np.uint8)

        binary_matrix = blue_mask.astype(np.uint8)

        # Reshape it to (38, 9, 64, 64)
        seg_masks= reshape_to_grid(binary_matrix,grid_height=de.sz, grid_width=de.sz, vertical_squares=9, horizontal_squares=38)

        # Define rows and columns for plotting
        rows, cols = 9, 38
        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)

        for row in range(rows):
            for col in range(cols):
                # Use the specific axis for this subplot
                ax = axes[row, col]
                ax.imshow(seg_masks[col,row,:,:])
                ax.axis('off')  # Turn off axis labels
                
        plt.savefig("test.png")

        # # Save multiple arrays in one file
        # np.savez_compressed("Seg_results/auto_segmentations.npz", sliced_mesh=seg_masks, seg_data=prep_seg, prep_data=prep_data)

        # print("✅ Data saved successfully!")
        # sys.exit()

        x_reso = 0.8522726893425
        y_reso = 0.8522726893425
        thickness = 3
        gap = 0.7499999049273698

        # Count occurrences of value 3 in each of the 13 rows
        bp_vol = np.sum(seg_masks == 1, axis=(1, 2, 3))*x_reso*y_reso*(thickness + gap)*0.001
        df = pd.DataFrame(bp_vol, columns=['bp_vol'])
        df.to_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/outputs_healthy_GUI/Corbinian/Bp_volumes_slices.csv", index=False)
        sys.exit()



    if stage == "all" or stage == "meshing":

        #Remove all meshes previously fitted to the segmentation masks
        de.fitted_meshes = {}

        print("Running Mesh Fitting...")
        #Fit 3D Volumetric meshes to the generated Segmentation masks
        end_dices = fit_mesh(de,training_steps=800, time_frames_to_fit="all_loop", burn_in_length=0, train_mode='normal',
		mode_loss_weight = 7.405277111193427e-07, #how strongly to penalise large mode values
		global_shift_penalty_weigth = 0.3, steps_between_progress_update=400,
		lr = 0.017379164382135315, num_cycle = 5, num_modes = 25) #fits a mesh to every time frame. Check the function definition for a list of its arguments

        print(f"Mean End Dice: {np.mean(np.array(end_dices))}")

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
