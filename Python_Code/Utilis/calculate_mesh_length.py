import sys,os
import pyvista as pv
import numpy as np
import pandas as pd
from pathlib import Path

def calculate_lv_height(file_path):
    # Load the unstructured grid (.vtu or .vtk)
    try:
        mesh = pv.read(file_path)
    except Exception as e:
        return f"Error loading file: {e}"

    # Extract the Z-coordinates of all points
    # mesh.points is an (N, 3) array: [x, y, z]
    z_coords = mesh.points[:, 2]

    # Instead of absolute min/max, use percentiles to handle 
    # mesh noise or 'stringy' artifacts at the base/apex.
    upper_bound = np.percentile(z_coords, 99) # The 'top' (Base)
    lower_bound = np.percentile(z_coords, 1)  # The 'bottom' (Apex)


    lv_height = upper_bound - lower_bound

    return lv_height



#Creates the LV length for all of the meshes and saves them into a csv file in the mesh analysis folder for that specific patient
root = Path("/data.lfpn/ibraun/Code/paper_volume_calculation/outputs_patient_data/SAX_final_results/")

for folder in root.iterdir():
    if folder.is_dir():
        print(folder.name)
        lengths_data = []

        #For the LAX slices
        # meshes_dir = f"/data/fpb/ibraun/Code/paper_volume_calculation/outputs_patient_data/LAX_results_128/{folder.name}/Meshes/VTK_files"
        # destination_dir = f"/data/fpb/ibraun/Code/paper_volume_calculation/outputs_patient_data/LAX_results_128/{folder.name}/Analysis_Meshes"

        #For the SAX slices
        meshes_dir = f"/data/fpb/ibraun/Code/paper_volume_calculation/outputs_patient_data/SAX_final_results/{folder.name}/meshes"
        destination_dir = f"/data/fpb/ibraun/Code/paper_volume_calculation/outputs_patient_data/SAX_final_results/{folder.name}/mesh_analysis_data"


        #Process all VTK files
        for filename in os.listdir(meshes_dir):
            if not filename.lower().endswith('.vtk'):
                continue
                
            time_frame = int(filename.split('=')[1].split('.')[0])
            vtk_path = os.path.join(meshes_dir, filename)
            
            # Calculate volumes
            lv_height = calculate_lv_height(vtk_path)
            
            lengths_data.append({
                'LV_length_cm': lv_height,
                'Time': time_frame
            })
            

        # Create and save results
        df_length = pd.DataFrame(lengths_data).sort_values('Time').reset_index(drop=True)

        df_length.to_csv(os.path.join(destination_dir, 'mesh_lengths.csv'), index=False)
