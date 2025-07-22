
import os
import pyvista as pv
import numpy as np



# patient_datasets = ['Baker', 'Blasius', 'Bobo' ,'Borris' , 'Corbinian', 'Gaerry', 'Gandalf', 'Gisbert', 'Jasper', 'Louie',
# 		'Mae', 'Maike', 'Norman', 'Palmiro', 'Prosper', 'Tibor', 'Ulita', 'Ursetta'] #list of all datasets for which the code should be run

patient_datasets = ['Corbinian']

for dataset in patient_datasets:
    #scale all meshes in cardiac cycle

    for i in range(38):

        # Read the original model from a VTK file
        mesh = pv.read(f"/data.lfpn/ibraun/Code/paper_volume_calculation/outputs_healthy_GUI/{dataset}/meshes/mesh_t={i}.vtk")

        # Define the scaling factor to change the unit from m to mm
        scale_factor = 0.001 # go back to unit of mm

        # Scale the mesh
        scaled_mesh = mesh.scale([scale_factor, scale_factor, scale_factor], inplace=False)

        # Copy all point data (features) from the original mesh to the scaled mesh
        for key in mesh.point_data.keys():
            scaled_mesh.point_data[key] = mesh.point_data[key]

        # Save the scaled mesh with all features to a new VTK file
        scaled_mesh.save(f"/data.lfpn/ibraun/Code/paper_volume_calculation/outputs_healthy_GUI/{dataset}/scaled_meshes/scaled_mesh_t={i}.vtk")

    print(f"{dataset}: Scaling complete")
