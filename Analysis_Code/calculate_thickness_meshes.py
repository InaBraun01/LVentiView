import os
import csv
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import Python_Code.Utilis.thickness_functions as thick

parent_folder = "/data.lfpn/ibraun/Code/paper_volume_calculation/outputs_patient_data/final_results"

log_file = "error_log.csv"

# create CSV with header if it doesn't exist yet
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "error"])

for name in os.listdir(parent_folder):
    path = os.path.join(parent_folder, name)
    if not os.path.isdir(path):
        continue

    dataset = name
    file_path = os.path.join(parent_folder, dataset)
    mesh_folder = os.path.join(file_path, "meshes")
    output_folder = os.path.join(file_path, "mesh_analysis_data", "thickness")

    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        time_step = 0
        while True:
            mesh_filename = os.path.join(mesh_folder, f"mesh_t={time_step}.vtk")

            if not os.path.exists(mesh_filename):
                next_filename = os.path.join(mesh_folder, f"mesh_t={time_step+1}.vtk")
                if os.path.exists(next_filename):
                    print(f"Skipping missing file for time_step={time_step}, continuing with {time_step+1}")
                    time_step += 1
                    continue
                else:
                    print(f"No files found for time_step={time_step} or {time_step+1}. Stopping.")
                    break

            print(f"Processing {dataset}, time step {time_step}...")

            mesh = pv.read(mesh_filename)
            points = mesh.points

            # Translate mesh to origin
            thick.translate_mesh_to_origin(mesh, points, threshold=0)

            # Convert to cylindrical coordinates
            r, theta, z = thick.cartesian_to_cylindrical(points)

            # Find ring-shaped z slices
            ring_slices, z_bins = thick.find_ring_slices(points, n_z_bins=15)

            # Plot the ring-shaped slices (close after plotting)
            thick.plot_ring_slices(points, ring_slices)
            plt.close("all")  # close matplotlib figures

            # Compute thickness map and filtered z coords
            thickness_map, filtered_z_coords = thick.compute_thickness_map(r, theta, z, ring_slices, z_bins)

            # Plot thickness map (close plot)
            thick.meshes_plot_thickness_map(thickness_map, filtered_z_coords, time_step, output_folder)
            plt.close("all")

            # Save results
            np.save(os.path.join(output_folder, f"thickness_map_{time_step}"), thickness_map)
            np.save(os.path.join(output_folder, f"filtered_z_coords_{time_step}"), filtered_z_coords)

            # Delete large variables to free memory
            del mesh, points, r, theta, z, ring_slices, z_bins, thickness_map, filtered_z_coords

            # move to next time step
            time_step += 1

    except Exception as e:
        # Log dataset + time step + error
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([dataset, time_step, str(e)])
        print(f"⚠️ Error in dataset {dataset}, time_step {time_step}: {e}")
