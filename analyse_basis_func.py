import os,sys
import numpy as np
import pyvista as pv

model_dir = "/data.lfpn/ibraun/Code/paper_volume_calculation/ShapeModel"
tmp_mesh = '/data.lfpn/ibraun/Code/paper_volume_calculation/ShapeModel/Mean/LV_mean_scaled.vtk'


mesh = pv.read(tmp_mesh)


proj_matrix_path = os.path.join(model_dir, 'ML_augmentation/Projection_matrix.npy')
PHI = np.asmatrix(np.load(proj_matrix_path))

n_points = PHI.shape[0] // 3

num_modes = 75
PHI3 = np.reshape(np.array(PHI), (n_points, 3, num_modes), order='F')

PHI3_scaled = PHI3 * 0.5

PHI_scaled = np.reshape(PHI3_scaled, PHI.shape, order='F')

PHI3_test = np.reshape(np.array(PHI_scaled), (n_points, 3, num_modes), order='F')

new_points = PHI3_test[:,:,0]

# Replace the points in the mesh
mesh.points = new_points

# Save the modified mesh
# mesh.save("updated_file_scaled_test.vtk")

scaled_proj_matrix_path = os.path.join(model_dir, 'ML_augmentation/Scaled_Projection_matrix.npy')
np.save(scaled_proj_matrix_path, PHI_scaled)

