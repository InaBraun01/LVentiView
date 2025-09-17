import pyvista as pv

# --- Load the VTU mesh ---
mesh = pv.read("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/idealized_bowl.vtu")

# --- Scale the mesh by 2.2 ---
scaled_mesh = mesh.copy()
scaled_mesh.points *= 2.2

# --- Save the scaled mesh (optional) ---
scaled_mesh.save("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/idealized_bowl_human.vtu")