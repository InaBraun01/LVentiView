import sys
import vtk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import pyvista as pv

import syn_mri_functions as syn

def plot_slice(i,points, inner_boundary, outer_boundary, mask, x_coords, y_coords, title="Slice View"):
    """
    Plots the segmentation mask, boundaries, and points for a single slice.
    
    Parameters:
    - points: Nx3 numpy array of (x, y, z) coordinates.
    - inner_boundary: Nx2 array of (x, y) coordinates (ordered).
    - outer_boundary: Nx2 array of (x, y) coordinates (ordered).
    - mask: 2D segmentation mask (numpy array).
    - x_coords, y_coords: 1D numpy arrays for extent of the mask (used to align image with coordinates).
    - title: Plot title.
    """
    fig, ax = plt.subplots(nrows = 1, ncols= 6, figsize=(6, 6))
    
    for i in range(6):
        # Plot mask
        extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
        im = ax[i].imshow(mask, cmap='gray', alpha=0.5, extent=extent, origin='lower')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Plot outer and inner boundaries
        ax[i].plot(outer_boundary[:, 0], outer_boundary[:, 1], 'r-', label='Outer boundary')
        ax[i].plot(inner_boundary[:, 0], inner_boundary[:, 1], 'b-', label='Inner boundary')
        
        # Plot raw points
        ax[i].scatter(points[:, 0], points[:, 1], s=5, c='cyan', alpha=0.7, label='Mesh points')

        ax[i].set_aspect('equal')
        ax[i].set_title(title)
        ax[i].legend()

    plt.tight_layout()
    plt.savefig(f"mesh_slices.png")
   # plt.show()

def plot_mask(mask, x_coords, y_coords, title="Slice View"):
    """
    Plots the segmentation mask, boundaries, and points for a single slice.
    
    Parameters:
    - points: Nx3 numpy array of (x, y, z) coordinates.
    - inner_boundary: Nx2 array of (x, y) coordinates (ordered).
    - outer_boundary: Nx2 array of (x, y) coordinates (ordered).
    - mask: 2D segmentation mask (numpy array).
    - x_coords, y_coords: 1D numpy arrays for extent of the mask (used to align image with coordinates).
    - title: Plot title.
    """
    mask = np.array(mask)

    n_cols = mask.shape[0]

    fig, ax = plt.subplots(nrows = 1, ncols= n_cols, figsize=(n_cols*6, 6))

    
    for i in range(n_cols):
        # Plot mask
        extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
        im = ax[i].imshow(mask[i,:,:], alpha=0.5, extent=extent, origin='lower')
        # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax[i].set_aspect('equal')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')

    plt.tight_layout()
    #plt.savefig(f"mesh_slices.png")
    plt.show()


def generate_segmentation_masks(mesh, z_distance = 3.75*1e-3, num_slices=6): 
    bounds = mesh.GetBounds()  
    prev_mask = None  # store previous slice mask for frustum calculation
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    #offset of first and last slice from position of heighest or lowest point
    offset_apex = 0.002 #+z_distance
    offset_base = 0.0002 #+z_distance

    #calculate volume with set number of slices
    z_slices = np.linspace(zmin + offset_apex, zmax - offset_base, num_slices)

    #calculate volume with set z distance
    #z_slices = np.arange(zmin + offset_apex, zmax - offset_base, step = z_distance)
    num_slices = len(z_slices)

    offset_base = zmax - z_slices[-1]
    #calculate distance between slices
    dist_z_slices = [abs(z_slices[j] - z_slices[j-1])*1e3 for j in range(1, len(z_slices))]

    distance = dist_z_slices[0]

    #initalize bakcground of segmentation masks
    mask_irreg = syn.generate_ellipsoidal_mask(67, 67, n_regions=5, region_size=20, value_range=(20, 160))

    #save array of generated masks
    masks = []
    myo_volume = 0
    bp_volume = 0
    for i, z in enumerate(z_slices):
        #slice mesh
        slice_polydata = syn.extract_slice(mesh, z)
        #extract points in slice as numpy array
        points = syn.polydata_to_numpy(slice_polydata)

        if len(points) < 10:
            print(len(points))

        #calculate inner and outer boundary
        inner_boundary, outer_boundary, center = syn.find_ring_boundaries_polar(points)
        ordered_inner_boundary = syn.order_points_clockwise(inner_boundary[:, :2])
        ordered_outer_boundary = syn.order_points_clockwise(outer_boundary[:, :2])

        #generate segmentation masks
        mask, x, y, ob, ib = syn.create_ring_mask_with_hole(mask_irreg, ordered_outer_boundary[:, :2], ordered_inner_boundary[:, :2],i, vol_calculation = True)

        #myo_vol, bp_vol = syn.calculate_vol(mask, i, distance, offset_base*1e3, offset_apex*1e3, num_slices)

        x_reso = 0.8523
        y_reso = 0.8523

        # if i == len(z_slices)-1:  # apex
        #     # Cap as cone to zero
        #     area_myo = np.sum(mask == 2) * x_reso * y_reso
        #     area_bp  = np.sum(mask == 3) * x_reso * y_reso
        #     myo_vol  = (offset_apex*1e3)/3 * area_myo
        #     bp_vol   = (offset_apex*1e3)/3 * area_bp

        if prev_mask is not None:
            # Normal truncated cone between slices
            myo_vol, bp_vol = syn.calculate_vol_cone_stump(prev_mask, mask, distance)
            

        else:  # base
            # # Cap as cone to zero
            # area_myo = np.sum(mask == 2) * x_reso * y_reso
            # area_bp  = np.sum(mask == 3) * x_reso * y_reso
            # myo_vol  = (offset_base*1e3)/3 * area_myo
            # bp_vol   = (offset_base*1e3)/3 * area_bp

            #Do not take the first slice into account
            # myo_vol  = 0.0
            # bp_vol   = 0.0

            #all
            myo_vol, bp_vol = syn.calculate_vol_cone_stump(mask, mask, distance)

        myo_volume += myo_vol
        bp_volume += bp_vol

        masks.append(mask)
        prev_mask = mask  # store current mask for next iteration

        # modi_mask = syn.modify_seg_mask_increa_bp(mask.copy())


        # modi_myo_vol, modi_bp_vol = syn.calculate_vol(modi_mask, i, distance, offset_base*1e3, offset_apex*1e3, num_slices)
        # print(f"Modified mask: {int(modi_myo_vol), int(modi_bp_vol)}")

        # myo_volume += modi_myo_vol
        # bp_volume += modi_bp_vol

        # masks.append(modi_mask)

    #plot_mask(masks, x, y, title="Slice View")

    return np.array(masks), myo_volume*0.001, bp_volume*0.001, distance, num_slices



#=== Example usage ===
if __name__ == "__main__":
    np.random.seed(42)

    filename = "Idealized_model/idealized_bowl.vtu"
    mesh_org = pv.read(filename)

    scale_factors = np.arange(0.3,3.1,0.1)

    df = pd.DataFrame(columns=["myo_vol", "bp_vol", "num_slices", "distance", "scale_factor"])

    for scale_factor in scale_factors:

        scale_factors = (scale_factor, scale_factor, scale_factor)
        scaled_mesh = mesh_org.scale(scale_factors, inplace=False)

        # Save the scaled mesh
        scaled_mesh.save("scaled_mesh.vtu")

        mesh = syn.load_vtu_mesh("scaled_mesh.vtu")

        masks,myo_volume, bp_volume,distance, num_slices = generate_segmentation_masks(mesh, z_distance = 3.75*1e-3)

        print(f"Myocardium: {myo_volume}")
        print(f"Blood pool: {bp_volume}")

        # Append a new row
        df.loc[len(df)] = [myo_volume, bp_volume, num_slices,distance, scale_factor]


    print(df)

    df.to_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Linear_Interpolation_Results/Linear_Interpolation_all_MRI_reso_scale.csv", index=False)

# if __name__ == "__main__":
#     np.random.seed(42)

#     df = pd.DataFrame(columns=["myo_vol", "bp_vol", "time_step","num_slices", "distance"])
    
#     #calculate volume for all time steps in the cardiac cycle
#     for i in range(38):

#         #filename = "idealized_bowl.vtu"  # Replace with your actual .vtu file
#         filename = f"/data.lfpn/ibraun/Code/paper_volume_calculation/outputs_healthy_GUI/Corbinian/scaled_meshes/scaled_mesh_t={i}.vtk"
#         mesh = syn.load_vtu_mesh(filename)
#         masks,myo_volume, bp_volume,distance,num_slices = generate_segmentation_masks(mesh,z_distance = 3.75*1e-3)
#         print(f"Myocardium: {myo_volume}")
#         print(f"Blood pool: {bp_volume}")

#         # Append a new row
#         df.loc[len(df)] = [myo_volume, bp_volume, i,num_slices,distance]

#     print(df)
#     # Max value in column 'A'
#     EDV = df['bp_vol'].max()
#     print("EDV:", EDV)
#     ESV = df['bp_vol'].min()
#     print("ESV:", ESV)
#     EF = ((EDV-ESV)/EDV)*100
#     print("EF:", EF)

#     df.to_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/outputs_healthy_GUI/Corbinian/Bp_volumes_gen_seg_masks.csv")


# if __name__ == "__main__":
#     np.random.seed(42)

#     filename = "Idealized_model/idealized_bowl.vtu"  # Replace with your actual .vtu file

#     mesh = syn.load_vtu_mesh(filename)

#     df = pd.DataFrame(columns=["myo_vol", "bp_vol", "num_slices", "distance"])

#     for num_slices in range(2,400): 

#         masks,myo_volume, bp_volume,distance,num_slices = generate_segmentation_masks(mesh,num_slices=num_slices)
#         print(f"Myocardium: {myo_volume}")
#         print(f"Blood pool: {bp_volume}")

#         # Append a new row
#         df.loc[len(df)] = [myo_volume, bp_volume, num_slices,distance]
#         #break


#     print(df)

#     df.to_csv("/data.lfpn/ibraun/Code/paper_volume_calculation/Idealized_model/Linear_Interpolation_Results/Linear_Interpolation_no_top.csv", index=False)



""""
Notes from ChatGTP I need to understand and explain in the paper:

I still overestimate the volume inside of the bowl and underestimate the volume of the bowl. Linear assumption is wrong and that is why it does not work and also does not converge to the correct value ?!?!?!

This makes sense if we look at what information you’re actually using:

You only have slice masks, i.e. 2D cross-sections.

Between slices you assume linear interpolation of cross-sectional area (cone-stump).

Now, if your test object is a bowl:

The inside volume of the bowl is concave. The true cross-sectional area between slices grows slower than linearly, so the cone-stump approximation overestimates it.

The bowl wall volume (myocardium equivalent) is convex. The true area grows faster than linearly, so cone-stump underestimates it.

So the bias you’re observing is exactly what geometry predicts.

Why does the cylindrical method “work better”?

Because the cylinder approximation essentially assumes constant area between slices. That’s actually closer to the truth for shapes where the cross-sectional change is nonlinear (like your bowl). In other words:

Cylinder = underestimates expansion, overestimates contraction.

Cone-stump = assumes linear area scaling, which doesn’t match concave/convex shapes.

For a paraboloid-like bowl, cone-stumps will systematically miss.


    
But cylindrical approach/ Simpsons method is essentially a rieman integral and therefore works. 
"""