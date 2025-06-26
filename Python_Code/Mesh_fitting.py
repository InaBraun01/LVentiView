import sys
import os

import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import imageio
import meshio
import csv
from copy import deepcopy
from scipy.spatial.transform import Rotation

# Custom utility imports
import Python_Code.Utilis.fit_mesh_utils as ut
from Python_Code.Utilis.fit_mesh_models import SliceExtractor, makeFullPPModelFromDicom
from Python_Code.Utilis.folder_utils import create_output_folders
from Python_Code.Utilis.train_mesh_fitting import train_fit_loop
from Python_Code.Utilis.visualizeDICOM import prepMeshMasks

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fit_mesh(dicom_exam, 
             time_frames_to_fit='all',
             burn_in_length=0,
             num_fits=1,
             num_cycle=5,
             lr=0.003,
             num_modes=25,
             training_steps=150,
             allow_global_shift_xy=True,
             allow_global_shift_z=True,
             allow_slice_shift=True,
             allow_rotations=True,
             mode_loss_weight=0.05,
             global_shift_penalty_weigth=0.3,
             slice_shift_penalty_weigth=10,
             rotation_penalty_weigth=1,
             steps_between_progress_update=50,
             steps_between_fig_saves=50,
             series_to_exclude=None,
             cp_frequency=50,
             mesh_model_dir='ShapeModel',
             train_mode='else',
             random_starting_mesh=False,
             show_progress=True,
             progress_callback=None):
    """
    Fits a shape model mesh to the DICOM segmentation masks.

    Args:
        dicom_exam: The DICOM exam object containing image and series information.
        time_frames_to_fit: Time frames to fit ('all' or specific list).
        burn_in_length: Number of frames to skip before fitting.
        num_fits: Number of full fitting repetitions.
        num_cycle: Number of cardiac cycles to consider.
        lr: Learning rate.
        num_modes: Number of PCA modes to use.
        initial_training_steps: Initial training steps for mesh fitting.
        training_steps: Training steps per time frame.
        allow_global_shift_xy, allow_global_shift_z, allow_slice_shift, allow_rotations: Flags to control transform degrees of freedom.
        mode_loss_weight, global_shift_penalty_weigth, slice_shift_penalty_weigth, rotation_penalty_weigth: Loss weights.
        steps_between_progress_update, steps_between_fig_saves: Logging and visualization intervals.
        slice_weighting, uncertainty_threshold: Slice weighting scheme for loss computation.
        series_to_exclude: List of DICOM series names to exclude.
        cp_frequency: Frequency of control points in shape model.
        mesh_model_dir: Path to shape model.
        random_starting_mesh: Start from a randomly perturbed mesh.
        save_*: Various flags to control result output.

    Returns:
        List of Dice scores for each fit.
    """
    
    if series_to_exclude is None:
        series_to_exclude = []

    # Enforce single series fitting
    if len(dicom_exam.series) != 1:
        print(f"Mesh fitting only works with one series. Found {len(dicom_exam.series)}.")


    dicom_exam.series_to_exclude = [s.lower() for s in series_to_exclude]

    # Setup paths and parameters
    use_bp_channel = True
    sz = dicom_exam.sz

    # Create output folders
    create_output_folders(dicom_exam, ['meshes'])

    # Load shape model components
    (mesh_1, starting_cp, PHI3, PHI, mode_bounds, mode_means, 
        mesh_offset, mesh_axes) = ut.load_ShapeModel(num_modes, sz, cp_frequency=cp_frequency, model_dir=mesh_model_dir)

    # Prepare voxelized mean mesh for input
    mean_arr_batch = ut.prepare_voxelized_mean_array(mesh_1, sz, use_bp_channel, mesh_offset, device)
    ones_input = torch.Tensor(np.ones((1, 1))).to(device)

    # Initialize model components
    se = SliceExtractor((sz, sz, sz), dicom_exam,
                        allow_global_shift_xy=allow_global_shift_xy,
                        allow_global_shift_z=allow_global_shift_z,
                        allow_slice_shift=allow_slice_shift,
                        allow_rotations=allow_rotations,
                        series_to_exclude=series_to_exclude)

    (pcaD, warp_and_slice_model, learned_inputs,
        li_model) = makeFullPPModelFromDicom(sz, num_modes, starting_cp, dicom_exam, mode_bounds, mode_means, PHI3, mesh_offset,
                                             allow_global_shift_xy=allow_global_shift_xy,
                                             allow_global_shift_z=allow_global_shift_z,
                                             allow_slice_shift=allow_slice_shift,
                                             allow_rotations=allow_rotations,
                                             series_to_exclude=series_to_exclude)

    eli = ut.evalLearnedInputs(learned_inputs, mode_bounds, mode_means, mesh_1, PHI)

    if random_starting_mesh:
        ut.init_random_start_mesh(learned_inputs, device)

    # Align initial mesh with DICOM coordinate system
    ut.set_initial_mesh_alignment(dicom_exam, mesh_axes, warp_and_slice_model, se)

    # Determine which time frames to fit
    tf_to_fit = ut.get_time_frames_to_fit(dicom_exam, time_frames_to_fit, burn_in_length, num_cycle)

    end_dices = []
    opt_method = optim.Adam

    # Run fitting for each repetition
    for rep in range(num_fits):

        # Reset model state if re-fitting
        if rep != 0:
            ut.resetModel(learned_inputs, eli, use_bp_channel, sz, mesh_offset, ones_input, pcaD, warp_and_slice_model)

        for idx, time_frame in enumerate(tf_to_fit):
            message = f"Fitting time-frame {time_frame} ({idx+1}/{len(tf_to_fit)})"
            if progress_callback:
                progress_callback(idx + 1, len(tf_to_fit))

            print(message)
            
            # Load segmentation mask and image for current time frame
            tensor_labels = ut.getTensorLabelsAndInputImage(dicom_exam, time_frame)

            # Initialize optimizer
            optimizer = optim.Adam(li_model.parameters(), lr=lr)

            # Run training loop
            outputs = train_fit_loop(
                dicom_exam,
                training_steps,
                learned_inputs,
                opt_method,
                optimizer,
                lr,
                li_model,
                mean_arr_batch,
                tensor_labels,
                mode_loss_weight,
                global_shift_penalty_weigth,
                slice_shift_penalty_weigth,
                rotation_penalty_weigth,
                se,
                eli,
                pcaD,
                warp_and_slice_model,
                train_mode,
                steps_between_fig_saves,
                steps_between_progress_update,
                mesh_offset,
                myo_weight=1,
                bp_weight=500,
                ts3=(training_steps / 3),
                show_progress=show_progress
            )

            # Save outputs and Dice score
            with torch.no_grad():
                end_dice = ut.save_results_post_training(
                    dicom_exam, outputs, time_frame, eli, se, sz, use_bp_channel,
                    mesh_offset, learned_inputs, tensor_labels)
                end_dices.append(end_dice)

    # Final processing: mask generation for fitted meshes
    prepMeshMasks(dicom_exam)

    fname = os.path.join(dicom_exam.folder['base'], 'end_dice.csv')
    np.savetxt(fname, end_dices, delimiter=",", fmt="%.4f")

    return end_dices




    