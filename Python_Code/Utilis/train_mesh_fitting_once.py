"""
Mesh Fitting Training Module

This module implements the training loop for fitting 3D meshes to medical imaging data,
specifically for cardiac segmentation tasks. It includes functionality for training
neural networks to predict mesh deformations and control points.
"""

import sys,os
import csv
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Python_Code.Utilis.loss_functions_once import meshFittingLoss
from Python_Code.Utilis.fit_mesh_utils_once import (
    voxelizeUniform, getSlices, dice_loss
)

import matplotlib.pyplot as plt 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_fit_loop(dicom_exam, train_steps, learned_inputs, opt_method,optimizer, lr, li_model, mean_arr_batch, 
                   tensor_labels, mode_loss_weight, global_shift_penalty_weigth,
                   slice_shift_penalty_weigth, rotation_penalty_weigth, se, eli, pcaD, 
                   warp_and_slice_model, train_mode, steps_between_fig_saves,
                   steps_between_progress_update, mesh_offset, myo_weight=1, bp_weight=500, 
                   ts3=None, show_progress=True):
    """
    Main training loop for mesh fitting to medical imaging data.
    
    This function trains a neural network to predict mesh deformations for cardiac
    segmentation by optimizing various loss components including dice loss, mode
    regularization, and geometric penalties.
    
    Args:
        dicom_exam: DICOM examination data containing medical images
        train_steps: Maximum number of training steps
        learned_inputs: Neural network model for learning input transformations
        optimizer: PyTorch optimizer 
        li_model: Main learning model for mesh fitting
        mean_arr_batch: Batch of mean arrays for input normalization
        tensor_labels: Ground truth segmentation labels
        mode_loss_weight: Weight for mode regularization loss
        global_shift_penalty_weigth: Weight for global shift penalty
        slice_shift_penalty_weigth: Weight for slice-wise shift penalty
        rotation_penalty_weigth: Weight for rotation penalty
        se: Shape encoder model
        eli: Mesh generation function
        pcaD: PCA decoder for control points
        warp_and_slice_model: Model for warping and slicing operations
        train_mode: Training mode ('until_no_progress' or 'normal' (fixed number of steps))
        steps_between_fig_saves: Interval for saving mesh renderings
        steps_between_progress_update: Interval for progress updates
        mesh_offset: Offset for mesh positioning
        myo_weight: Weight for myocardium segmentation (default: 1)
        bp_weight: Initial weight for blood pool segmentation (default: 500)
        ts3: Training step threshold for weight scheduling (default: train_steps/3)
        show_progress: Whether to display training progress (default: True)
    
    Returns:
        array: final_outputs
            - final_outputs: Final model predictions

    """
    # Initialize training variables
    i = 0

    # Initialize Dataframes for blood pool and myocardium dice
    bp_column_names = list(range(len(dicom_exam.time_frames_to_fit)))
    df_bp_dice = pd.DataFrame(columns=bp_column_names)
    myo_column_names = list(range(len(dicom_exam.time_frames_to_fit)))
    df_myo_dice = pd.DataFrame(columns=myo_column_names)

    losses_list = []
    dice_losses_list = []

    bp_weights = []
    
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    #Tensor of ones used as input
    ones_input = torch.Tensor(np.ones((1, 1))).to(device)

    #define optimizer
    optimizer = opt_method(li_model.parameters(), lr=lr)
    #Define scheduler
    scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=200,
    threshold=1e-4,
    threshold_mode='rel',
    cooldown=50,
    min_lr=1e-6)

    while should_continue_training(i, df_myo_dice, df_bp_dice, train_steps, dicom_exam):

        # Initialize training variables
        if i > 2*ts3: #linearly decrease blood pool weight during training
            bp_weight = 0
        elif i > ts3:
            bp_weight = (2*ts3-i)/ts3
        else:
            bp_weight = 1
            
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass through the model
        outputs, modes_out, global_shifts_out, rot_out, predicted_cp, slice_shifts_out = li_model([mean_arr_batch, ones_input])

        bp_weights.append(myo_weight)
        current_bp_weight = bp_weight
        
        # Calculate all loss components
        dice_loss, modes_loss, global_shift_loss, rotation_loss, slice_shift_loss = meshFittingLoss(
            outputs, modes_out, global_shifts_out, slice_shifts_out, rot_out, tensor_labels,
            mode_loss_weight, global_shift_penalty_weigth, slice_shift_penalty_weigth,
            rotation_penalty_weigth, myo_weight, current_bp_weight, slice_weights=1
        )

        # Total loss
        loss = 5*sum(dice_loss) + sum(modes_loss) + sum(global_shift_loss) + sum(rotation_loss) + sum(slice_shift_loss)

        prev_lr = scheduler.get_last_lr()[0]
        scheduler.step(sum(dice_loss).item())
        new_lr = scheduler.get_last_lr()[0]

        if new_lr != prev_lr:
            print(f"[Step {i}] Learning rate changed: {prev_lr:.6e} → {new_lr:.6e}")

        # Store losses for tracking
        losses_list.append(loss.item())
        dice_losses_list.append(sum(dice_loss).item())
        
        losses = [loss, sum(dice_loss), sum(modes_loss), sum(global_shift_loss), sum(rotation_loss), sum(slice_shift_loss)]

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Evaluation and logging (no gradient computation needed)
        with torch.no_grad():
            # Update mesh rendering periodically
            steps_between_fig_saves = 1
            steps_between_progress_update = 1
            
            if i % steps_between_fig_saves == 0:
                    print("Update mesh")
                    mean_arr_batch = update_mesh_rendering_and_training_state(
                    dicom_exam, se, eli, warp_and_slice_model, learned_inputs, 
                    pcaD, mesh_offset)
                
            # Print progress and calculate metrics periodically
            if i % steps_between_progress_update == 0:
                print("traing progress")
                d0,d1 = print_training_progress(
                    i, train_steps, losses, outputs, tensor_labels,dicom_exam ,show_progress
                )

                df_myo_dice.loc[len(df_myo_dice)] = d0
                df_bp_dice.loc[len(df_bp_dice)] = d1

        i += 1

    df_myo_dice.to_csv(dicom_exam.folder['base'] + '/myo_dice_history.csv')
    df_bp_dice.to_csv(dicom_exam.folder['base'] + '/bp_dice_history.csv')

    return outputs


def should_continue_training(i, df_myo_dice, df_bp_dice, train_steps, dicom_exam):
    """
    Determine whether to continue training based on the training mode.
    
    Args:
        i: Current training step
        dice_history: List of dice scores from previous steps
        train_mode: Training mode ('until_no_progress' or 'normal' (fixed number of steps))
        train_steps: Maximum number of training steps
    
    Returns:
        bool: True if training should continue, False otherwise
    """

    #if the dice is very high for both the myocardium and the blood pool stop

    if len(df_myo_dice) > 0:
        last_myo_dice = df_myo_dice.iloc[-1].mean()
        last_bp_dice = df_bp_dice.iloc[-1].mean()
    
        if last_myo_dice >= 0.875:
            print("Dice Scores are high enough Early Stopping activated:")
            print(f'Myocardium dice: {last_myo_dice:.3e}, Blood pool dice: {last_bp_dice:.3e}')
            # Append current step to CSV
            with open(dicom_exam.folder['base'] + '/numbers_epochs_fit.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i])

            return False
        
        else:
            # Fixed number of training steps
            return i < train_steps

    else:
        return i < train_steps

def calculate_blood_pool_weight(i, ts3, initial_bp_weight):
    """
    Calculate the current blood pool weight based on training step.
    
    The blood pool weight decreases linearly during training:
    - Steps 0 to ts3: full weight
    - Steps ts3 to 2*ts3: linear decrease from full to zero
    - Steps > 2*ts3: zero weight
    
    Args:
        i: Current training step
        ts3: Training step threshold (typically train_steps/3)
        initial_bp_weight: Initial blood pool weight
    
    Returns:
        float: Current blood pool weight
    """
    if i > 2 * ts3:
        return 0
    elif i > ts3:
        return initial_bp_weight * (2 * ts3 - i) / ts3
    else:
        return initial_bp_weight


def update_mesh_rendering_and_training_state(dicom_exam, se, eli, warp_and_slice_model, 
                                           learned_inputs, pcaD, mesh_offset):
    """
    Update mesh rendering and training state by generating new mesh slices.
    
    This function generates a new mesh, creates slice renderings, and optionally
    updates the starting mesh with currently predicted control points.
    
    Args:
        dicom_exam: DICOM examination data
        se: Shape encoder model
        eli: Mesh generation function
        warp_and_slice_model: Model for warping and slicing operations
        learned_inputs: Neural network for learning input transformations
        pcaD: PCA decoder for control points
        mesh_offset: Offset for mesh positioning
    
    Returns:
        torch.Tensor: Rendered mesh slices
    """
    # Generate new mesh
    print("Here")
    msh = eli()
    sz = dicom_exam.sz
    ones_input = torch.Tensor(np.ones((1, 1))).to(device)

    mean_bp_arrays = []
    predicted_cps = []
    for time_step in dicom_exam.time_frames_to_fit:

        # Update the starting mesh with current predictions
        update_starting_mesh = True
        if update_starting_mesh:
            mean_arr, mean_bp_arr, origin = voxelizeUniform(msh[time_step], sz, bp_channel=True)
            mean_arr_batch = torch.Tensor(np.concatenate([mean_arr[None,None], mean_bp_arr[None,None]], axis=1)).to(device)

            # Get current mode predictions and convert to control points
            ones_input = torch.Tensor(np.ones((1, 1))).to(device)
            modes_output, _, _, _, _ = learned_inputs(ones_input)
            modes_output_reshape = modes_output.view(1,learned_inputs.num_modes, learned_inputs.num_time_steps)
            predicted_cp = pcaD(modes_output_reshape[:, :,time_step])

            mean_bp_arrays.append(mean_arr_batch)
            predicted_cps.append(predicted_cp)
       
        # Update the warp and slice model with new control points
        warp_and_slice_model.control_points = predicted_cps  #make control points a list
    
    return  mean_bp_arrays


def print_training_progress(i, train_steps, losses, outputs, tensor_labels,dicom_exam, show_progress):
    """
    Print training progress including losses and dice scores.
    
    Args:
        i: Current training step
        mesh_render: Rendered mesh slices
        train_steps: Total training steps
        losses: List of loss components [total, dice, modes, global_shift, rotation, slice_shift]
        outputs: Model predictions
        tensor_labels: Ground truth labels
        show_progress: Whether to print progress information
    
    Returns:
        float: Current average dice score across slices
    """

    d0_values = []
    d1_values = []

    for time_step in dicom_exam.time_frames_to_fit:

        d0 = dice_loss(outputs[time_step][:,:1], tensor_labels[:,:1,:,:,:,time_step])  # Myocardium
        d1 = dice_loss(outputs[time_step][:,1:], tensor_labels[:,1:,:,:,:,time_step])  # Blood pool

        d0_values.append(d0.item())
        d1_values.append(d1.item())

    latest_loss = losses[0].item()
    
    if show_progress:
        # Print main progress metrics
        print(f"{i}/{train_steps}: loss = {latest_loss:.3f}, Myo dice = {np.mean(np.array(d0_values)):.3f}, Blood pool dice = {np.mean(np.array(d1_values)):.3f}")
        
        # Print detailed loss breakdown
        print(f"loss breakdown: 1-dice = {losses[1].item():.3f}, modes = {losses[2].item():.3f}, "
              f"global shifts = {losses[3].item():.3f}, slice shifts = {losses[5].item():.3f}, "
              f"rotation = {losses[4].item():.3f}")

    return d0_values,d1_values


