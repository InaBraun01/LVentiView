"""
Mesh Fitting Training Module

This module implements the training loop for fitting 3D meshes to medical imaging data,
specifically for cardiac segmentation tasks. It includes functionality for training
neural networks to predict mesh deformations and control points.
"""

import sys,os
import csv
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Python_Code.Utilis.loss_functions import meshFittingLoss
from Python_Code.Utilis.fit_mesh_utils import (
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

    dice_history = []
    prediction_history = []
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

    while should_continue_training(i, dice_history, train_mode, train_steps, dicom_exam):

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
        outputs, modes_out, global_shifts_out, rot_out, predicted_cp, slice_shifts_out = li_model([mean_arr_batch[:1], ones_input])

        bp_weights.append(myo_weight)
        current_bp_weight = bp_weight

        # Calculate all loss components
        dice_loss, modes_loss, global_shift_loss, rotation_loss, slice_shift_loss = meshFittingLoss(
            outputs, modes_out, global_shifts_out, slice_shifts_out, rot_out, tensor_labels,
            mode_loss_weight, global_shift_penalty_weigth, slice_shift_penalty_weigth,
            rotation_penalty_weigth, myo_weight, current_bp_weight, slice_weights=1
        )
        
        # Total loss
        loss = 5*dice_loss + modes_loss + global_shift_loss + rotation_loss + slice_shift_loss

        prev_lr = scheduler.get_last_lr()[0]
        scheduler.step(dice_loss.item())
        new_lr = scheduler.get_last_lr()[0]

        if new_lr != prev_lr:
            print(f"[Step {i}] Learning rate changed: {prev_lr:.6e} → {new_lr:.6e}")

        # Store losses for tracking
        losses_list.append(loss.item())
        dice_losses_list.append(dice_loss.item())
        
        losses = [loss, dice_loss, modes_loss, global_shift_loss, rotation_loss, slice_shift_loss]

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Evaluation and logging (no gradient computation needed)
        with torch.no_grad():
            # Update mesh rendering periodically
            if i % steps_between_fig_saves == 0:
                mesh_render, mean_arr_batch, origin = update_mesh_rendering_and_training_state(
                    dicom_exam, se, eli, warp_and_slice_model, learned_inputs, 
                    pcaD, mesh_offset)
                
            # Print progress and calculate metrics periodically
            if i % steps_between_progress_update == 0:
                d0,d1 = print_training_progress(
                    i, mesh_render, train_steps, losses, outputs, tensor_labels, show_progress
                )
                dice_history.append((d0.item(),d1.item()))
                prediction_history.append(outputs.detach().cpu().numpy())

        i += 1

    with open(dicom_exam.folder['base'] + '/dice_history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Myocardium Dice', 'Blood Pool Dice'])  # optional header
        writer.writerows(dice_history)

    return outputs


def should_continue_training(i, dice_history, train_mode, train_steps, dicom_exam):
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
    
    if len(dice_history) > 0 and i > 100 and dice_history[-1][0] >= 0.875:
        print("Dice Scores are high enough Early Stopping activated:")
        print(f'Myocardium dice: {dice_history[-1][0]:.3e}, Blood pool dice: {dice_history[-1][1]:.3e}')
        # Append current step to CSV
        with open(dicom_exam.folder['base'] + '/numbers_epochs_fit.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i])

        return False

    if train_mode == 'until_no_progress':
        # Continue if we haven't seen enough history, if we're still improving,
        # or if we haven't reached minimum steps
        if (len(dice_history) < 2 or 
            (len(dice_history) - np.argmax(dice_history)) < 4 or 
            i < train_steps):
            return True
        else:
            return False
    else:
        # Fixed number of training steps
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
    msh = eli()
    sz = dicom_exam.sz
    ones_input = torch.Tensor(np.ones((1, 1))).to(device)
    
    # Get mesh slices
    mesh_render = getSlices(se, msh, sz, True, mesh_offset, learned_inputs, ones_input)

    # Update the starting mesh with current predictions
    update_starting_mesh = True
    if update_starting_mesh:
        mean_arr, mean_bp_arr, origin = voxelizeUniform(msh, sz, bp_channel=True)
        mean_arr_batch = torch.Tensor(np.concatenate([mean_arr[None,None], mean_bp_arr[None,None]], axis=1)).to(device)
        # Get current mode predictions and convert to control points
        ones_input = torch.Tensor(np.ones((1, 1))).to(device)
        modes_output, _, _, _, _ = learned_inputs(ones_input)
        predicted_cp = pcaD(modes_output)
        
        # Update the warp and slice model with new control points
        warp_and_slice_model.control_points = predicted_cp

    
    return mesh_render, mean_arr_batch, origin


def print_training_progress(i, mesh_render, train_steps, losses, outputs, tensor_labels, show_progress):
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

    d0 = dice_loss(outputs[:,:1], tensor_labels[:,:1])  # Myocardium
    d1 = dice_loss(outputs[:,1:], tensor_labels[:,1:])  # Blood pool

    latest_loss = losses[0].item()
    
    if show_progress:
        # Print main progress metrics
        print(f"{i}/{train_steps}: loss = {latest_loss:.3f}, Myo dice = {d0:.3f}, Blood pool dice = {d1:.3f}")
        
        # Print detailed loss breakdown
        print(f"loss breakdown: 1-dice = {losses[1].item():.3f}, modes = {losses[2].item():.3f}, "
              f"global shifts = {losses[3].item():.3f}, slice shifts = {losses[5].item():.3f}, "
              f"rotation = {losses[4].item():.3f}")

    return d0,d1


