import sys
import torch
import matplotlib.pyplot as plt

def meshFittingLoss(pred, modes, global_shifts, slice_shifts, rots, target,
                    modes_weigth,
                    global_shifts_weight,
                    slice_shifts_weight,
                    rotations_weight,
                    myo_weight,
                    bp_weight,
                    slice_weights=1):
    """
    Compute mesh fitting loss for cardiac segmentation with shape model regularization.
    
    This function calculates the total loss for mesh-based cardiac segmentation,
    combining data fitting terms with regularization terms for shape model parameters.
    
    Args:
        pred (torch.Tensor): Predicted segmentation masks 
        modes (torch.Tensor): Shape model mode coefficients 
        global_shifts (torch.Tensor): Global translation parameters
        slice_shifts (torch.Tensor): Per-slice translation parameters
        rots (torch.Tensor): Rotation parameters
        target (torch.Tensor): Ground truth segmentation masks 
        modes_weigth (float): Weight for shape modes regularization
        global_shifts_weight (float): Weight for global shift regularization
        slice_shifts_weight (float): Weight for slice shift regularization
        rotations_weight (float): Weight for rotation regularization
        myo_weight (float): Weight for myocardium segmentation loss
        bp_weight (float): Weight for blood pool segmentation loss
        slice_weights (float or torch.Tensor): Per-slice weighting factors
    
    Returns:
        tuple: (data_loss, modes_loss, global_shift_loss, rotation_loss, slice_shift_loss)
            - data_loss: Main segmentation loss (Dice)
            - modes_loss: Regularization loss for shape mode coefficients
            - global_shift_loss: Regularization loss for global translations
            - rotation_loss: Regularization loss for rotations
            - slice_shift_loss: Regularization loss for slice translations
    """
    
    # Use Dice loss for segmentation masks
    # print(bp_weight)
    # # bp_weight = 1
    # print(myo_weight)
    # Calculate weighted Dice losses for different anatomical structures
    d0 = one_minus_dice_loss(pred[:,:1], target[:,:1], slice_weights) * myo_weight  # Myocardium
    d1 = one_minus_dice_loss(pred[:,1:], target[:,1:], slice_weights) * bp_weight   # Blood pool

    # print(f"Myocardium: {d0}")
    # print(f"Blood pool: {d1}")
    # sys.exit()

    # pred = pred[:,1:][0,0,:,:].cpu().detach().numpy()
    # target = target[:,1:][0,0,:,:].cpu().detach().numpy()

    # print('Blood pool')
    # print(pred.sum())
    # print(target.sum())

    # print('Myocardium')
    # pred = pred[:,:1][0,0,:,:].cpu().detach().numpy()
    # target = target[:,:1][0,0,:,:].cpu().detach().numpy()
    # print(pred.sum())
    # print(target.sum())

    # sys.exit()
    # rows = 8
    # fig, axs = plt.subplots(rows,2)
    # # Plot each mask
    # for row in range(rows):

    #         mask_pred = pred[:,:,row]
    #         im = axs[row,0].imshow(mask_pred, cmap='viridis', interpolation='nearest')
            
    #         # Remove axis ticks for cleaner look
    #         axs[row,0].set_xticks([])
    #         axs[row,0].set_yticks([])
    #         axs[0, 0].set_title("Pred")

    #         mask_target = target[:,:,row]
    #         im = axs[row,1].imshow(mask_target, cmap='viridis', interpolation='nearest')
            
    #         # Remove axis ticks for cleaner look
    #         axs[row,1].set_xticks([])
    #         axs[row,1].set_yticks([])
    #         axs[0,1].set_title("Target")
    
    
    # # Adjust layout and display
    # plt.tight_layout()
    # plt.savefig("test_2_new.png")
    # sys.exit()
    
    # Combine segmentation losses
    d_loss = d0 + d1
    
    # Calculate regularization losses for shape model parameters
    modes_loss = torch.mean(modes**2) * modes_weigth
    global_shift_loss = torch.mean(global_shifts**2) * global_shifts_weight
    slice_shift_loss = torch.mean(slice_shifts**2) * slice_shifts_weight
    rotation_loss = torch.mean(rots**2) * rotations_weight
    
    return d_loss, modes_loss, global_shift_loss, rotation_loss, slice_shift_loss


def one_minus_dice_loss(pred, target, slice_weights=1):
    """
    Compute Dice loss for segmentation evaluation.
    
    The Dice loss is computed as 1 - Dice coefficient, where the Dice coefficient
    measures the overlap between predicted and target segmentation masks.
    
    Args:
        pred (torch.Tensor): Predicted segmentation probabilities/masks
                            Shape: (batch, channels, height, width, depth)
        target (torch.Tensor): Ground truth segmentation masks  
                              Shape: (batch, channels, height, width, depth)
        slice_weights (float or torch.Tensor): Weighting factor for different slices
    
    Returns:
        torch.Tensor: Mean Dice loss across all samples and channels
        
    Note:
        - Small epsilon (0.00001) added to numerator and denominator for numerical stability
        - Loss is averaged across all spatial dimensions (0,1,2,3)
        - Final loss is weighted by slice_weights parameter
    """
    # Calculate intersection (numerator of Dice coefficient)
    # Sum over batch, channel, height, width dimensions
    numerator = 2 * torch.sum(pred * target, axis=(0,1,2,3))
    
    # Calculate union (denominator of Dice coefficient) 
    # Sum of predicted and target masks
    denominator = torch.sum(pred + target, axis=(0,1,2,3))
    
    # Compute Dice coefficient with numerical stability
    # Add small epsilon to avoid division by zero
    dloss = (numerator + 0.00001) / (denominator + 0.00001)
    
    # Convert to loss (1 - Dice coefficient) and apply slice weighting
    dloss = (1 - dloss) * slice_weights
    
    # Return mean loss across all computed values
    return torch.mean(dloss)