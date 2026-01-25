import sys
import gc
import numpy as np
from scipy.ndimage import zoom, label
from scipy.ndimage.measurements import center_of_mass
import torch
import torch.nn as nn

def produce_segmentation_at_required_resolution(data, pixel_spacing, is_sax=True):
    """
    Normalize and resample input data, then generate segmentation using a pre-trained model.

    Args:
        data (np.ndarray): 4D array of input images (time, z, height, width).
        pixel_spacing (tuple): Pixel spacing values (z, y, x).
        is_sax (bool): If True, use SAX model; otherwise, use LAX model.

    Returns:
        tuple: normalized data, segmentation map, center coordinates (c1, c2)
    """

    # Resample to 1mm x 1mm in-plane resolution
    zoom_factors = (1, 1, pixel_spacing[1], pixel_spacing[2])
    print(f"Zoom factors: {zoom_factors}")
    data = zoom(data, zoom_factors, order=1)


    # Normalize intensities
    data = data - data.min()
    data = np.clip(data, 0, np.percentile(data, 99.5))
    data = data / data.max()

    #Load segmentation model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    model_path = "SegmentationModels/pytorch_my_model.pth" if is_sax else "SegmentationModels/pytorch_my_LAX_model.pth"
    print(model_path)
    model = torch.load(model_path, map_location=device)
    model.eval()

    #Perform Segmentation
    pred, c1, c2 = get_segmentation(data, model, device)
    pred = np.sum(pred * [[[[[1, 2, 3]]]]], axis=-1)

    #Empty memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return data, pred, c1, c2

def get_image_at(c1, c2, data, sz=256):
    """Crop a square region centered at (c1, c2) from each frame in the data."""
    return np.pad(data, ((0, 0), (0, 0), (sz // 2, sz // 2), (sz // 2, sz // 2)))[:, :, c1:c1 + sz, c2:c2 + sz]

def hard_softmax(pred, threshold=0.3):
    """Convert softmax output into a hard one-hot encoded prediction."""
    content = np.sum(pred, axis=-1) > threshold
    pred_class = np.argmax(pred, axis=-1)
    pred *= 0
    for i in range(3):
        pred[pred_class == i, i] = 1
    pred *= content[..., None]
    return pred

def get_segmentation(data, model, device, sz=256):
    """
    Iteratively find center of left ventricle and generate segmentation.

    Args:
        data (np.ndarray): Input image data.
        model (torch.nn.Module): Segmentation model.
        device (torch.device): Torch device.
        sz (int): Size of cropped square region.

    Returns:
        tuple: segmentation map, c1, c2
    """
    _, _, height, width = data.shape
    c1, c2 = height // 2, width // 2
    center_moved = True
    all_c1c2 = [(c1, c2)]
    center_moved_counter = -1

    while center_moved:
        center_moved_counter += 1
        center_moved = False


        #crop MRI data around current center coordinates
        roi = get_image_at(c1, c2, data).reshape((-1, sz, sz, 1))

        roi = np.transpose(roi, (0, 3, 1, 2))
        roi_tensor = torch.from_numpy(roi).float().to(device)

        predictions = []
        batch_size = 20
        with torch.no_grad():
            #iterate over all batches in data set
            for i in range(0, roi_tensor.shape[0], batch_size):
                batch = roi_tensor[i:i + batch_size]
                #predict segmentation masks
                pred_batch = model(batch)
                predictions.append(pred_batch.cpu())

        pred_tensor = torch.cat(predictions, dim=0)
        pred = pred_tensor.cpu().numpy()
        pred = hard_softmax(pred)

        # Average predictions across all slices (axis 0),
        # then compute the center of mass (c1,c2) of the LV blood pool (channel 2)
        new_c1, new_c2 = center_of_mass(np.mean(pred, axis=0)[..., 2])

        if np.isnan(new_c1) or np.isnan(new_c2):
            print("Invalid center of mass detected, aborting.")
            sys.exit()

        new_c1, new_c2 = int(np.round(new_c1)), int(np.round(new_c2))
        new_c1 = c1 + new_c1 - sz // 2
        new_c2 = c2 + new_c2 - sz // 2

        if abs(c1 - new_c1) > 2 or abs(c2 - new_c2) > 2:
            center_moved = True
            c1, c2 = new_c1, new_c2
            #algorithm is cycling through the same positions
            if (c1, c2) in all_c1c2:
                #average center positions in detected loop
                all_c1c2 = all_c1c2[all_c1c2.index((c1, c2)) :]
                c1, c2 = np.mean(all_c1c2, axis=0).astype(int)
                break
            all_c1c2.append((c1, c2))

    # Pad prediction back to original spatial size
    pred = np.pad(pred, ((0, 0), (c1, data.shape[2] - c1), (c2, data.shape[3] - c2), (0, 0)))
    # Remove padding introduced during ROI extraction
    pred = pred[:, sz // 2:-sz // 2, sz // 2:-sz // 2]
    # Reshape prediction to match input data shape with 3-class output
    pred = pred.reshape(data.shape + (3,))
    return pred, c1, c2

def simple_shape_correction(msk):
    """
    Clean the segmentation mask by keeping the largest connected components for each class.
    
    Args:
        msk (np.ndarray): Segmentation mask with shape (time, z, H, W)
                         Values: 0=background, 1=RV blood pool, 2=LV myocardium, 3=LV blood pool
    
    Returns:
        np.ndarray: Cleaned segmentation mask with same shape as input
    """
    
    # Process each time frame and slice
    for i in range(msk.shape[0]):
        for j in range(msk.shape[1]):
            
            # Keep only largest LV myocardium component (class 2)
            lvmyo = msk[i, j] == 2
            labels, count = label(lvmyo) #labels and counts connected components found
            if count:
                #find largest component and remove all other components
                largest_cc = np.argmax([np.sum(labels == k) for k in range(1, count + 1)]) + 1
                msk[i, j] -= (1 - (labels == largest_cc)) * lvmyo * 2
            
            # Keep only largest LV blood pool component (class 3)
            lvbp = msk[i, j] == 3
            labels, count = label(lvbp)
            if count:
                largest_cc = np.argmax([np.sum(labels == k) for k in range(1, count + 1)]) + 1
                msk[i, j] -= (1 - (labels == largest_cc)) * lvbp * 2
            
            # Remove small RV blood pool (class 1) components (<20 pixels)
            rvbp = msk[i, j] == 1
            labels, count = label(rvbp)
            for idx in range(1, count + 1):
                cc = labels == idx
                #remove componets with less than 20 pixels
                if np.sum(cc) < 20:
                    msk[i, j] *= (1 - cc)
    
    return msk