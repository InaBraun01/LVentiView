import sys
import gc
import numpy as np
from scipy.ndimage import zoom, label
from scipy.ndimage.measurements import center_of_mass
import torch
import torch.nn as nn
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd

from cinema import ConvUNetR

import matplotlib.pyplot as plt

def plot_all_channels(pred, slice_idx=None, n_classes=4):
    """
    Plot each channel of a one-hot segmentation prediction side by side.

    Args:
        pred (np.ndarray): shape (n_slices, H, W, n_classes)
        slice_idx (int, optional): which slice to plot; defaults to the middle one.
        n_classes (int): number of channels in pred.
    """
    if slice_idx is None:
        slice_idx = pred.shape[0] // 2

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))

    for c in range(n_classes):
        axes[c].imshow(pred[slice_idx, ..., c], cmap="gray")
        axes[c].set_title(f"Channel {c}\n(sum={pred[slice_idx, ..., c].sum():.0f})")
        axes[c].axis("off")

    plt.tight_layout()
    plt.savefig("test.png")


def produce_segmentation_at_required_resolution(
    data, pixel_spacing, is_sax=True, trained_dataset="mnms2", seed=0,
    lv_class=2, n_classes=4
):
    """
    Normalize and resample input data, then generate segmentation using
    CineMA's pretrained ConvUNetR model.

    Args:
        data (np.ndarray): 4D array of input images (time, z, height, width).
        pixel_spacing (tuple): Pixel spacing values (z, y, x).
        is_sax (bool): If True, use SAX model; otherwise, use LAX 4-chamber model.
        trained_dataset (str): CineMA checkpoint family to use (e.g. "mnms2").
        seed (int): which seed of the fine-tuned checkpoint to load.
        lv_class (int): class index of LV blood pool — verify per checkpoint.
        n_classes (int): number of output classes for the chosen checkpoint.

    Returns:
        tuple: normalized data, segmentation map, center coordinates (c1, c2)
    """
    # Resample to 1mm x 1mm in-plane resolution
    zoom_factors = (1, 1, pixel_spacing[1], pixel_spacing[2])

    data = zoom(data, zoom_factors, order=1)

    # Normalize intensities
    data = data - data.min()
    data = np.clip(data, 0, np.percentile(data, 99.5))
    data = data / data.max()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    view = "sax" if is_sax else "lax_4c"

    model = ConvUNetR.from_finetuned(
        repo_id="mathpluscode/CineMA",
        model_filename=f"finetuned/segmentation/{trained_dataset}_{view}/{trained_dataset}_{view}_{seed}.safetensors",
        config_filename=f"finetuned/segmentation/{trained_dataset}_{view}/config.yaml",
    )
    model.eval()
    model.to(device)

    if view == "sax":
        pred, c1, c2 = get_segmentation_sax(data, model, device, view=view,
                                           lv_class=lv_class, n_classes=n_classes)  
    else:
        pred, c1, c2 = get_segmentation_lax(data, model, device, view=view,
                                     lv_class=lv_class, n_classes=n_classes)

    pred = pred[..., :3]          # Drop the fourth channel that is empty
    pred = np.sum(pred * [1, 2, 3], axis=-1)
                                
    #pred = np.sum(pred * np.array([1, 2, 3] + [0] * (n_classes - 3)), axis=-1)
    print(np.unique(pred))  

    old = pred.copy()

    pred[old == 0] = 3
    pred[old == 1] = 0
    pred[old == 2] = 1
    pred[old == 3] = 2

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


def get_segmentation_sax(data, model, device, view="sax", sz=192,
                      lv_class=3, n_classes=4):
    """
    Iteratively find center of left ventricle and generate segmentation
    using CineMA's ConvUNetR.

    Args:
        data (np.ndarray): Input image data, shape (time, z, height, width).
        model (torch.nn.Module): CineMA ConvUNetR model, already on device/eval.
        device (torch.device): Torch device.
        view (str): "sax" or "lax_4c" — must match the model's trained view.
        sz (int): Size of cropped square region (must match model's trained resolution).
        lv_class (int): Channel/class index corresponding to LV blood pool.
            !! VERIFY against the checkpoint's config.yaml before trusting !!
        n_classes (int): Total number of output classes the model predicts.

    Returns:
        tuple: segmentation map, c1, c2
    """
    n_time, n_z, height, width = data.shape
    c1, c2 = height // 2, width // 2
    center_moved = True
    all_c1c2 = [(c1, c2)]
    center_moved_counter = -1

    transform = Compose([
        ScaleIntensityd(keys=view),
        SpatialPadd(keys=view, spatial_size=(sz, sz, 16), method="end"),
    ])

    def run_model_on_roi(roi):
        """
        roi: np.ndarray, shape (time, z, sz, sz)
        Returns: pred, shape (time, z, sz, sz, n_classes) softmax probabilities
        """
        pred_per_frame = []
        with torch.no_grad():
            for t in range(roi.shape[0]):
                frame = np.transpose(roi[t], (1, 2, 0))  # (sz, sz, z) #check frame looks like it should

                batch = transform({view: torch.from_numpy(frame[None, ...])})
                batch = {k: v[None, ...].to(device=device, dtype=torch.float32)
                          for k, v in batch.items()}
                logits = model(batch)[view]          # (1, n_classes, sz, sz, z_padded)
                logits = logits[..., :n_z]            # crop back z padding
                probs = torch.softmax(logits, dim=1)[0]           # (n_classes, sz, sz, z)
                probs = probs.permute(3, 1, 2, 0).cpu().numpy()   # (z, sz, sz, n_classes)
                pred_per_frame.append(probs)
        return np.stack(pred_per_frame, axis=0)  # (time, z, sz, sz, n_classes)

    while center_moved:
        center_moved_counter += 1
        center_moved = False

        # crop MRI data around current center coordinates
        roi = get_image_at(c1, c2, data, sz=sz)  # (time, z, sz, sz)


        pred = run_model_on_roi(roi)  # (time, z, sz, sz, n_classes)
        pred = pred.reshape((-1, sz, sz, n_classes))

        pred = hard_softmax(pred)

        # Average predictions across all slices (axis 0),
        # then compute the center of mass (c1,c2) of the LV blood pool
        new_c1, new_c2 = center_of_mass(np.mean(pred, axis=0)[..., lv_class])



        if np.isnan(new_c1) or np.isnan(new_c2):
            print("Invalid center of mass detected, aborting.")
            sys.exit()

        new_c1, new_c2 = int(np.round(new_c1)), int(np.round(new_c2))
        new_c1 = c1 + new_c1 - sz // 2
        new_c2 = c2 + new_c2 - sz // 2

        if abs(c1 - new_c1) > 2 or abs(c2 - new_c2) > 2:
            center_moved = True
            c1, c2 = new_c1, new_c2
            # algorithm is cycling through the same positions
            if (c1, c2) in all_c1c2:
                # average center positions in detected loop
                all_c1c2 = all_c1c2[all_c1c2.index((c1, c2)):]
                c1, c2 = np.mean(all_c1c2, axis=0).astype(int)
                break
            all_c1c2.append((c1, c2))

    # Pad prediction back to original spatial size
    # pred = np.pad(pred, ((0, 0), (c1, data.shape[2] - c1), (c2, data.shape[3] - c2), (0, 0)))

    # 1. Pad the array with zeros across all channels
    pred = np.pad(
        pred,
        ((0, 0), (c1, data.shape[2] - c1), (c2, data.shape[3] - c2), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # 2. Fill channel 0's top/bottom/left/right padded borders with 1
    # Top padding
    if c1 > 0:
        pred[:, :c1, :, 0] = 1

    # Bottom padding
    bottom_pad = data.shape[2] - c1
    if bottom_pad > 0:
        pred[:, -bottom_pad:, :, 0] = 1

    # Left padding
    if c2 > 0:
        pred[:, :, :c2, 0] = 1

    # Right padding
    right_pad = data.shape[3] - c2
    if right_pad > 0:
        pred[:, :, -right_pad:, 0] = 1


    # Remove padding introduced during ROI extraction
    pred = pred[:, sz // 2:-sz // 2, sz // 2:-sz // 2]
    # Reshape prediction to match input data shape with n_classes output
    pred = pred.reshape(data.shape + (n_classes,))

    return pred, c1, c2

def get_segmentation_lax(data, model, device, view="lax_4c", sz=256,
                          lv_class=3, n_classes=4):
    """
    Iteratively find center of left ventricle and generate segmentation
    on LAX images using CineMA's ConvUNetR. Each (time, z) slice is
    processed independently since the model expects single 2D slices.

    Args:
        data (np.ndarray): Input image data, shape (time, z, height, width).
            NOTE: if z > 1, confirm each z index is actually the same
            anatomical view this checkpoint (`view`) was trained on —
            e.g. if z represents [2ch, 3ch, 4ch], only the 4ch index
            should be passed through a `lax_4c` checkpoint.
        model (torch.nn.Module): CineMA ConvUNetR model, already on device/eval.
        device (torch.device): Torch device.
        view (str): must match the model's trained view, e.g. "lax_4c".
        sz (int): crop size — 256 per the LAX config's patch_size.
        lv_class (int): channel index for LV blood pool — VERIFY against config.
        n_classes (int): number of output classes (4 per config's out_chans).

    Returns:
        tuple: segmentation map, c1, c2
    """
    n_time, n_z, height, width = data.shape
    c1, c2 = height // 2, width // 2
    center_moved = True
    all_c1c2 = [(c1, c2)]
    center_moved_counter = -1

    transform = Compose([
        ScaleIntensityd(keys=view),
        SpatialPadd(keys=view, spatial_size=(sz, sz), method="end"),  # 2D, no trailing z
    ])

    def run_model_on_roi(roi):
        """
        roi: np.ndarray, shape (time, z, sz, sz)
        Returns: pred, shape (time, z, sz, sz, n_classes) softmax probabilities
        """
        pred_per_tz = []
        with torch.no_grad():
            for t in range(roi.shape[0]):
                pred_per_z = []
                for z in range(roi.shape[1]):
                    frame = roi[t, z][None, ...]  # (1, sz, sz) — channel-first, 2D



                    batch = transform({view: torch.from_numpy(frame)})
                    batch = {k: v[None, ...].to(device=device, dtype=torch.float32)
                              for k, v in batch.items()}  # (1, 1, sz, sz)
                    
                    logits = model(batch)[view]            # (1, n_classes, sz, sz)
                    probs = torch.softmax(logits, dim=1)[0]            # (n_classes, sz, sz)
                    probs = probs.permute(1, 2, 0).cpu().numpy()       # (sz, sz, n_classes)
                    pred_per_z.append(probs)
                pred_per_tz.append(np.stack(pred_per_z, axis=0))  # (z, sz, sz, n_classes)
        return np.stack(pred_per_tz, axis=0)  # (time, z, sz, sz, n_classes)

    while center_moved:
        center_moved_counter += 1
        center_moved = False

        roi = get_image_at(c1, c2, data, sz=sz)  # (time, z, sz, sz)

        pred = run_model_on_roi(roi)  # (time, z, sz, sz, n_classes)
        pred = pred.reshape((-1, sz, sz, n_classes))  # (time*z, sz, sz, n_classes)

        pred = hard_softmax(pred)

        new_c1, new_c2 = center_of_mass(np.mean(pred, axis=0)[..., lv_class])

        if np.isnan(new_c1) or np.isnan(new_c2):
            print("Invalid center of mass detected, aborting.")
            sys.exit()

        new_c1, new_c2 = int(np.round(new_c1)), int(np.round(new_c2))
        new_c1 = c1 + new_c1 - sz // 2
        new_c2 = c2 + new_c2 - sz // 2

        if abs(c1 - new_c1) > 2 or abs(c2 - new_c2) > 2:
            center_moved = True
            c1, c2 = new_c1, new_c2
            if (c1, c2) in all_c1c2:
                all_c1c2 = all_c1c2[all_c1c2.index((c1, c2)):]
                c1, c2 = np.mean(all_c1c2, axis=0).astype(int)
                break
            all_c1c2.append((c1, c2))

    # Pad prediction back to original spatial size
    pred = np.pad(
        pred,
        ((0, 0), (c1, data.shape[2] - c1), (c2, data.shape[3] - c2), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # Fill channel 0's padded borders with 1 (background), matching SAX version
    if c1 > 0:
        pred[:, :c1, :, 0] = 1
    bottom_pad = data.shape[2] - c1
    if bottom_pad > 0:
        pred[:, -bottom_pad:, :, 0] = 1
    if c2 > 0:
        pred[:, :, :c2, 0] = 1
    right_pad = data.shape[3] - c2
    if right_pad > 0:
        pred[:, :, -right_pad:, 0] = 1

    # Remove padding introduced during ROI extraction
    pred = pred[:, sz // 2:-sz // 2, sz // 2:-sz // 2]
    # Reshape prediction to match input data shape (time, z, H, W) with n_classes
    pred = pred.reshape(data.shape + (n_classes,))

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