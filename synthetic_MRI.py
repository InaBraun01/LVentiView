"""
Cardiac MRI Synthesis from Segmentation Masks
Input:  .npy file of shape (20, 9, 128, 128)
        (Time, Z position, H, W)
Labels: 0 = background
        1 = myocardium
        2 = blood pool (LV cavity)

Output: .npy file of same shape with synthetic MRI intensities
"""
import sys, os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from Python_Code.Utilis.load_DicomExam import loadDicomExam

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Path to your segmentation .npy file
SEGMENTATION_PATH = "your_segmentation.npy"
# Output path
OUTPUT_PATH = "synthetic_cardiac_mri.npy"

# ── Realistic bSSFP signal intensities (normalised 0–1) ──────────────────────
# Based on typical bSSFP/TrueFISP/FIESTA cardiac cine sequences
# Blood pool is bright (~0.9), myocardium is medium (~0.45), contrast ratio ~2:1
INTENSITY_MAP = {
    0: 0.20,   # Background / air  → near zero
    1: 0.29,   # Myocardium        → medium signal
    2: 0.49,   # Blood pool (LV)   → bright signal
}

# ── Gaussian blur (simulates PSF and slight motion blur) ─────────────────────
# Applied per 2D slice (in-plane blur, as in real MRI acquisition)
BLUR_SIGMA = 1.0          # Typical range: 0.8–1.5

# ── Gaussian noise ────────────────────────────────────────────────────────────
NOISE_STD = 0.07        # Typical range: 0.02–0.06

# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────

def load_segmentation(path):
    """Load segmentation .npy file and validate shape."""
    mask = np.load(path).astype(np.int32)
    assert mask.ndim == 4, f"Expected 4D array (T, Z, H, W), got shape {mask.shape}"
    T, Z, H, W = mask.shape
    print(f"Loaded segmentation: shape={mask.shape} (T={T}, Z={Z}, H={H}, W={W})")
    print(f"Unique labels found: {np.unique(mask)}")
    return mask


def apply_intensity_map(mask, intensity_map):
    """
    Convert integer labels to floating-point MRI intensities.
    Operates on the full 4D array at once.
    """
    synthetic = np.zeros(mask.shape, dtype=np.float32)
    for label, intensity in intensity_map.items():
        synthetic[mask == label] = intensity

    # Warn about unexpected labels
    for label in np.unique(mask):
        if label not in intensity_map:
            print(f"  Warning: unknown label {label}, mapping to background.")
            synthetic[mask == label] = intensity_map[0]

    return synthetic


def add_gaussian_blur(synthetic, sigma):
    """
    Apply 2D Gaussian blur independently to each (T, Z) slice.
    This simulates in-plane PSF blurring, as in real MRI.
    """
    T, Z, H, W = synthetic.shape
    blurred = np.zeros_like(synthetic)
    for t in range(T):
        for z in range(Z):
            blurred[t, z] = gaussian_filter(synthetic[t, z], sigma=sigma)
    return blurred


def add_gaussian_noise(synthetic, noise_std):
    """
    Add Gaussian noise to simulate thermal/electronic noise.
    Noise is added independently per slice as in real MRI.
    """
    noise = np.random.normal(loc=0.0, scale=noise_std,
                             size=synthetic.shape).astype(np.float32)
    noisy = np.clip(synthetic + noise, 0.0, 1.0)
    return noisy

def visualise(mask, output_path):

    T, Z, H, W = mask.shape

    fig, axes = plt.subplots(Z, T, figsize=(30, 18))

    for t in range(T):
        for z in range(Z):
            axes[z, t].imshow(mask[t, z], cmap="gray")
            axes[z, t].axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_path, "visualization.png"),
        dpi=150,
        bbox_inches="tight"
    )

    print("Saved Visualization!")

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def synthesise_cardiac_mri(
    mask,
    output_path,
    intensity_map=INTENSITY_MAP,
    blur_sigma=BLUR_SIGMA,
    noise_std=NOISE_STD,
    visualise_time_idx=0,
    realistic = False
):
    print("=" * 50)
    print("Cardiac MRI Synthesis Pipeline")
    print("=" * 50)

    # 2. Apply realistic intensity contrast
    print("[1/3] Applying realistic bSSFP intensity contrast...")

    synthetic = apply_intensity_map(mask, intensity_map)
    print(f"      Blood pool intensity : {intensity_map[2]:.2f}")
    print(f"      Myocardium intensity : {intensity_map[1]:.2f}")
    print(f"      Contrast ratio       : {intensity_map[2]/intensity_map[1]:.2f}:1")

    if realistic:
        # 3. Apply Gaussian blur (per 2D slice)
        print(f"[2/3] Applying Gaussian blur (sigma={blur_sigma}) per slice...")
        synthetic = add_gaussian_blur(synthetic, sigma=blur_sigma)

        # 4. Add Gaussian noise
        print(f"[3/3] Adding Gaussian noise (std={noise_std})...")
        synthetic = add_gaussian_noise(synthetic, noise_std=noise_std)

    # Save output
    if realistic:
        np.save(os.path.join(output_path, "realistic_data.npy"), synthetic)
    else:
        np.save(os.path.join(output_path, "idealized_data.npy"), synthetic)

    print(f"\nSaved synthetic MRI to: {output_path}")
    print(f"Output shape: {synthetic.shape}  (T, Z, H, W)")

    # Visualise
    visualise(synthetic, output_path)

    print("\nDone!")
    return synthetic


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    dataset_to_use = 'SCD3701'

    data_dir = '/data/fpb/ibraun/Code/paper_volume_calculation/Patient_data/'
    input_path = os.path.join(data_dir, dataset_to_use)
    LAX_result_folder='outputs_patient_data/LAX_results_128'

    de = loadDicomExam(input_path,LAX_result_folder)
    
    SAX_series = de.series[1]
    slices = np.argmax(SAX_series.mesh_seg, axis=4)
    assert slices.ndim == 4, f"Expected sliced meshes to be a 4D array (T, Z, H, W), got shape {slices.shape}"
    T, Z, H, W = slices.shape
    print(f"Loaded segmentation: shape={slices.shape} (T={T}, Z={Z}, H={H}, W={W})")
    print(f"Unique labels found: {np.unique(slices)}")

    #created idealised MRI
    synthesise_cardiac_mri(
        slices,
        output_path=f"/data/fpb/ibraun/Code/paper_volume_calculation/Idealized_Human_model/{dataset_to_use}",
        realistic = False
    )

    #create realistic MRI
    synthesise_cardiac_mri(
        slices,
        output_path=f"/data/fpb/ibraun/Code/paper_volume_calculation/Realistic_Human_Model/{dataset_to_use}/",
        realistic = True
    )