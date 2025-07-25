"""
DicomExam Class for Medical Image Analysis

This module contains the DicomExam class for loading, processing, and analyzing 
cardiac MRI DICOM series with MRI cleaning, visualization and landmark detection 
capabilities.
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import imageio
from scipy.ndimage import zoom
from skimage import measure
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# Local imports
from Python_Code.DicomSeries import DicomSeries
from Python_Code.Utilis.folder_utils import (
    generate_exam_folders, sort_folder_names, path_leaf, create_output_folders
)
from Python_Code.Utilis.pytorch_segmentation_utils import (
    produce_segmentation_at_required_resolution, simple_shape_correction
)
from Python_Code.Utilis.visualizeDICOM import planeToXYZ, to3Ch
from Python_Code.Utilis.clean_MRI_utils import (
    clean_slices_base, clean_slices_apex, estimateValvePlanePosition, clean_time_frames,postprocess_cleaned_data
)


class DicomExam:
    """
    A class for handling and processing cardiac MRI DICOM examinations.
    
    This class loads multiple DICOM series from a directory, processes them for
    cardiac analysis. This includes cleaning of the MRI images (removing incomplete
    time frames, slices above/below valve plane/base) and provides methods for landmark detection,
    and visualization of MRI data with if given segmentation masks and slices meshes.
    
    Attributes:
        id_string (str): Unique identifier for the exam
        base_dir (str): Path to the base directory containing DICOM series
        series (list): List of DicomSeries objects (containing one MRI scan)
        series_names (list): Names of the loaded Dicom series
        output_folder (str): Path to output directory
        time_frames (int): Number of time frames in the cardiac cycle
        device (torch.device): Computing device (CPU/GPU)
        fitted_meshes (dict): Dictionary storing fitted 3D meshes
        folder (dict): Dictionary of output folder paths
        
    Cardiac-specific attributes:
        vpc (np.ndarray): Valve plane center coordinates
        sax_normal (np.ndarray): Short axis normal vector
        valve_center (np.ndarray): Aortic valve center coordinates
        aortic_valve_direction (np.ndarray): Aortic valve direction vector
        base_center (np.ndarray): Center of cardiac base
        center (np.ndarray): Overall center of the cardiac volume
    """
    
    def __init__(self, base_dir, output_folder='outputs_healthy_test', id_string=None):
        """
        Initialize a DicomExam object.
        
        Args:
            base_dir (str): Path to directory containing DICOM series folders
            output_folder (str): Name of output folder for results
            id_string (str, optional): Custom patient identifier. If None, derived from base_dir
        """
        # Basic attributes
        self.id_string = path_leaf(base_dir).split('.')[0] if id_string is None else id_string
        self.base_dir = base_dir
        self.series = []
        self.series_names = []
        self.output_folder = output_folder
        
        # Processing attributes
        self.sax_slice_intersections = None
        self.series_to_exclude = []
        self.fitted_meshes = {}
        self.folder = generate_exam_folders(self.output_folder, self.id_string)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cardiac landmark attributes (initialized later)
        self.vpc = None
        self.sax_normal = None
        self.valve_center = None
        self.aortic_valve_direction = None
        self.base_center = None
        self.center = None

        # Load DICOM series from directory
        self._load_series()
        
        # Set time frames based on loaded series
        if self.num_series == 1:
            self.time_frames = self.series[0].frames
        else:
            print("WARNING: Segmentation and mesh generation for multiple series not implemented!")
            sys.exit()

    def _load_series(self):
        """
        Load all DICOM series from the base directory.
        
        Excludes hidden folders and RV-specific series from loading.
        """
        # Get ordered list of series directories
        ordered_series = sort_folder_names(os.listdir(self.base_dir))
        ordered_series = [x for x in ordered_series if not x.startswith('.')]

        # Load each series (excluding RV series)
        for series_dir in ordered_series:
            full_path = os.path.join(self.base_dir, series_dir)
            
            # Skip RV-specific series
            if 'rv' not in series_dir.lower():
                ds = DicomSeries(full_path)
                
                # Only add series with actual data
                if np.prod(ds.data.shape) > 1:
                    self.series.append(ds)
                    self.series_names.append(series_dir)
        
        self.num_series = len(self.series)

    def __str__(self):
        """Return string representation of the DicomExam."""
        s = ''
        for i, series_name in enumerate(self.series_names):
            original_shape = str(self.series[i].data.shape)
            prepped_shape = str(self.series[i].prepped_data.shape)
            s += f'{series_name} {original_shape} ({prepped_shape})\n'
        return s

    def __getitem__(self, index):
        """Get series by index."""
        return self.series[index]

    def __len__(self):
        """Return number of series."""
        return self.num_series

    def save(self):
        """
        Save the DicomExam object to a pickle file.
        
        Creates the output directory if it doesn't exist and saves the entire
        object state for later loading and analysis.
        """
        if not os.path.exists(self.folder['base']):
            os.makedirs(self.folder['base'])
                
        fname = os.path.join(self.folder['base'], 'DicomExam.pickle')
        with open(fname, "wb") as file_to_save:
            pickle.dump(self, file_to_save)

    def clean_data(self, percentage_base = 0.5,percentage_apex = 0.2 ,slice_threshold = 1):
        """
        Clean and preprocess all DICOM series data.
        
        Performs the following cleaning operations:
        1. Remove slices above the cardiac base
        2. Remove slices below the cardiac apex  
        3. Remove incomplete time frames with missing slices
        
        Updates the time_frames count and count of number of z slices

        Args: 
            percentage_base: threshold for number of missing segmentation in order for z slice 
                             to be removed at base(default 0.3 = 30%)
            percentage_apex: threshold for number of missing segmentation in order for z slice 
                             to be removed at apex (default 0.2 = 20%)
            slice_threshold: threshold of missing slices in order for the time frame 
                             to be removed (default = 1)
        """
        for series in self:

            # Remove planes above base
            clean_slices_base(series,percentage_base)

            # Remove slices below apex
            clean_slices_apex(series, percentage_apex)
            
            # Remove time frames with more than 2 missing z slices
            incomplete_frames = clean_time_frames(series, slice_threshold)

        # Update time frame count after cleaning
        self.time_frames = self.time_frames - len(incomplete_frames)

        #Postprecess cleaned data: Calculate world coordinates and new crop
        postprocess_cleaned_data(self, self.series)

            

    def save_images(self, downsample_factor=1, subfolder=None, prefix='', 
                   use_mesh_images=False, overlay=True):
        """
        Save visualization images of the DICOM data and segmentations or 
        sliced and voxelized mesh
        
        Args:
            downsample_factor (int): Factor to downsample images (default: 1)
            subfolder (str, optional): Subfolder name within output directory
            prefix (str): Prefix for output filenames
            use_mesh_images (bool): Use sliced and voxelized mesh instead of 
                                    segmentation masks
            overlay (bool): Overlay segmentations/ sliced mesh on MRI images
        """
        ds = downsample_factor
        output_folder = (self.folder['mesh_segs'] if use_mesh_images 
                        else self.folder['initial_segs'])

        if subfolder is not None:
            output_folder = os.path.join(output_folder, subfolder)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for s_ind, series in enumerate(self):
            # Select segmentation mask or sliced and volxelized mesh
            seg_data = series.mesh_seg if use_mesh_images else series.prepped_seg

            # Concatenate and downsample image data
            img = np.concatenate(np.concatenate(
                series.prepped_data[:, :, ::ds, ::ds], axis=2))
            img = to3Ch(img)

            # data = series.prepped_data

            # output_folder = "/data.lfpn/ibraun/Code/paper_volume_calculation/GUI_Results/Graphics/Segmentation_Mesh_Fitting_Flow"
            # times = [2,16,38]
            # z_heights = [0,4,8,11,15]
            # os.makedirs(output_folder, exist_ok=True)

            # for t in times:
            #     for z in z_heights:
            #         img = data[t, z, :, :]  # 2D slice at time t and height z

            #         # Save using matplotlib
            #         plt.imsave(
            #             fname=os.path.join(output_folder, f"Seg_Masks_t{t}_z{z}.pdf"),
            #             arr=img,
            #             cmap='gray'  # Use 'gray' or change as needed
            #         )

            # Save overlaid segmentation and MRI slices
            # if seg_data is not None:
            #     for t in times:
            #         for z in z_heights:
            #             mri_slice = data[t, z, :, :]         # (H, W) grayscale
            #             seg_slice = seg_data[t, z, :, :]     # (H, W) labels

            #             # Convert to 3-channel RGB
            #             mri_rgb = to3Ch(mri_slice)           # Grayscale → RGB
            #             seg_rgb = to3Ch(seg_slice)           # Labels → RGB

            #             overlay = mri_rgb + 0.8 * seg_rgb
            #             overlay = overlay / overlay.max()  # scale down if >1
            #             overlay = np.clip(overlay, 0, 1)

            #             # Convert to uint8
            #             overlay_uint8 = (overlay * 255).astype('uint8')

            #             # Save to file
            #             filename = f"Overlay_t{t}_z{z}.pdf"
            #             filepath = os.path.join(output_folder, filename)
            #             imageio.imwrite(filepath, overlay_uint8)

            # data_seg = series.prepped_seg
            # output_folder = "/data.lfpn/ibraun/Code/paper_volume_calculation/GUI_Results/Graphics/Segmentation_Mesh_Fitting_Flow"
            # times = [2,39]
            # z_heights = [0,4,8,11,15]
            # os.makedirs(output_folder, exist_ok=True)

            # for t in times:
            #     for z in z_heights:
            #         img = data_seg[t, z, :, :]  # 2D slice at time t and height z

            #         # Save using matplotlib
            #         plt.imsave(
            #             fname=os.path.join(output_folder, f"Seg_Masks_t{t}_z{z}.png"),
            #             arr=img,
            #             cmap='gray'  # Use 'gray' or change as needed
            #         )


            if seg_data is not None:
                #Downsample the segmentation and convert it to RGB format
                lab = np.concatenate(np.concatenate(
                    seg_data[:, :, ::ds, ::ds], axis=2))
                lab = to3Ch(lab)

                # Combine image and segmentation
                img = np.concatenate([img, lab], axis=0)
                img = (img * 255).astype('uint8')
                
                # Create overlay if requested
                if overlay:
                    img1, img2 = img[:img.shape[0]//2], img[img.shape[0]//2:]
                    img = np.clip(img1 + img2 * 0.7, 0, 255)
                    img = img.astype('uint8')

            # Save image
            filename = f'{prefix}_{series.series_folder_name}.png'
            imageio.imwrite(os.path.join(output_folder, filename), img)

    def summary(self):       
        """Print a comprehensive summary of the DICOM exam."""
        print('DICOM Exam Summary:')
        print(f'\tSource directory: {self.base_dir}')
        print(f'\tNumber of series: {self.num_series}')
        print('\tSeries details:')
        for series in self.series:
            print(f'\t\t{series}')

    def predict_aortic_valve_position(self):
        """
        Predict the position of the aortic valve using heuristic methods.
        
        Analyzes SAX series to identify valve plane positions and estimates
        the center of the aortic valve based on myocardium and blood pool
        segmentations.
        """
        valve_xyzs = []
        
        for series in self:
            if series.view != 'SAX':
                continue

                
            # Skip if heuristics not available
            if series.VP_heuristic2 is None:
                continue

            # Identify slices with C-shape that are within LV
            slices_to_use = ((1 - series.VP_heuristic2) * 
                           (1 - series.slice_above_valveplane))

            for t in range(self.time_frames):
                for j in range(series.slices):
                    if slices_to_use[t, j] and series.distance_from_center[j] > 0:
                        # Extract myocardium and blood pool masks
                        myo = series.prepped_seg[t, j] == 2
                        bp = series.prepped_seg[t, j] == 3

                        # Create approximate aortic valve mask
                        approx_valve_mask = (binary_dilation(bp) * 
                                           (1 - bp) * (1 - myo))

                        # Get XYZ coordinates for aortic valve 
                        xyz = series.XYZs[j].reshape((self.sz, self.sz, 3))
                        valve_coords = xyz[approx_valve_mask == 1]
                        
                        if len(valve_coords) > 0:
                            valve_xyzs.append(np.mean(valve_coords, axis=0))

        # Calculate median valve center if aortic valve found
        if len(valve_xyzs) > 0:
            self.valve_center = np.median(np.array(valve_xyzs), axis=0)
        else:
            self.valve_center = None

    def estimate_landmarks(self, series_to_use='all', center_shift=None, init_mode=1):
        """
        Estimate cardiac anatomical landmarks from SAX series.
        
        Identifies key cardiac landmarks including:
        - Valve plane center (vpc)
        - Short axis normal vector (sax_normal)  
        - Aortic valve position and direction
        
        Args:
            series_to_use (str or list): Series names to use ('all' or list of names)
            center_shift (np.ndarray, optional): Manual center adjustment
            init_mode (int): Initialization mode (0 or 1): (default = mode 1)
                                Mode 0: use all image slices for center calculation
                                Mode 1: use only slices withing LV for center calculation
        """

        all_slices = []
        for s in self:
            all_slices.extend(s.XYZs)

        grid = np.concatenate(all_slices)
        self.center = np.mean(grid, axis=0)

        #Apply manual center shift if provided
        if center_shift is not None:
            self.center -= center_shift

        # Initialize landmark lists for averaging across series
        self.vpc, self.sax_normal = [], []
        
        # Process each SAX series
        for series in self:
            if series.view != 'SAX':
                continue
                
            if series_to_use != 'all' and series.name not in series_to_use:
                continue

            # Calculate valve plane center
            if series.slice_above_valveplane is None:
                # Use first slice if valve plane not estimated
                self.vpc.append(np.mean(series.XYZs[0], axis=0) - self.center)
            else:
                # Use first slice within LV
                for k in range(len(series.XYZs)):
                    if (not series.slice_above_valveplane[0, k] and 
                        not np.sum(series.data[0, k]) == 0):
                        self.vpc.append(np.mean(series.XYZs[k], axis=0) - self.center)
                        break

            # Fallback if no suitable slice found
            if len(self.vpc) == 0:
                self.vpc.append(np.mean(series.XYZs[0], axis=0) - self.center)

            # Calculate short-axis normal vector
            X_xyz = series.orientation[3:]
            Y_xyz = series.orientation[:3]
            self.sax_normal.append(np.cross(Y_xyz, X_xyz))

        # Average landmarks across series
        if len(self.vpc) > 0:
            self.vpc = np.mean(self.vpc, axis=0)
            self.sax_normal = np.mean(self.sax_normal, axis=0)
        else:
            print('WARNING: No SAX slices found for landmark calculation')
            self.vpc = self.sax_normal = None
            return

        # Calculate base center and distances from center
        max_dist_from_center = 0
        self.base_center = None
        
        for series in self:
            if series.view != 'SAX':
                continue
                
            series.distance_from_center = []
            
            for k in range(len(series.XYZs)):
                # Center of current slice
                slice_center = np.mean(series.XYZs[k], axis=0)
                
                # Normalize SAX normal vector
                n = self.sax_normal / np.linalg.norm(self.sax_normal, 2)
                
                # Project slice center onto SAX normal line
                intersection_point = (self.center + 
                                    n * np.dot(slice_center - self.center, n))
                
                # Calculate distance from center along normal
                dist_from_center = np.mean((intersection_point - self.center) / 
                                         self.sax_normal)

                series.distance_from_center.append(dist_from_center)
                
                # Track most basal slice
                if dist_from_center > max_dist_from_center:
                    self.base_center = slice_center
                    max_dist_from_center = dist_from_center

        # Set default base center if none found
        if self.base_center is None:
            for series in self:
                if series.view == 'SAX':
                    self.base_center = np.mean(series.XYZs[0], axis=0)
                    break

        # Recursive call with center adjustment if needed
        if center_shift is None:
            adjustment = self.sax_normal * (58 - max_dist_from_center)
            self.estimate_landmarks(series_to_use=series_to_use, 
                                  center_shift=adjustment, init_mode=init_mode)
            return

        # Predict aortic valve position and direction
        self.predict_aortic_valve_position()
        if self.valve_center is not None:
            self.valve_center = self.valve_center - self.center
            
            # Calculate aortic valve direction projected onto SAX plane
            aortic_direction = self.valve_center / np.linalg.norm(self.valve_center)
            aortic_proj = (aortic_direction - 
                          np.dot(aortic_direction, self.sax_normal) * self.sax_normal)
            self.aortic_valve_direction = aortic_proj / np.linalg.norm(aortic_proj)