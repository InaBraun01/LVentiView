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
    generate_exam_folders, sort_folder_names, path_leaf
)

from Python_Code.Utilis.visualizeDICOM import  to3Ch
from Python_Code.Utilis.clean_MRI_utils import (
    clean_slices_base, clean_slices_apex, clean_time_frames,postprocess_cleaned_data
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
    
    def __init__(self, base_dir, output_folder,dict_z_slices_removed = None ,dict_time_frames_removed = None, id_string=None):
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
        self.dict_z_slices_removed = dict_z_slices_removed
        self.dict_time_frames_removed = dict_time_frames_removed
        
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
        
        self.time_frames = self.series[0].frames


    def _load_series(self):
        """
        Load all DICOM series from the base directory.
        
        Excludes hidden folders and RV-specific series from loading.
        """
        # Get ordered list of series directories
        ordered_series = sort_folder_names(os.listdir(self.base_dir))
        ordered_series = [x for x in ordered_series if not x.startswith('.')]

        if self.dict_time_frames_removed is None:
            self.dict_time_frames_removed = {}

        for series_dir in ordered_series:
            if series_dir not in self.dict_time_frames_removed:
                self.dict_time_frames_removed[series_dir] = None

        if self.dict_z_slices_removed is None:
            self.dict_z_slices_removed = {}

        for series_dir in ordered_series:
            if series_dir not in self.dict_z_slices_removed:
                self.dict_z_slices_removed[series_dir] = None


        # Load each series (excluding RV series)
        for series_dir in ordered_series:
            print(series_dir)
            full_path = os.path.join(self.base_dir, series_dir)
            
            # Skip RV-specific series
            if 'rv' not in series_dir.lower():
                ds = DicomSeries(full_path, self.dict_z_slices_removed[series_dir],self.dict_time_frames_removed[series_dir])
                
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

    def clean_data(self, percentage_base = 0.7,percentage_apex = 0.2 ,slice_threshold = 2, remove_time_steps = None, remove_z_slices = None):
        """
        Clean and preprocess all DICOM series data.
        
        Performs the following cleaning operations:
        1. Remove slices above the cardiac base
        2. Remove slices below the cardiac apex  
        3. Remove incomplete time frames with missing slices
        
        Updates the time_frames count and count of number of z slices

        Args: 
            percentage_base: threshold for number of missing segmentation in order for z slice 
                             to be removed at base(default 0.7 = 70%)
            percentage_apex: threshold for number of missing segmentation in order for z slice 
                             to be removed at apex (default 0.2 = 20%)
            slice_threshold: threshold of missing slices in order for the time frame 
                             to be removed (default = 1)
            remove_time_steps: list of times steps (integers) to remove
            remove_z_slices: list of z heights (integers) to remove
        """
        for series in self:

            if series.view not in ['SAX', 'unknown']:

                if remove_time_steps == []:
                    remove_time_steps = None

                if remove_z_slices == []:
                    remove_z_slices = None

                # if remove_time_steps:
                #     remove_time_steps = [x - 1 for x in remove_time_steps]
                #     incomplete_frames = clean_time_frames(series, slice_threshold,remove_time_steps)

                # else:
                #     incomplete_frames = clean_time_frames(series, slice_threshold)

                if remove_z_slices:
                    remove_z_slices = [x - 1 for x in remove_z_slices]
                    clean_slices_apex(series, percentage_apex,remove_z_slices)

                # else:
                #     # Remove planes above base
                #     clean_slices_base(series,incomplete_frames,percentage_base)

                #     # Remove slices below apex
                #     apex_slices = clean_slices_apex(series, percentage_apex)

                # # Update time frame count after cleaning
                # self.time_frames = self.time_frames - len(incomplete_frames)

                #Postprecess cleaned data: Calculate world coordinates and new crop
                postprocess_cleaned_data(self)

    def standardiseTimeframes(self, resample_to='fewest'):
        '''
        make sure all series have the same number of time frames, and resample them if they don't.
        '''

        print('standardising number of time frames across series by resampling..')

        if len(self.series) == 1:
            print('only one series, so no resampling required')
            self.time_frames = self.series[0].frames
            return

        time_frames_seen = []

        for s in self.series:
            time_frames_seen.append(s.frames) #append number of frames in each series

        time_frames_seen = np.unique(time_frames_seen) #collect only the unique values ß


        if len(time_frames_seen) == 1: #all series have the same number of time frames
            self.time_frames = time_frames_seen[0]
            print('all series already have %d time frame(s), so no resampling required' % (time_frames_seen[0]))
            return

        if 1 in time_frames_seen:
            print('some series only have a single time frame, these will be ignored')
            time_frames_seen = [t for t in time_frames_seen if t != 1]


        if resample_to == 'fewest':
            #downsample to smallest time resolution:
            print(time_frames_seen)
            target_slices = np.min(time_frames_seen)
            print(target_slices)
        else: 
            #otherwise upsample to highest temporal resolution:
            target_slices = np.max(time_frames_seen)

        print('resampling all series to %d time frames' % (target_slices,))
        for s in self.series:

            #skip series with only 1 time frame (upsampling them doesn't really make any sense..)
            if s.frames == 1:
                continue

            if s.frames != target_slices:
                s.prepped_data = zoom(s.data, (target_slices/s.data.shape[0],1,1,1), order=1) #zoom of data in time direction so that number of target slices is reached, while not inducing any change in other directions
                s.cleaned_data = zoom(s.data, (target_slices/s.data.shape[0],1,1,1), order=1) #zoom of data in time direction so that number of target slices is reached, while not inducing any change in other directions
                # is_sax = (s.view in ['SAX', 'unknown'])
                # dat, seg, c1, c2 = produceSegAtRequiredRes(resampled_data, s.pixel_spacing, is_sax, use_tta)
                # sz = 128
                # c1 = np.clip(c1-sz//2, 0, seg.shape[2]-sz) 
                # c2 = np.clip(c2-sz//2, 0, seg.shape[3]-sz) 
                # s.prepped_seg_resampled = np.transpose(seg[:,:,c1:c1+sz,c2:c2+sz], (0,1,3,2))
                # s.prepped_data_resampled = np.transpose(dat[:,:,c1:c1+sz,c2:c2+sz], (0,1,3,2))
                # s.resampled = True
                s.frames = target_slices

        self.time_frames = target_slices
            

    def save_images(self, downsample_factor=1, subfolder=None, prefix=None, 
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
            filename = ("Mesh_Visulization" if use_mesh_images
            else "Segmentation_Masks")

            # Select segmentation mask or sliced and volxelized mesh
            seg_data = series.mesh_seg if use_mesh_images else series.prepped_seg

            # Concatenate and downsample image data
            img = np.concatenate(np.concatenate(
                series.prepped_data[:, :, ::ds, ::ds], axis=2))
            img = to3Ch(img)

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
            
            else:
                img = (img * 255).astype('uint8')

            # Save image
            if prefix:
                filename = f'{prefix}_{series.name}_{filename}.pdf'
            else:
                filename = f'{series.name}_{filename}.pdf'

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
            slices_to_use = ((1 - series.VP_heuristic1) * 
                           (1 - series.slice_above_valveplane))

            for t in range(self.time_frames):
                for j in range(series.slices):
                    
                    #Extract myocardium and blood pool mask
                    myo = series.prepped_seg[t, j] == 2
                    bp = series.prepped_seg[t, j] == 3

                    # Step 1: Dilate the blood pool
                    dilated_bp = binary_dilation(bp)

                    # Step 2: Exclude original bp and myocardium from dilated region
                    approx_valve_mask = dilated_bp & ~bp & ~myo
                    approx_valve_mask = approx_valve_mask.astype(np.uint8)

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

        #for series in self:
            # with open(f'{series.name}_XYZs.pkl', 'rb') as f:
            #     series.XYZs = pickle.load(f)

        init_mode = 1
        if init_mode == 0:
            print("init mode is 0")
            all_slices = []
            for s in self:
                all_slices.extend(s.XYZs)
            grid = np.concatenate(all_slices)
            self.center = np.mean(grid, axis=0)
			

        elif init_mode == 1:
            print("init mode is 1")
            all_slices = []
            for s in self:
                if s.view == 'SAX':
                    
                    if s.slice_above_valveplane is None:
                        all_slices.extend(s.XYZs)
                    else:
                        for k in range(len(s.XYZs)):
                            if not s.slice_above_valveplane[0,k]:
                                print(k)
                                all_slices.append(s.XYZs[k])
                else:
                    all_slices.extend(s.XYZs)
            grid = np.concatenate(all_slices)
            self.center = np.mean(grid, axis=0)

        if center_shift is not None:
            self.center -= center_shift



        all_slices = []
        for s in self:
            all_slices.extend(s.XYZs)

        grid = np.concatenate(all_slices)
        self.center = np.mean(grid, axis=0)

        #Apply manual center shift if provided
        if center_shift is not None:
            self.center -= center_shift


        self.vpc, self.sax_normal, self.rv_center = [], [], [] #make an array and then take average to handel the case of multiple SAX series
        for s in self:
            s.uncertainty = None

            if s.view == 'SAX':

                if series_to_use != 'all' and s.name not in series_to_use:
                    continue

                ### calculate the valve plane center (relative to self.center)
                #if we haven't yet estimated the valve-plane position, just use the first slice in the SAX series
                if s.slice_above_valveplane is None:
                    self.vpc.append( np.mean(s.XYZs[0], axis=0) - self.center )
                else: #otherwise, use the center of first (most basal) slice in the LV:
                    for k in range(len(s.XYZs)):
                        if not s.slice_above_valveplane[0,k] and not np.sum(s.data[0,k])==0:
                            self.vpc.append( np.mean(s.XYZs[k], axis=0) - self.center )
                            break

                if len(self.vpc) == 0: #catch the (strange) case where no slice seems to be in the LV
                    self.vpc.append( np.mean(s.XYZs[0], axis=0) - self.center )

                #short-axis normal:
                Xxyz = s.orientation[3:]
                Yxyz = s.orientation[:3]
                self.sax_normal.append( np.cross(Yxyz,Xxyz) )

                #get center of RV (relative to self.center):
                rv_xyzs = []
                for j in range(s.prepped_seg.shape[0]):
                    
                    RV = (s.prepped_seg[j] == 1)
                    
                    if np.sum(RV) == 0:
                        continue
                    
                    for i in range(len(RV)):
                        # if s.uncertainty is not None and np.mean(s.uncertainty[j,i],axis=-1) < 0.25:
                        if s.uncertainty is not None and np.mean(s.uncertainty[j,i],axis=-1) > 0.75:
                            # print(j, 'uncertain')
                            continue
                        if s.slice_above_valveplane is not None and s.slice_above_valveplane[j,i]:
                            # print(j, 'slice_above_valveplane')
                            continue

                        # print(RV[i].shape)
                        #rv_xyzs.append(s.XYZs[i].reshape((128,128,3))[RV[i]==1])
                        #rv_xyzs.append(s.XYZs[i].reshape((200,200,3))[RV[i]==1])
                        rv_xyzs.append(s.XYZs[i].reshape((160,160,3))[RV[i]==1])

                if len(rv_xyzs) > 0:
                    rv_xyzs = np.concatenate(rv_xyzs,axis=0)
                    rv_center = np.mean(rv_xyzs,axis=0) - self.center
                    self.rv_center.append( rv_center )

        if len(self.vpc) > 0:
            self.vpc = np.mean(self.vpc, axis=0)
            self.sax_normal = np.mean(self.sax_normal, axis=0)
            self.rv_center = np.mean(self.rv_center, axis=0)
        else:
            print('warning: no SAX slices found for calculating landamrks in estimateLandmarks()')
            self.vpc, self.sax_normal, self.rv_center, self.rv_direction = None, None, None, None
            return

        #calculate the center of the base as a point in 3D space
        max_dist_from_center = 0
        self.base_center = None
        for s in self:
            if s.view == 'SAX':

                s.distance_from_center = []
                for k in range(len(s.XYZs)): #for each slice:

                    x = np.mean(s.XYZs[k],axis=0) #center of slice
                    u = self.center #center of volume
                    n = self.sax_normal / np.linalg.norm(self.sax_normal, 2) #make sure normal is length 1

                    #project the center of the slice onto the sax_normal line, passing up throuygh the volume center:
                    intersection_point = u + n*np.dot(x - u, n)

                    #calculate the ditance of this projected point fromn the center (positive--> more toward base, negative--> more toward apex)
                    dist_fom_center = np.mean((intersection_point - self.center)/self.sax_normal)#all three dims should be the same, so we just take the mean
                    s.distance_from_center.append( dist_fom_center )

                    if dist_fom_center > max_dist_from_center:
                        self.base_center = x
                        max_dist_from_center = dist_fom_center

        if self.base_center is None:
            for s in self:
                if s.view == 'SAX':
                    self.base_center = np.mean(s.XYZs[0],axis=0)

        if center_shift is None:
            self.estimate_landmarks(series_to_use=series_to_use, center_shift=self.sax_normal*(58-max_dist_from_center), init_mode=init_mode)

        rv_direction = self.rv_center / np.linalg.norm(self.rv_center)
        rv_direction_projected_on_sax_plane = rv_direction - np.dot(rv_direction, self.sax_normal) * self.sax_normal
        self.rv_direction = rv_direction_projected_on_sax_plane / np.linalg.norm(rv_direction_projected_on_sax_plane)

        self.predict_aortic_valve_position()
        if self.valve_center is not None:
            self.valve_center = self.valve_center - self.center
            aortic_valve_direction = self.valve_center / np.linalg.norm(self.valve_center)
            aortic_valve_direction_projected_on_sax_plane = aortic_valve_direction - np.dot(aortic_valve_direction, self.sax_normal) * self.sax_normal
            self.aortic_valve_direction = aortic_valve_direction_projected_on_sax_plane / np.linalg.norm(aortic_valve_direction_projected_on_sax_plane)

        print(self.center)

        # # Initialize landmark lists for averaging across series
        # self.vpc, self.sax_normal, self.rv_center  = [], [], []
        
        # # Process each SAX series
        # for series in self:
        #     if series.view != 'SAX':
        #         continue
                
        #     if series_to_use != 'all' and series.name not in series_to_use:
        #         continue

        #     # Calculate valve plane center
        #     if series.slice_above_valveplane is None:
        #         # Use first slice if valve plane not estimated
        #         self.vpc.append(np.mean(series.XYZs[0], axis=0) - self.center)
        #     else:
        #         # Use first slice within LV
        #         for k in range(len(series.XYZs)):
        #             if (not series.slice_above_valveplane[0, k] and 
        #                 not np.sum(series.data[0, k]) == 0):
        #                 self.vpc.append(np.mean(series.XYZs[k], axis=0) - self.center)
        #                 break

        #     # Fallback if no suitable slice found
        #     if len(self.vpc) == 0:
        #         self.vpc.append(np.mean(series.XYZs[0], axis=0) - self.center)

        #     # Calculate short-axis normal vector
        #     X_xyz = series.orientation[3:]
        #     Y_xyz = series.orientation[:3]
        #     self.sax_normal.append(np.cross(Y_xyz, X_xyz))

        #     #get center of RV (relative to self.center):
        #     rv_xyzs = []
        #     for j in range(s.prepped_seg.shape[0]):
                
        #         RV = (s.prepped_seg[j] == 1)
                
        #         if np.sum(RV) == 0:
        #             continue

        #     for i in range(len(RV)):

        #         if s.slice_above_valveplane is not None and s.slice_above_valveplane[j,i] == 0:
        #             continue
                
        #         rv_xyzs.append(s.XYZs[i].reshape((self.sz,self.sz,3))[RV[i]==1])

        #     if len(rv_xyzs) > 0:
        #         rv_xyzs = np.concatenate(rv_xyzs,axis=0)
        #         rv_center = np.mean(rv_xyzs,axis=0) - self.center
        #         self.rv_center.append( rv_center )

        # # Average landmarks across series
        # if len(self.vpc) > 0:
        #     self.vpc = np.mean(self.vpc, axis=0)
        #     self.sax_normal = np.mean(self.sax_normal, axis=0)
        #     self.rv_center = np.mean(self.rv_center, axis=0)
        # else:
        #     print('WARNING: No SAX slices found for landmark calculation')
        #     self.vpc, self.sax_normal, self.rv_center, self.rv_direction = None, None, None, None
        #     return

        # # Calculate base center and distances from center
        # max_dist_from_center = 0
        # self.base_center = None
        
        # for series in self:
        #     if series.view != 'SAX':
        #         continue
                
        #     series.distance_from_center = []
            
        #     for k in range(len(series.XYZs)):
        #         # Center of current slice
        #         slice_center = np.mean(series.XYZs[k], axis=0)
                
        #         # Normalize SAX normal vector
        #         n = self.sax_normal / np.linalg.norm(self.sax_normal, 2)
                
        #         # Project slice center onto SAX normal line
        #         intersection_point = (self.center + 
        #                             n * np.dot(slice_center - self.center, n))
                
        #         # Calculate distance from center along normal
        #         dist_from_center = np.mean((intersection_point - self.center) / 
        #                                  self.sax_normal)

        #         series.distance_from_center.append(dist_from_center)
                
        #         # Track most basal slice
        #         if dist_from_center > max_dist_from_center:
        #             self.base_center = slice_center
        #             max_dist_from_center = dist_from_center

        # # Set default base center if none found
        # if self.base_center is None:
        #     for series in self:
        #         if series.view == 'SAX':
        #             self.base_center = np.mean(series.XYZs[0], axis=0)
        #             break

        # # Recursive call with center adjustment if needed
        # if center_shift is None:
        #     adjustment = self.sax_normal * (58 - max_dist_from_center)
        #     self.estimate_landmarks(series_to_use=series_to_use, 
        #                           center_shift=adjustment, init_mode=init_mode)
        #     return

        # rv_direction = self.rv_center / np.linalg.norm(self.rv_center)
        # rv_direction_projected_on_sax_plane = rv_direction - np.dot(rv_direction, self.sax_normal) * self.sax_normal
        # self.rv_direction = rv_direction_projected_on_sax_plane / np.linalg.norm(rv_direction_projected_on_sax_plane)

        # # Predict aortic valve position and direction
        # self.predict_aortic_valve_position()
        # if self.valve_center is not None:
        #     self.valve_center = self.valve_center - self.center
            
        #     # Calculate aortic valve direction projected onto SAX plane
        #     aortic_direction = self.valve_center / np.linalg.norm(self.valve_center)
        #     aortic_proj = (aortic_direction - 
        #                   np.dot(aortic_direction, self.sax_normal) * self.sax_normal)
        #     self.aortic_valve_direction = aortic_proj / np.linalg.norm(aortic_proj)