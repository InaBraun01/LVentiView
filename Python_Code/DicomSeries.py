"""
DICOM Series Management

This module provides a class for handling individual DICOM series within a cardiac MRI exam.
Each series represents a specific imaging sequence (e.g., short-axis, 2-chamber, 4-chamber views).
"""

import numpy as np
from Python_Code.Utilis.folder_utils import path_leaf
from Python_Code.Utilis.load_Dicom import dataArrayFromDicom, dataArrayFromNifti


class DicomSeries(object):
    """
    A class for representing a single DICOM series within a cardiac MRI exam.
    
    This class loads and manages DICOM data, automatically determining the cardiac view
    type and storing relevant metadata for further processing.
    
    Attributes:
        full_path (str): Path to the DICOM series folder
        name (str): Series identifier
        data (np.ndarray): Image data with shape (frames, slices, height, width)
        view (str): Cardiac view type (SAX, 2CH, 3CH, 4CH, or unknown)
        seg (np.ndarray): Segmentation mask (set later during processing)
        pixel_spacing (tuple): Pixel dimensions in mm
        orientation (np.ndarray): DICOM image orientation vectors
    """
    
    def __init__(self, full_path, id_string=None):
        """
        Initialize DICOM series from folder path.
        
        Args:
            full_path (str): Path to folder containing DICOM files
            id_string (str, optional): Custom identifier for the series
        """
        # Store path information
        self.full_path = full_path
        self.series_folder_name = path_leaf(full_path).lower().split('.')[0]
        self.name = self.series_folder_name if id_string is None else id_string

        if full_path.endswith('nii.gz'):
            print('loading series from NIfTI')

            (data, pixel_spacing, image_ids, dicom_details,
            slice_locations, trigger_times, image_positions,
            is3D, multifile, orientation) = dataArrayFromNifti(full_path)
            self.orientation = orientation
        
        else:
            # Load DICOM data and metadata
            (data, pixel_spacing, image_ids, dicom_details,
            slice_locations, trigger_times, image_positions,
            is3D, multifile) = dataArrayFromDicom(full_path)
        
            # Store DICOM metadata
            self.orientation = np.array(list(dicom_details['ImageOrientation']))
        
        self.data = data
        self.pixel_spacing = pixel_spacing
        self.image_ids = image_ids
        self.dicom_details = dicom_details
        self.slice_locations = slice_locations
        self.trigger_times = trigger_times
        self.image_positions = image_positions
        self.is3D = is3D
        self.multifile = multifile
        self.VP_heuristic1 = None
        
        # Initialize processing variables
        self.prepped_data = self.data  # Will be modified during preprocessing
        self.cleaned_data = self.data  #Will be modified as the data is cleaned
        self.frames = self.data.shape[0]  # Number of cardiac phases
        self.slices = self.data.shape[1]  # Number of image slices
        self.seg = None  # Segmentation mask (populated later)
        
        # Automatically determine cardiac view type
        self.guessView()
    
    def __str__(self):
        """
        Return string representation of the DICOM series.
        
        Returns:
            str: Formatted description including view type, shape, and processing status
        """
        details_str = f"{self.series_folder_name} ({self.view}), data shape = {self.data.shape}"
        
        # Add resampling information if data was modified
        if self.prepped_data.shape != self.data.shape:
            details_str += f" (resampled to {self.prepped_data.shape})"
        
        # Add segmentation status
        if self.seg is not None:
            details_str += " (has been segmented)"
            
        return details_str
    
    def guessView(self):
        """
        Automatically determine the cardiac view type from series name and properties.
        
        Uses heuristics based on:
        1. Folder name keywords (sax, 2ch, 3ch, 4ch)
        2. File extensions (.nii.gz, .npy assumed to be SAX)
        3. Number of slices (>3 slices typically indicates SAX)
        
        Sets self.view to one of: SAX, 2CH, 3CH, 4CH, unknown
        
        Returns:
            str: The determined view type
        """
        folder_name = self.series_folder_name
        
        # Check for explicit view indicators in folder name
        if 'sax' in folder_name or 'sa' in folder_name:
            self.view = 'SAX'
        elif 'lax' in folder_name or 'la' in folder_name:
            self.view = 'LAX'
        elif '2ch' in folder_name:
            self.view = '2CH'
        elif '3ch' in folder_name:
            self.view = '3CH'
        elif '4ch' in folder_name:
            self.view = '4CH'
        # Handle non-DICOM formats (typically preprocessed SAX data)
        elif 'nii.gz' in folder_name or '.npy' in folder_name:
            self.view = 'SAX'
        # Use slice count heuristic (SAX typically has multiple slices)
        elif self.data.shape[1] > 3:
            print(f"Processing data as SAX")
            self.view = 'SAX'
        else:
            self.view = 'unknown'
            
        return self.view



