import os,sys
import pydicom
import numpy as np


def classify_series_folder(folder):
    files = get_sorted_dicom_filenames(folder)
    ds = pydicom.read_file(files[0], force=True)
    
 
    slice_locations = ds.get('ImageOrientationPatient', '?')
    
    print(f"Folder: {os.path.basename(folder)}")
    print(f"  Orientation       : {slice_locations}")
    
    return 0

def dataArrayFromDicom(PathDicom, z_height_remove= None, time_frame_remove = None, multifile='unknown'):
    """
    Load image data from DICOM files in either single-file or multi-file format.

    Parameters:
        PathDicom (str): Path to the DICOM directory or file.
        multifile (bool or 'unknown'): If 'unknown', function will auto-detect format.

    Returns:
        tuple: Contains image data array, pixel spacing, image IDs, directory metadata,
               slice locations, trigger times, image positions, is3D flag, and multifile flag.
    """
    if multifile == 'unknown':
        
        lstFilesDCM = get_sorted_dicom_filenames(PathDicom)  # get list of sorted DICOM files

        if len(lstFilesDCM) >= 3:  # multiple images assumed
            multifile = True
        else:
            multifile = False

    if multifile == True:

        # # Sort folders by z position (ImagePositionPatient z-coordinate)
        # def get_z_position(folder):
        #     files = get_sorted_dicom_filenames(folder)
        #     ds = pydicom.read_file(files[0], force=True)
        #     return float(ds.ImagePositionPatient[2])

        # parent_folder = "/data/fpb/ibraun/Code/paper_volume_calculation/UMG_data/Test Scans/Normalbefund_mit_Perikarderguss/DICOM/0000ABDC/AAF0E4F5/AA0001D2"
        # #parent_folder = "/data/fpb/ibraun/Code/paper_volume_calculation/UMG_data/Test Scans/Schlechte_Quali_SAX/DICOM/00000895/AAB758C6/AAE57A14"

        # sax_folders = sorted([
        #     os.path.join(parent_folder, f) 
        #     for f in os.listdir(parent_folder) 
        #     if os.path.isdir(os.path.join(parent_folder, f))
        # ])

        # sax_folders_sorted = sorted(sax_folders, key=get_z_position)


        # return load_multi_folder_sax(sax_folders_sorted)
        return dataArrayFromDicomFolder(PathDicom, z_height_remove, time_frame_remove)
    
    elif multifile == False:

        return dataArrayFromDicomSingleFile(PathDicom)
    else:
        print('error, "multifile" should be True or False, got:', multifile)
        return None

def get_sorted_dicom_filenames(dicom_path):
    """
    Return a sorted list of valid DICOM file paths, excluding hidden and GIF files.

    Parameters:
        dicom_path (str): Path to the DICOM folder.

    Returns:
        list: Sorted list of valid DICOM file paths.
    """
    return sorted(
        os.path.join(dirpath, f)
        for dirpath, _, files in os.walk(dicom_path)
        for f in files
        if not f.startswith('.') and not f.endswith('.gif')
    )

import os
import numpy as np

def load_multi_folder_sax(sax_folders, z_height_remove=None, time_frame_remove=None):
    """
    Load a SAX series split across multiple folders (one z-slice per folder)
    and combine them into the standard (time, z, y, x) format.

    Parameters:
        sax_folders (list of str): Ordered list of folder paths, one per z-slice.
                                   Should be sorted in the correct z order (apex to base or vice versa).
    Returns:
        Same tuple as dataArrayFromDicomFolder
    """
    all_data, all_positions, all_locations = [], [], []

    for folder in sax_folders:
        data, ConstPixelSpacing, image_ids, dicom_dir_details, slice_locations, trigger_times, image_positions, is3D, multifile = \
            dataArrayFromDicomFolder(folder)
        # data shape: (time, 1, y, x)
        all_data.append(data)
        all_positions.append(image_positions[0])
        all_locations.append(slice_locations[0])

    # Stack along z axis → (time, z, y, x)
    combined_data = np.concatenate(all_data, axis=1)

    # Recalculate z spacing from ImagePositionPatient
    z_positions = sorted([float(pos[2]) for pos in all_positions])
    z_spacing = round(float(np.mean(np.diff(z_positions))), 4)
    ConstPixelSpacing = (z_spacing, ConstPixelSpacing[1], ConstPixelSpacing[2])

    combined_image_positions = all_positions
    combined_slice_locations = all_locations

    if z_height_remove:
        combined_data = np.delete(combined_data, z_height_remove, axis=1)
        combined_image_positions = [item for i, item in enumerate(combined_image_positions) if i not in z_height_remove]
        combined_slice_locations = [item for i, item in enumerate(combined_slice_locations) if i not in z_height_remove]

    if time_frame_remove:
        combined_data = np.delete(combined_data, time_frame_remove, axis=0)
    
    return combined_data, ConstPixelSpacing, image_ids, dicom_dir_details, combined_slice_locations, trigger_times, combined_image_positions, is3D, multifile

def dataArrayFromDicomFolder(PathDicom, z_height_remove=None, time_frame_remove=None):
    """
    Load 4D image data (time, z, y, x) from a folder of DICOM files.

    Parameters:
        PathDicom (str): Path to the DICOM directory.
        z_height_remove (list): Indices of slice locations to remove.
        time_frame_remove (list): Indices of time frames to remove.

    Returns:
        tuple: image data array, pixel spacing, image IDs, metadata dictionary,
               slice locations, trigger times, image positions, is3D flag, multifile flag.
    """
    import pandas as pd

    lstFilesDCM = get_sorted_dicom_filenames(PathDicom)

    # Try to read a reference DICOM file that contains pixel spacing
    for i, f in enumerate(lstFilesDCM):
        try:
            RefDs = pydicom.read_file(lstFilesDCM[i], force=True)
            RefDs.PixelSpacing != None
            break
        except:
            pass

    # Build records dataframe
    records = []
    for filenameDCM in lstFilesDCM:
        ds = pydicom.read_file(filenameDCM, force=True)
        location = ds.get('SliceLocation', '?')
        t_time   = ds.get('TriggerTime', '?')
        inst_num = ds.get('InstanceNumber', '?')
        series   = ds.get('SeriesNumber', '?')
        records.append((inst_num, series, location, t_time, filenameDCM))

    df = pd.DataFrame(records, columns=['InstanceNumber', 'SeriesNumber', 'location', 'ttime', 'file'])
    df = df[df['location'] != '?'].copy()
    df = df[df['ttime'] != '?'].copy()
    df = df.sort_values(['location', 'ttime']).reset_index(drop=True)

    # Assign phase index within each slice location
    df['phase'] = df.groupby('location').cumcount()

    n_phases        = int(df['phase'].max() + 1)
    slice_locations = sorted(df['location'].unique())
    trigger_times   = sorted(df['ttime'].unique())  # kept for return value compatibility

    print(f"Phases per slice:\n{df.groupby('location')['phase'].max() + 1}")
    print(f"Matrix: {n_phases} phases x {len(slice_locations)} slices")

    # Metadata dictionary
    dicom_dir_details = {
        'SliceLocation':    RefDs.get('SliceLocation', '?'),
        'InstanceNumber':   RefDs.get('InstanceNumber', '?'),
        'ImageSize':        RefDs.pixel_array.shape,
        'ImagePosition':    RefDs.get('ImagePositionPatient', '?'),
        'ImageOrientation': RefDs.get('ImageOrientationPatient', '?'),
        'PatientPosition':  RefDs.get('PatientPosition', '?'),
        'X,Y PixelSpacing': RefDs.get('PixelSpacing', '?'),
        'Z PixelSpacing':   RefDs.get('SliceThickness', '?'),
    }

    # Extract pixel spacing (Z, Y, X)
    ConstPixelSpacing = (
        float(RefDs.SliceThickness),
        float(RefDs.PixelSpacing[0]),
        float(RefDs.PixelSpacing[1])
    )

    # Create empty arrays
    data            = np.zeros((n_phases, len(slice_locations), int(RefDs.Rows), int(RefDs.Columns)), dtype=RefDs.pixel_array.dtype)
    placement       = np.zeros((n_phases, len(slice_locations)), dtype=int)
    image_ids       = np.zeros((n_phases, len(slice_locations)), dtype=int)
    image_positions = [None] * len(slice_locations)

    loc_index = {loc: i for i, loc in enumerate(slice_locations)}

    # Fill data array using phase index
    for _, row in df.iterrows():
        ds = pydicom.read_file(row['file'], force=True)
        z = loc_index[row['location']]
        t = int(row['phase'])
        if ds.pixel_array.shape == (int(RefDs.Rows), int(RefDs.Columns)):
            data[t, z]      = ds.pixel_array
            placement[t, z] = 1
            image_ids[t, z] = int(row['InstanceNumber']) if row['InstanceNumber'] != '?' else 0
            image_positions[z] = ds.get('ImagePositionPatient', '?')

    empty = (placement == 0).sum()
    print(f"Empty cells after loading: {empty} / {placement.size}")

    # Optional: remove specified z heights and time frames
    if z_height_remove:
        data            = np.delete(data, z_height_remove, axis=1)
        image_ids       = np.delete(image_ids, z_height_remove, axis=1)
        image_positions = [item for i, item in enumerate(image_positions) if i not in z_height_remove]
        slice_locations = [item for i, item in enumerate(slice_locations) if i not in z_height_remove]

    if time_frame_remove:
        data      = np.delete(data, time_frame_remove, axis=0)
        image_ids = np.delete(image_ids, time_frame_remove, axis=0)

    is3D      = False
    multifile = True

    # data = data[:, 2:, :, :]
    # image_ids = image_ids[:,2:]
    # slice_locations = slice_locations[2:]
    # image_positions = image_positions[2:]
    
    # #segmentation network for SAX is a 3D segmentation network, with max number of z heights input 16
    # # use only the top 16 z slices 
    # if data.shape[1] > 16:
    #     data = data[:, :16, :, :]
    #     image_ids = image_ids[:,:16]
    #     slice_locations = slice_locations[:16]
    #     image_positions = image_positions[:16]


    return data, ConstPixelSpacing, image_ids, dicom_dir_details, slice_locations, trigger_times, image_positions, is3D, multifile


# from collections import Counter


# def is_valid_mri_slice(ds, threshold=0.1):
#     """Filtert Junk-Dateien (Screenshots, Berichte, zu dunkle Bilder)."""
#     try:
#         # Filter nach DICOM-Typ
#         img_type = ds.get("ImageType", [])
#         if any(x in img_type for x in ['DERIVED', 'SECONDARY', 'SCREENSHOT']):
#             return False

#         # Filter nach Varianz (schließt fast schwarze Bilder aus)
#         if not hasattr(ds, 'pixel_array'):
#             return False
#         if np.std(ds.pixel_array) < threshold:
#             return False

#         return True

#     except:
#         return False


# def dataArrayFromDicomFolder(PathDicom, z_height_remove=None, time_frame_remove=None):
#     file_list = [os.path.join(PathDicom, f) for f in os.listdir(PathDicom) if f.lower().endswith('.dcm')]

#     all_data = []
#     series_counts = Counter()

#     print(f"Analysiere {len(file_list)} Dateien...")

#     # --- SCHRITT 1: Alle Metadaten lesen und Serie bestimmen ---
#     for f in file_list:
#         try:
#             ds = pydicom.read_file(f)
#             if is_valid_mri_slice(ds):
#                 # Wir merken uns die SeriesNumber (z.B. 6 oder 7)
#                 series_num = ds.get("SeriesNumber", "Unknown")
#                 loc = round(float(ds.get('SliceLocation', 0)), 2)
#                 t_val = float(ds.get('TriggerTime', ds.get('InstanceNumber', 0)))

#                 all_data.append({
#                     'file': f,
#                     'series': series_num,
#                     'loc': loc,
#                     'time': t_val,
#                     'pixel': ds.pixel_array,
#                     'pos': ds.get('ImagePositionPatient', [0, 0, 0])
#                 })
#                 series_counts[series_num] += 1
#         except:
#             continue

#     if not all_data:
#         print("Keine validen MRT-Daten gefunden.")
#         return None

#     # --- SCHRITT 2: Nur die Haupt-Serie behalten ---
#     # Die Serie mit den meisten Bildern ist unser 4D-MRT
#     main_series = series_counts.most_common(1)[0][0]
#     valid_data = [d for d in all_data if d['series'] == main_series]

#     print(f"Haupt-Serie identifiziert: {main_series} ({len(valid_data)} Bilder).")
#     print(f"Ignoriere andere Serien (z.B. { [s for s in series_counts if s != main_series] }).")

#     # --- SCHRITT 3: Gitter-Dimensionen bestimmen ---
#     unique_locs = sorted(list(set(d['loc'] for d in valid_data)))

#     # Zeitpunkte bestimmen: Pro Slice zählen wir die Bilder
#     # Da TriggerTimes leicht variieren können, sortieren wir pro Slice nach Zeit
#     # und weisen den Bildern Index 0, 1, 2... zu.
#     num_z = len(unique_locs)
#     num_t = len(valid_data) // num_z

#     print(f"Erstelle Gitter: {num_t} Zeitpunkte x {num_z} Slices.")

#     # --- SCHRITT 4: Array befüllen ---
#     h, w = valid_data[0]['pixel'].shape
#     data = np.zeros((num_t, num_z, h, w), dtype=valid_data[0]['pixel'].dtype)
#     image_ids = np.zeros((num_t, num_z))
#     image_positions = [None] * num_z

#     # Wir gruppieren die Daten nach Location
#     for z_idx, loc in enumerate(unique_locs):
#         # Alle Bilder für diese Schicht finden und nach Zeit sortieren
#         slice_images = sorted([d for d in valid_data if d['loc'] == loc], key=lambda x: x['time'])

#         # Falls eine Schicht weniger Bilder hat als andere (Lücke), nehmen wir was da ist
#         for t_idx, item in enumerate(slice_images):
#             if t_idx < num_t:
#                 data[t_idx, z_idx] = item['pixel']
#                 # ID aus Dateiname (z.B. 0459)
#                 try:
#                     fname = os.path.basename(item['file'])
#                     image_ids[t_idx, z_idx] = int(''.join(filter(str.isdigit, fname.split('-')[-1])))
#                 except:
#                     pass
#                 image_positions[z_idx] = item['pos']

#     # --- SCHRITT 5: Cleanup & Metadaten ---
#     if z_height_remove:
#         data = np.delete(data, z_height_remove, axis=1)
#         image_ids = np.delete(image_ids, z_height_remove, axis=1)
#         unique_locs = [v for i, v in enumerate(unique_locs) if i not in z_height_remove]
#         image_positions = [v for i, v in enumerate(image_positions) if i not in z_height_remove]

#     if time_frame_remove:
#         data = np.delete(data, time_frame_remove, axis=0)
#         image_ids = np.delete(image_ids, time_frame_remove, axis=0)

#     # Referenz-Metadaten vom ersten Bild der Hauptserie
#     ref_ds = pydicom.read_file(valid_data[0]['file'])
#     dicom_dir_details = {
#         'ImageOrientation': ref_ds.get('ImageOrientationPatient', [1, 0, 0, 0, 1, 0]),
#         'PixelSpacing': ref_ds.get('PixelSpacing', [1, 1]),
#         'Rows': h, 'Columns': w,
#         'SeriesNumber': main_series
#     }

#     z_sp = abs(unique_locs[1] - unique_locs[0]) if len(unique_locs) > 1 else 1.0
#     spacing = (float(z_sp), float(ref_ds.PixelSpacing[0]), float(ref_ds.PixelSpacing[1]))

#     return (data, spacing, image_ids, dicom_dir_details, unique_locs, list(range(data.shape[0])), image_positions, False, True)

def dataArrayFromDicomSingleFile(PathDicom):
    """
    Load image data from a single DICOM file.

    Parameters:
        PathDicom (str): Path to a DICOM file or a folder with one file.

    Returns:
        tuple: image data array, pixel spacing, None, metadata dictionary,
               None, None, None, is3D flag, multifile flag.
    """
    lstFilesDCM = get_sorted_dicom_filenames(PathDicom)

    # Use direct path or first file in folder
    if len(lstFilesDCM) == 0:
        f = pydicom.read_file(PathDicom)
    else:
        f = pydicom.read_file(lstFilesDCM[0])

    # Collect some header information
    dicom_dir_details = {
        'SliceLocation': f.get('SliceLocation', '?'),
        'InstanceNumber': f.get('InstanceNumber', '?'),
        'ImagePosition': f.get('ImagePositionPatient', '?'),
        'ImageOrientation': f.get('ImageOrientationPatient', '?'),
        'PatientPosition': f.get('PatientPosition', '?'),
        'PixelSpacing': f.get('PixelSpacing', '?'),
    }

    try:
        imgdata = f.pixel_array
    except:
        imgdata = np.zeros((1, 1, 1))  # Fallback for unreadable images

    # Print shape and header if valid
    if np.prod(imgdata.shape) > 1:
        print(imgdata.shape, dicom_dir_details)

    is3D = True
    multifile = False

    return imgdata[None], (0.45, 0.45, 0.45), None, dicom_dir_details, None, None, None, is3D, multifile


def dataArrayFromNifti(full_path):

    import nibabel as nib
    img = nib.load(full_path)
    hdr = img.header

    data = np.transpose(img.get_fdata(), (3,2,0,1))
    pixel_spacing = [hdr.get('pixdim')[3], hdr.get('pixdim')[1], hdr.get('pixdim')[2]]

    image_ids = None
    dicom_details = None
    slice_locations = [0 for k in range(data.shape[1])]
    # self.image_positions = [[0,0,-k*self.pixel_spacing[0]] for k in range(self.data.shape[1])]
    image_positions = [[0,0,k*pixel_spacing[0]] for k in range(data.shape[1])]
    trigger_times = None
    is3D = False
    multifile = None
    orientation = [0,1,0,1,0,0]

    return (data, pixel_spacing, image_ids, dicom_details, slice_locations, trigger_times, image_positions, is3D,multifile, orientation)
