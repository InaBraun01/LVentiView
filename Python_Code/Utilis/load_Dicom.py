import os,sys
import pydicom
import numpy as np

def dataArrayFromDicom(PathDicom, multifile='unknown'):
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
        return dataArrayFromDicomFolder(PathDicom)
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

def dataArrayFromDicomFolder(PathDicom):
    """
    Load 4D image data (time, z, y, x) from a folder of DICOM files.

    Parameters:
        PathDicom (str): Path to the DICOM directory.

    Returns:
        tuple: image data array, pixel spacing, image IDs, metadata dictionary,
               slice locations, trigger times, image positions, is3D flag, multifile flag.
    """
    lstFilesDCM = get_sorted_dicom_filenames(PathDicom)  # get all DICOM filenames in the folder

    # Try to read a reference DICOM file that contains pixel spacing
    for i, f in enumerate(lstFilesDCM):
        try:
            RefDs = pydicom.read_file(lstFilesDCM[i], force=True)
            RefDs.PixelSpacing != None
            break
        except:
            pass

    # Metadata dictionary
    dicom_dir_details = {
        'SliceLocation': RefDs.get('SliceLocation', '?'),
        'InstanceNumber': RefDs.get('InstanceNumber', '?'),
        'ImageSize': RefDs.pixel_array.shape,
        'ImagePosition': RefDs.get('ImagePositionPatient', '?'),
        'ImageOrientation': RefDs.get('ImageOrientationPatient', '?'),
        'PatientPosition': RefDs.get('PatientPosition', '?'),
        'X,Y PixelSpacing': RefDs.get('PixelSpacing', '?'),
        'Z PixelSpacing': RefDs.get('SpacingBetweenSlices', '?'),
    }

    # Extract pixel spacing (Z, Y, X)
    ConstPixelSpacing = (
        float(RefDs.SpacingBetweenSlices),
        float(RefDs.PixelSpacing[0]),
        float(RefDs.PixelSpacing[1])
    )

    # Collect unique slice locations and trigger times
    slice_locations, trigger_times = [], []
    for filenameDCM in lstFilesDCM:
        ds = pydicom.read_file(filenameDCM, force=True)
        location, t_time = ds.get('SliceLocation', '?'), ds.get('TriggerTime', '?')
        if location != '?' and t_time != '?':
            slice_locations.append(location)
            trigger_times.append(t_time)

    slice_locations = sorted(set(slice_locations))  # unique and sorted
    trigger_times = sorted(set(trigger_times))      # unique and sorted

    # Create empty array to hold the image data
    data = np.zeros(
        (len(trigger_times), len(slice_locations), int(RefDs.Rows), int(RefDs.Columns)),
        dtype=RefDs.pixel_array.dtype
    )
    placment = np.zeros((len(trigger_times), len(slice_locations)))
    image_ids = np.zeros((len(trigger_times), len(slice_locations)))
    image_positions = [None for _ in range(len(slice_locations))]

    # Fill data array with image slices
    for i, filenameDCM in enumerate(lstFilesDCM):
        ds = pydicom.read_file(filenameDCM, force=True)
        location, t_time = ds.get('SliceLocation', '?'), ds.get('TriggerTime', '?')
        if location != '?' and t_time != '?':
            z = slice_locations.index(location)
            t = trigger_times.index(t_time)
            if ds.pixel_array.shape == data[t, z].shape:
                data[t, z] = ds.pixel_array  # Store image pixel values
            placment[t, z] = 1
            image_ids[t, z] = i
            image_positions[z] = ds.get('ImagePositionPatient', '?')  # Position of patient

    # Merge adjacent timeframes with poor spatial coverage
    i = 0
    while i < data.shape[0] - 1:
        if np.max((placment[i] > 0).astype(int) + (placment[i + 1] > 0).astype(int)) <= 1:
            # Merge and remove the weak timeframe
            data = np.concatenate([data[:i], data[i + 1:i + 2] + data[i:i + 1], data[i + 2:]], axis=0)
            placment = np.concatenate([placment[:i], placment[i + 1:i + 2] + placment[i:i + 1], placment[i + 2:]], axis=0)
            image_ids = np.concatenate([image_ids[:i], image_ids[i + 1:i + 2] + image_ids[i:i + 1], image_ids[i + 2:]], axis=0)
        else:
            i += 1

    # Placeholder: This data is treated as 4D even though it's likely not volumetric over time
    is3D = False
    multifile = True


    # remove_indices, remove_slices = [32,39],[2,6]
    # remove_zstacks = True

    # # Use a list comprehension to create a mask of indices to keep
    # keep_indices = [i for i in range(data.shape[0]) if i not in remove_indices]

    # # Create a new array with only the desired time frames
    # filtered_data = data[keep_indices, ...]

    # if remove_zstacks == True:
    #     filtered_data = filtered_data[:, remove_slices[0]:-remove_slices[1], :, :]

    # data = filtered_data

    # print(data.shape)
    # sys.exit()

    return data, ConstPixelSpacing, image_ids, dicom_dir_details, slice_locations, trigger_times, image_positions, is3D, multifile

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
