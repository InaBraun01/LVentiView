import os
import re
import ntpath


def generate_exam_folders(output_folder, id_string):
    """
    Generates a dictionary of output paths for an exam based on a predefined folder structure.

    Parameters:
        output_folder (str): Root output directory.
        id_string (str): Unique identifier for the exam.

    Returns:
        dict: Mapping from folder key names to full output paths.
    """
    base_path = os.path.join(output_folder, id_string)

    folder_structure = {
        'base': [],
        'mesh_plots': ['seg_analysis_data', 'meshes', 'plots'],
        'seg_plots': ['seg_analysis_data', 'segmentation', 'plots'],
        'seg_output_seg': ['seg_analysis_data', 'segmentation'],
        'seg_output_mesh': ['seg_analysis_data', 'meshes'],
        'meshes': ['meshes'],
        'mesh_segs': ['images', 'mesh_seg_images'],
        'initial_segs': ['images', 'initial_nn_seg_images'],
        'mesh_seg_uncertainty': ['mesh_seg_uncertainty'],
        'mesh_vol_plots': ['mesh_analysis_data', 'plots'],
        'meshes_output': ['mesh_analysis_data']
    }

    return {
        name: os.path.join(base_path, *subdirs)
        for name, subdirs in folder_structure.items()
    }


def sort_folder_names(iterable):
    """
    Sorts folder names in natural (human) order.

    Parameters:
        iterable (iterable): List of folder names.

    Returns:
        list: Naturally sorted list of folder names.
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split(r'([0-9]+)', key)]

    return sorted(iterable, key=alphanum_key)


def path_leaf(path):
    """
    Returns the last component of a path (file or folder name).

    Parameters:
        path (str): A file or folder path.

    Returns:
        str: The final component of the path.
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def create_output_folders(dicom_exam, folder_keys):
    """
    Creates output folders as specified in `dicom_exam.folder`.

    Parameters:
        dicom_exam: An object with a `folder` dictionary attribute.
        folder_keys (list): Keys in the `folder` dict for which folders should be created.
    """
    for key in folder_keys:
        path = dicom_exam.folder.get(key)
        if path and not os.path.exists(path):
            os.makedirs(path)