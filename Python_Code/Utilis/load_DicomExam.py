import os
import sys
import pickle
from Python_Code.Utilis.folder_utils import path_leaf

def loadDicomExam(base_dir, output_folder='outputs', id_string=None):
    print("<<<< LOAD DICOM EXAM >>>>")
    '''loads a saved DicomExam (pass the same params as when creating the dicom exam)'''
    id_string = path_leaf(base_dir).split('.')[0] if id_string is None else id_string
    print(f"DicomExam id string: {id_string}")
    fname = os.path.join(output_folder, id_string+'', 'DicomExam.pickle')
    print(f"DicomExam id file name: {fname}")
    file_to_read = open(fname, "rb")
    de = pickle.load(file_to_read)
    file_to_read.close()
    return de