import pydicom
import os

path_to_folder = "/data.lfpn/ibraun/Code/paper_volume_calculation/Patient_data/Heart_failure_infarct/SCD0000101/CINESAX_300"  # or unnamed, etc.
for file in os.listdir(path_to_folder):
    if file.endswith(".dcm"):
        dcm = pydicom.dcmread(os.path.join(path_to_folder, file))
        print(f"Series: {dcm.SeriesDescription}")
        print(f"Orientation: {dcm.ImageOrientationPatient}")
        break