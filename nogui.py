import CellTrack_utils
import analysisNOGUI
import cv2
from PIL import Image
import numpy as np
dh=CellTrack_utils.data_handler() 

#data_path = 'C:\\Users\\tranne\\Downloads\\HiTIPS-main' # this is metadata directory
#data_path = '/home/tranne/Desktop/'
#out_df = dh.return_metadata_df(data_path)

#print(out_df)

#nuclei segmentation
#initialize the class
#nuc_data_path = "C:\\Users\\tranne\\Downloads\\HiTIPS-main"
nuc_data_path = "/data/tranne/hitips-nicole"
nuc=analysisNOGUI.ImageAnalyzer(nuc_data_path)
#call nuclear segmentation function (correctly spelled ver.)
img = Image.open('sample.tif')
img_arr = np.array(img)
print(img_arr)
ImageForNucMask = img_arr
normalized_nuc_img = cv2.normalize(ImageForNucMask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
print(normalized_nuc_img)
nuc_bndry, nuc_mask = nuc.nuclei_segmenter(input_img=normalized_nuc_img)
print("boundary and stuff")
print(nuc_bndry,nuc_mask)
print("done")


