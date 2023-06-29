
import numpy as np
import cv2
import os
from scipy.ndimage import label
from scipy import ndimage
from PIL import Image
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter
import math
from skimage.filters import median, gaussian, sobel
from skimage.morphology import disk, binary_closing, skeletonize, binary_opening, binary_erosion, white_tophat
from skimage.filters import threshold_li
from skimage.segmentation import watershed, find_boundaries
from skimage.exposure import equalize_adapthist
from skimage.measure import regionprops, regionprops_table
import pandas as pd
from cellpose import models, utils
import tensorflow as tf
from tensorflow.keras import backend as K
import pickle
from skimage.transform import rescale, resize, downscale_local_mean
from deepcell.applications import NuclearSegmentation


class ImageAnalyzer(object):
    
#     with open('/data2/test_images/for_training/trained_model1.pickle', 'rb') as handle:  ### Random forest model
#     with open('/data2/test_images/for_training/all_trained_models/Nearest_Neighbors.pickle', 'rb') as handle:  ### Nearest Neighbors model    
        
#     with open("/data2/test_images/for_training/all_trained_models/NeuralNet_deep.pickle", 'rb') as handle: ### neural netowork model (8,16,32,64,32,16,8)
#         model1 = pickle.load(handle)

    def __init__(self, data_path):
        #spot_params_dict =self.INITIALIZE_SPOT_ANALYSIS_PARAMS()
        #for now, I will initialize it to the defaults
        filepath = os.path.join(data_path,'input_template.xlsx')
        print(filepath)
        #mydoc = minidom.parse(filepath) xml only
        #maybe create an excel template where you only have limited choices similar to a gui? a drop down menu
        self.input_params = pd.read_excel(filepath)
        self.input_params = self.input_params['Col'].tolist()
        print(self.input_params)
        PATH_TO_FILES = os.path.split(filepath)[0]
        self.spot_params_dict = self.INITIALIZE_SPOT_ANALYSIS_PARAMS()
    
    def nuclei_segmenter(self, input_img, pixpermic=None):
        
        #self.AnalysisGui.NucDetectMethod.currentText()
        print(self.input_params[5])
        if self.input_params[5] == "Int.-based":
            
            first_thresh = self.input_params[6]*2.55 #separationslider
            second_thresh = 255-(self.input_params[7]*2.55) #detectionslider
            Cell_Size = self.input_params[8]
            
            boundary, mask = self.segmenter_function(input_img, cell_size=Cell_Size, 
                                                     first_threshold=first_thresh, second_threshold=second_thresh)
            
        if self.input_params[5] == "Marker Controlled":
            pixpermic=1
          
            Cell_Size = self.input_params[8]
            max_range = np.sqrt(Cell_Size/3.14)*2/float(pixpermic)
            nuc_detect_sldr = self.input_params[7]
            first_thresh = np.ceil((1-(nuc_detect_sldr/100))*max_range).astype(int)
            
            second_thresh = self.input_params[6]
            
            print(Cell_Size, first_thresh, second_thresh)
            boundary, mask = self.watershed_scikit(input_img, cell_size=Cell_Size, 
                                                     first_threshold=first_thresh, second_threshold=second_thresh)

        if self.input_params[5] == "CellPose-GPU":
            pixpermic = 0.1 #commented out earlier, why?
            Cell_Size = self.input_params[8]
            cell_diameter = np.sqrt(Cell_Size/(float(pixpermic)*float(pixpermic)))*2/3.14
            
            boundary, mask = self.cellpose_segmenter(input_img, use_GPU=1, cell_dia=cell_diameter)
            
        if self.input_params[5] == "CellPose-CPU":
            
            Cell_Size = self.input_params[8]
            cell_diameter = np.sqrt(Cell_Size/(float(pixpermic)*float(pixpermic)))*2/3.14
            
            boundary, mask = self.cellpose_segmenter(input_img, use_GPU=0, cell_dia=cell_diameter)
            
        if self.input_params[5] == "CellPose-Cyto":
            
            Cell_Size = self.input_params[8]
            cell_diameter = np.sqrt(Cell_Size/(float(pixpermic)*float(pixpermic)))*2/3.14
            
            boundary, mask = self.cellpose_segmenter(input_img, use_GPU=1, cell_dia=cell_diameter)
                
        if self.input_params[5] == "DeepCell":
            
            Cell_Size = self.input_params[8]
            cell_diameter = Cell_Size/100
            
            boundary, mask = self.deepcell_segmenter(input_img, cell_dia=cell_diameter)
        
#         kernel = np.ones((7,7), np.uint8)
#         first_pass = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         boundary = find_boundaries(first_pass, connectivity=1, mode='thick', background=0)
#         mask= (255*first_pass).astype('uint8')
#         boundary= (255*boundary).astype('uint8')
        #ask for an output directory to return the file
        return boundary, mask   
    def INITIALIZE_SPOT_ANALYSIS_PARAMS(self):
        #change so that it initializes from the input file, ask for an input file path?
        #spotanalysismethod =  detection method = nuclei channel? [0]
        #thresholdmethod is as read [1]
        #thresholdslider=threshold value (goes from 0 to 100) [2]
        #sensitivityspinbox = kernel size? (goes from 1 to 9) [3]
        #spotsperchspinbox = spots/channel [4]

        self.spot_params_dict={
            
            "Ch1": np.array([self.input_params[0],self.input_params[1],
                    self.input_params[2], self.input_params[3],
                            self.input_params[4]]),
            "Ch2": np.array([self.input_params[0],self.input_params[1],
                    self.input_params[2], self.input_params[3],
                            self.input_params[4]]),
            "Ch3": np.array([self.input_params[0],self.input_params[1],
                    self.input_params[2], self.input_params[3],
                            self.input_params[4]]),
            "Ch4": np.array([self.input_params[0],self.input_params[1],
                    self.input_params[2], self.input_params[3],
                            self.input_params[4]]),
            "Ch5": np.array([self.input_params[0],self.input_params[1],
                    self.input_params[2], self.input_params[3],
                            self.input_params[4]])
            }
        return self.spot_params_dict
    def segmenter_function(self, input_img, cell_size=None, first_threshold=None, second_threshold=None):
    
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        
        
        ## First blurring round
        if (cell_size %2)==0:
            cell_size = cell_size + 1
        median_img = cv2.medianBlur(img_uint8,cell_size)
        gaussian_blurred = cv2.GaussianBlur(median_img,(5,5),0)
        ## Threhsolding and Binarizing
        ret, thresh = cv2.threshold(gaussian_blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        bin_img = (1-thresh/255).astype('bool')
        ## Binary image filling
        filled = ndimage.binary_fill_holes(bin_img)
        filled_int= (filled*255).astype('uint8')
        ## Gray2RGB to feed the watershed algorithm
        img_rgb  = cv2.cvtColor(img_uint8,cv2.COLOR_GRAY2RGB)
        boundary = img_rgb
        boundary = boundary - img_rgb
        ## Distance trasform and thresholing to set the watershed markers
        dt = cv2.distanceTransform(filled.astype(np.uint8), 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
        _, dt = cv2.threshold(dt, first_threshold, 255, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (255 / (ncc + 1))
        lbl = lbl.astype(np.int32)
        ## First round of Watershed transform
        cv2.watershed(img_rgb, lbl)
        ## Correcting image boundaries
        boundary[lbl == -1] = [255,255,255]
        boundary[0,:] = 0
        boundary[-1,:] = 0
        boundary[:,0] = 0
        boundary[:, -1] = 0
        b_gray = cv2.cvtColor(boundary,cv2.COLOR_BGR2GRAY)
        diff = filled_int-b_gray

        kernel = np.ones((11,11), np.uint8)
        first_pass = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

        ## Second round of marker generation and watershed 
        kernel = np.ones((5,5),np.uint8)
        aa = first_pass.astype('uint8')
        erosion = cv2.erode(aa,kernel,iterations = 1)
        kernel = np.ones((11,11), np.uint8)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        blur = cv2.GaussianBlur(opening,(11,11),50)
        ret2, thresh2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        dt = cv2.distanceTransform(255-thresh2, 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
        _, dt = cv2.threshold(dt, second_threshold, 255, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (255 / (ncc + 1))
        lbl = lbl.astype(np.int32)
        cv2.watershed(img_rgb, lbl)
        ########
        boundary = img_rgb
        boundary = boundary - img_rgb

        boundary[lbl == -1] = [255,255,255]
        boundary_img = boundary[3:boundary.shape[0]-3,3:boundary.shape[1]-3]
        bound_gray = cv2.cvtColor(boundary_img,cv2.COLOR_BGR2GRAY)
        resized_bound = cv2.resize(bound_gray,(input_img.shape[1],input_img.shape[0]))

        kernel = np.ones((3,3),np.uint8)
        boundary = cv2.dilate(resized_bound,kernel,iterations = 1)
        filled1 = ndimage.binary_fill_holes(boundary)
        fin= 255*filled1-boundary
        mask = ndimage.binary_fill_holes(fin)
        mask = (255*mask).astype(np.uint8)

        return boundary, mask

    def watershed_scikit(self, input_img, cell_size=None, first_threshold=None, second_threshold=None):
        
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        
        med_scikit = median(img_uint8, disk(1))
        thresh = threshold_li(med_scikit)
        binary = med_scikit > thresh
        filled = ndimage.binary_fill_holes(binary)
        filled_blurred = gaussian(filled, 1)
        filled_int= (filled_blurred*255).astype('uint8')
        
# #         edge_sobel = sobel(img_uint8)
# #         enhanced = 50*edge_sobel/edge_sobel.max() + img_uint8
# #         enhanced.astype('uint8')
# #         med_scikit = median(img_uint8, disk(5))
        thresh = threshold_li(filled_int)
        binary = filled_int > thresh
        filled = ndimage.binary_fill_holes(binary)
        filled_int = binary_opening(filled, disk(5))
        filled_int = ndimage.binary_fill_holes(filled_int)
#         filled_blurred = gaussian(openeed, 3)
        
#         thresh = threshold_li(filled_int)
#         binary = filled_int > thresh
        #binary = binary_erosion(filled_int, disk(5))
        distance = ndimage.distance_transform_edt(filled_int)
        binary1 = distance > first_threshold
        distance1 = ndimage.distance_transform_edt(binary1)
        binary2 = distance1 > second_threshold

        labeled_spots, num_features = label(binary2)
        spot_labels = np.unique(labeled_spots)    

        spot_locations = ndimage.measurements.center_of_mass(binary2, labeled_spots, spot_labels[spot_labels>0])

        mask = np.zeros(distance.shape, dtype=bool)
        if spot_locations:
            mask[np.ceil(np.array(spot_locations)[:,0]).astype(int), np.ceil(np.array(spot_locations)[:,1]).astype(int)] = True
        markers, _ = ndimage.label(mask)
        labels = watershed(-distance, markers, mask=binary, compactness=0.5, watershed_line=True)
        boundary = find_boundaries(labels, connectivity=1, mode='thick', background=0)
        boundary_img = (255*boundary[3:boundary.shape[0]-3,3:boundary.shape[1]-3]).astype('uint8')
        resized_bound = cv2.resize(boundary_img,(input_img.shape[1],input_img.shape[0]))
        filled1 = ndimage.binary_fill_holes(resized_bound)
        
        mask= (255*filled1).astype('uint8')-resized_bound
        boundary= resized_bound.astype('uint8')
        
        return boundary, mask
    def cellpose_segmenter(self, input_img, use_GPU, cell_dia=None):
        
        if self.input_params[9] == True: #remove boundary
            img_uint8 = input_img
        else:
            img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)

        if self.input_params[5] == "CellPose-Cyto":
            model = models.Cellpose(gpu=use_GPU, model_type='cyto')
        else:
            model = models.Cellpose(gpu=use_GPU, model_type='nuclei')
        masks, flows, styles, diams = model.eval(img_uint8, diameter=cell_dia, flow_threshold=None)
                
        boundary = find_boundaries(masks, connectivity=1, mode='thick', background=0)

        if self.input_params[9] == True:

            boundary_img = (255*boundary).astype('uint8')
            filled1 = ndimage.binary_fill_holes(boundary_img)
            mask1= (255*filled1).astype('uint8')-boundary_img
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask1.astype('uint8'), cv2.MORPH_OPEN, kernel)
#             kernel = np.ones((1,1),np.uint8)
#             erosion = cv2.erode(mask,kernel,iterations = 1)

            
            boundary_img = find_boundaries(mask, connectivity=1, mode='thick', background=0)
            resized_bound = cv2.resize((255*boundary_img).astype('uint8'),(input_img.shape[1],input_img.shape[0]))
            filled1 = ndimage.binary_fill_holes(resized_bound)
        else:
            boundary_img = (255*boundary[3:boundary.shape[0]-3,3:boundary.shape[1]-3]).astype('uint8')
            resized_bound = cv2.resize(boundary_img,(input_img.shape[1],input_img.shape[0]))
            filled1 = ndimage.binary_fill_holes(resized_bound)
        
        mask1= (255*filled1).astype('uint8')-resized_bound
        kernel = np.ones((11,11), np.uint8)
        mask = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
        mask = 255*(mask==255).astype('uint8')
        mask[mask>0]=255
        boundary= resized_bound.astype('uint8')

        return boundary, mask
    
    def deepcell_segmenter(self, input_img, cell_dia=None):
        
        app = NuclearSegmentation()
        im = np.expand_dims(input_img, axis=-1)
        im = np.expand_dims(im, axis=0)
        
        masks1 = app.predict(im, image_mpp=cell_dia)
        masks = np.squeeze(masks1)
        boundary = find_boundaries(masks, connectivity=1, mode='thick', background=0)
        boundary_img = (255*boundary).astype('uint8')
        resized_bound = boundary_img
        filled1 = ndimage.binary_fill_holes(resized_bound)
        mask= (255*filled1).astype('uint8')-resized_bound
        boundary= resized_bound.astype('uint8')

        return boundary, mask