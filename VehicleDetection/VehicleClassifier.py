# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 15:32:54 2018

@author: ahmed
"""

import cv2
import numpy as np
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label

from VehicleDetection.lesson_functions import extract_features, slide_window, draw_boxes, get_hog_features, bin_spatial, color_hist, convert_color
from VehicleDetection.heat_map import add_heat, apply_threshold, draw_labeled_bboxes

import itertools
import multiprocessing as mp

class VehicleClassifier:
    def __init__(self, clsfr=None, scaler=None):
        self.classifier = clsfr
        self.X_scaler = scaler
        
        # Set hyper parameters
        self.color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9 # HOG orientations
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32) # Spatial binning dimensions
        self.hist_bins = 32    # Number of histogram bins
        self.spatial_feat = True # Spatial features on or off
        self.hist_feat = True # Histogram features on or off
        self.hog_feat = True # HOG features on or off
        self.y_start_stop = [350, 680] #544] # Min and max in y to search in slide_window()
        #self.x_start_stop = [0,800]
        self.svm_C = 0.005
        self.test_size = 0.2
        
    def getRawData(self, car_image_files, noncar_image_files):
        # Read in car and non-car images
        car_images = []
        noncar_images = []
        
        for i in range(0, len(car_image_files)):
            car_images.append(glob.glob(car_image_files[i]))
        
        for i in range(0, len(noncar_image_files)):
            noncar_images.append(glob.glob(noncar_image_files[i]))
            
        return list(np.concatenate(car_images)), list(np.concatenate(noncar_images))

    def extractFeatures(self, car_images, noncar_images):
        cars = []
        notcars = []
        
        for image in noncar_images:
            notcars.append(image)
        
        for image in car_images:
            cars.append(image)

        car_features = extract_features(cars, color_space=self.color_space, 
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        
        notcar_features = extract_features(notcars, color_space=self.color_space, 
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=rand_state)
            
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = self.X_scaler.transform(X_train)
        X_test = self.X_scaler.transform(X_test)
        
        print('Using:',self.orient,'orientations', self.pix_per_cell,
            'pixels per cell and', self.cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))
        return X_train, y_train, X_test, y_test

    def extractFeaturesNoSplit(self, images, cars=True):
        objects = []
        for image in images:
            objects.append(image)

        features = extract_features(objects, color_space=self.color_space, 
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                                orient=self.orient, pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        
        # Create an array stack of feature vectors
        X = np.array(features).astype(np.float64)
        y = None
        # Define the labels vector
        if(cars==True):
            y = np.ones(len(features))
        else:
            y = np.zeros(len(features))
        
        # Apply the scaler to X
        X = self.X_scaler.transform(X)
        
        return X, y
        
    def train(self, car_image_files, noncar_image_files, extra_cars_train=None, extra_noncars_train=None):
        car_images, noncar_images = self.getRawData(car_image_files, noncar_image_files)
        X_train, y_train, X_test, y_test = self.extractFeatures(car_images, noncar_images)
        if(extra_cars_train is not None):
            X_train2, y_train2 = self.extractFeaturesNoSplit(extra_cars_train, cars=True)
            X_train = np.vstack((X_train, X_train2))
            y_train = np.hstack((y_train, y_train2))
        if(extra_noncars_train is not None):
            X_train2, y_train2 = self.extractFeaturesNoSplit(extra_noncars_train, cars=False)
            X_train = np.vstack((X_train, X_train2))
            y_train = np.hstack((y_train, y_train2))
            
        plt.hist(y_train)
        plt.show()
        plt.hist(y_test)
        plt.show()
        # Use a linear SVC 
        #self.classifier = LinearSVC(C=self.svm_C)
        self.classifier = AdaBoostClassifier(base_estimator=LinearSVC(C=self.svm_C), algorithm='SAMME', random_state=100)
        
        # Check the training time for the SVC
        t=time.time()
        self.classifier.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.classifier.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
    
    # Define a function to extract features from a single image window
    # This function is very similar to extract_features()
    # just for a single image rather than list of images
    def single_img_features(self, img):
        #1) Define an empty list to receive features
        #print("img.shape = ", img.shape)
        #cv2.imshow("test", img)
        #cv2.waitKey(0)
        img_features = []
        #2) Apply color conversion if other than 'RGB'
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else: feature_image = np.copy(img)      
        #3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=self.spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if self.hist_feat == True:
            hist_features = color_hist(feature_image, nbins=self.hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if self.hog_feat == True:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_f, hog_img = get_hog_features(feature_image[:,:,channel], 
                                        self.orient, self.pix_per_cell, self.cell_per_block, 
                                        vis=True, feature_vec=True)
                    hog_features.extend(hog_f)
                #print("single_img_features hog_features.shape = ", len(hog_features))
            else:
                hog_features, hog_img = get_hog_features(feature_image[:,:,self.hog_channel], self.orient, 
                            self.pix_per_cell, self.cell_per_block, vis=True, feature_vec=True)
                #print("single_img_features hog_features.shape = ", len(hog_features))
            #8) Append features to list
            img_features.append(hog_features)
    
        #9) Return concatenated array of features
        return np.concatenate(img_features)
    
    
    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, args): #img, scale, y_range=[350,680], sframe_id=""):
        img, scale, y_range, x_range = args
        #draw_img = np.copy(img)
        #img = img.astype(np.float32)/255
        ystart = y_range[0] #self.y_start_stop[0] # 
        ystop = y_range[1] #self.y_start_stop[1] # 
        xstart = x_range[0]
        xstop = x_range[1]
        #feature_image = img[ystart:ystop,:,:]
        feature_image = img[ystart:ystop, xstart:xstop, :]
        #copy_img = img[ystart:ystop,:,:]
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                #feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2HSV)
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2HSV)
            elif self.color_space == 'LUV':
                #feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2LUV)
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2LUV)
            elif self.color_space == 'HLS':
                #feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2HLS)
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2HLS)
            elif self.color_space == 'YUV':
                #feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2YUV)
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2YUV)
            elif self.color_space == 'YCrCb':
                #feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2YCrCb)
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2YCrCb)
        if scale != 1:
            imshape = feature_image.shape
            feature_image = cv2.resize(feature_image, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            #copy_img = cv2.resize(copy_img, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = feature_image[:,:,0]
        ch2 = feature_image[:,:,1]
        ch3 = feature_image[:,:,2]
    
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        #nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = int(np.ceil(1*scale))  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        
        # Compute individual channel HOG features for the entire image
        hog1, hog_img1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False, vis=True)
        hog2, hog_img2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False, vis=True)
        hog3, hog_img3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False, vis=True)

        #hog_f, hog_img = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, vis=True, feature_vec=True)
        #print("feature_img shape = ", ch1.shape)
        on_windows = []
        #plt.subplots(12, 12, figsize=(15,15))
        plot_pos = 1
        for yb in range(nysteps):
            for xb in range(nxsteps):
                test_features = []
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                
                #hog_subimg = hog_img1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]
                
                '''
                print("hog_subimg.shape = ", hog_subimg.shape )
                plt.subplot(12,12,plot_pos)
                plt.imshow(hog_subimg)
                '''
                
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                #print("find_cars hog_features shape = ", hog_features.shape)
                
                #test_features.append(hog_features)
                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell
    
                # Extract the image patch
                subimg = cv2.resize(feature_image[ytop:ytop+window, xleft:xleft+window], (64,64))
                #cv2.rectangle(copy_img, (xleft,ytop),(xleft+window,ytop+window), (0,0,255), 2)
                #print("subimg.shape = ", subimg.shape)
                
                # Get color features
                if(self.spatial_feat==True):
                    spatial_features = bin_spatial(subimg, size=self.spatial_size)
                    #test_features.append(spatial_features)
                if(self.hist_feat==True):
                    hist_features = color_hist(subimg, nbins=self.hist_bins)
                    #test_features.append(hist_features)
                    
                # Scale features and make a prediction
                #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
                test_features = self.X_scaler.transform(test_features)
                
                test_prediction = self.classifier.predict(test_features)
                
                '''
                plt.subplot(12,12,plot_pos+1)
                plt.imshow(subimg)
                
                if(plot_pos==64):
                    #cv2.imwrite("../test_images/project_test01_car02_3.jpg", cv2.cvtColor(subimg, cv2.COLOR_YCrCb2BGR))
                    cv2.imshow("test", cv2.cvtColor(subimg, cv2.COLOR_YCrCb2BGR))
                    cv2.waitKey(0)
                '''
                
                if test_prediction == 1:
                    #print("test_prediction == 1")
                    #subimg = (subimg*255).astype('uint8')
                    #cv2.imwrite("test_images/noncar_" + sframe_id + "_" + str(scale) + "_" + str(plot_pos) +".png", cv2.cvtColor(subimg, cv2.COLOR_YCrCb2BGR))
                    #cv2.imshow("test", cv2.cvtColor(subimg, cv2.COLOR_YCrCb2BGR))
                    #cv2.waitKey(0)
                    
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                    #cv2.rectangle(copy_img, (xleft,ytop),(xleft+window,ytop+window), (0,255,0), 1)
                    on_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                plot_pos += 1
        '''
        plt.show()
        fig = plt.figure(figsize=(10,12))
        plt.subplot(1,1,1)
        plt.imshow(np.flip(copy_img, axis=2))
        plt.show()
        '''
        #return draw_img, on_windows
        return on_windows

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars2(self, img, scale):
        
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
        ystart = self.y_start_stop[0]
        ystop = self.y_start_stop[1]
        feature_image = img[ystart:ystop,:,:]
        copy_img = img[ystart:ystop,:,:]
        
        if scale != 1:
            imshape = feature_image.shape
            feature_image = cv2.resize(feature_image, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            copy_img = cv2.resize(copy_img, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = feature_image[:,:,0]
    
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        #nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2 #int(np.ceil(2*scale))  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        
        # Compute individual channel HOG features for the entire image
        #hog1, hog_img1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False, vis=True)
        #hog2, hog_img2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False, vis=True)
        #hog3, hog_img3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False, vis=True)
        #print("hog_img shape = ", hog_img1.shape)
        #print("feature_img shape = ", ch1.shape)
        on_windows = []
        #plt.subplots(nxsteps, nysteps, figsize=(15,15))
        plot_pos = 1
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                
                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell
    
                # Extract the image patch
                subimg = cv2.resize(feature_image[ytop:ytop+window, xleft:xleft+window], (64,64))
                #cv2.rectangle(copy_img, (xleft,ytop),(xleft+window,ytop+window), (0,0,255), 2)
                #print("subimg.shape, min, max = ", subimg.shape, np.min(subimg), np.max(subimg))
                subimg = (subimg*255).astype('uint8')
                
                test_features = self.single_img_features(subimg)
                test_features = self.X_scaler.transform(np.array(test_features).reshape(1, -1))
                test_prediction = self.classifier.predict(test_features)
                
                
                #cv2.imshow("test", subimg)
                #if cv2.waitKey(0) & 0xFF == ord('s'):
                #if(plot_pos==177 or plot_pos==178):
                #    filename = "C:/Yaser/Udacity/CarND-Term1/vehicle-detection/test_images/project_test01_car_00" + str(scale) + "_" + str(plot_pos) + ".png"
                #    cv2.imwrite(filename, subimg*255)
                
                #plt.subplot(nxsteps,nysteps,plot_pos)
                #plt.imshow(subimg)
                
                #print("prediction = ", test_prediction)   
                #print("type of prediction = ", type(test_prediction))
                if(test_prediction == 1):
                    #cv2.imshow("test", subimg)
                    #cv2.waitKey(0)
                    
                    xleft_scaled = np.int(xleft*scale)
                    ytop_scaled = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    '''
                    filename = "C:/Yaser/Udacity/CarND-Term1/vehicle-detection/test_images/project_test01_car_00" + str(scale) + "_" + str(xleft_scaled) + "_" + str(ytop_scaled) + ".png"
                    #cv2.imwrite(filename, subimg)
                    subimg2 = cv2.imread(filename)
                    test_features2 = self.single_img_features(subimg2)
                    test_features2 = self.X_scaler.transform(np.array(test_features2).reshape(1, -1))
                    test_prediction2 = self.classifier.predict(test_features2)
                    if(test_prediction2 == 1):
                        print(" * MATCH * ")
                        cv2.imshow("test", subimg2)
                        cv2.waitKey()
                    '''
                    #featuresFile = open("./featuresFile_02" + str(plot_pos) + "_" + str(scale) + ".pkl", "wb")
                    #pickle.dump(test_features, featuresFile)
                    #featuresFile.close()
                    #print("test_prediction == 1, plot_pos = ", plot_pos)
                    
                    cv2.rectangle(draw_img,(xleft_scaled, ytop_scaled+ystart),(xleft_scaled+win_draw,ytop_scaled+win_draw+ystart),(0,0,255),6) 
                    #cv2.rectangle(copy_img, (xleft,ytop),(xleft+window,ytop+window), (0,255,0), 1)
                    on_windows.append(((xleft_scaled, ytop_scaled+ystart),(xleft_scaled+win_draw,ytop_scaled+win_draw+ystart)))
                plot_pos += 1
        
        #plt.show()
        #fig = plt.figure(figsize=(10,12))
        #plt.subplot(1,1,1)
        #plt.imshow(np.flip(copy_img, axis=2))
        #plt.show()
        return draw_img, on_windows
    
    # Define a function you will pass an image 
    # and the list of windows to be searched (output of slide_windows())
    def scan_windows(self, img, windows):
    
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            #4) Extract features for that window using single_img_features()
            features = self.single_img_features(test_img)            
            #5) Scale extracted features to be fed to classifier
            test_features = self.X_scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = self.classifier.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows


    def identifyVehicles(self, frame, search_scales=[1.0, 1.5, 2.5], heatmap=None, heat_threshold=2, sframe_id="", visualize=0):
        #draw_image = np.copy(image)
        
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        #image = image.astype(np.float32)/255
        
        #xy_window = [64,64]
        #windows = slide_window(frame, x_start_stop=[None, None], y_start_stop=self.y_start_stop, 
                            #xy_window=xy_window, xy_overlap=(0.5, 0.5))
        
        #hot_windows = self.scan_windows(frame, windows)
        print(search_scales)
        on_windows = []
        #find_cars_nb = numba.jit(self.find_cars)
        y_range = [
                    [350,500], # for scale = 1
                    [350,512], # for scale = 1.5
                    [400,680], # for scale = 2.5
                    #[400,680] # for scale = 3.0
                   ]
        
        x_range = [
                    [400,1200], # for scale = 1
                    [400,1200], # for scale = 2
                    [0, frame.shape[1]], # for scale = 2.5
                    #[0, frame.shape[1]] # for scale = 3.0
                    ]
        
        i=0
        
        for scale in search_scales:
            #on_windows_img, windows = self.find_cars(frame, scale, y_range[i], sframe_id=sframe_id)
            windows = self.find_cars((frame, scale, y_range[i], x_range[i])) #, sframe_id=sframe_id)
            #on_windows_img, windows = find_cars_nb(frame, scale)
            #on_windows_img.append(img)
            on_windows.append(windows)
            #on_windows_img_list.append(on_windows_img)
            i += 1
            
        on_windows = list(itertools.chain(*on_windows))
        
        '''
        
        w = 3
        pool = mp.Pool(processes=w)
        result = pool.map(self.find_cars, [(frame, search_scales[0], y_range[0]), (frame, search_scales[1], y_range[1]), (frame, search_scales[2], y_range[2]), (frame, search_scales[3], y_range[3])])
        
        #for r in result:
        #    print(r)
            #on_windows_img_list.append(r[0])
        #    on_windows.append(r[0])
        
        if(len(result[0])!=0):
            on_windows = list(itertools.chain(*result))
        
        pool = None
        result = None
        '''
        
        #window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        #plt.figure(figsize=(10,10))
        #plt.imshow(window_img)
        #plt.imshow(hot_windows_img)
        
        #return hot_windows
        # Add heat to each box in box list
        if(heatmap is None):
            heatmap = np.zeros_like(frame[:,:,0]).astype(np.float)
        
        heatmap = add_heat(heatmap, on_windows)
        
        
        # Apply threshold to help remove false positives
        heatmap = apply_threshold(heatmap, heat_threshold) #1
        
        '''
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heatmap, 0, 255)
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img, hot_windows = draw_labeled_bboxes(np.copy(frame), labels)
        
        fig = plt.figure(figsize=(15,15))
        plt.subplot(151)
        plt.imshow(np.flip(draw_img, axis=2))
        plt.title('Car Positions')
        plt.subplot(152)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        '''
        
        if(visualize==1):
            on_windows_img = np.copy(frame)
            on_windows_img_list = []
            for win in on_windows:
                cv2.rectangle(on_windows_img, win[0], win[1], (0,0,255), 6)
            
            on_windows_img_list.append(on_windows_img)
            
            fig = plt.figure(figsize=(8,8))
            for j in range(len(on_windows_img_list)):
                plt.subplot(111+j)
                plt.imshow(np.flip(on_windows_img_list[j], axis=2))
                plt.title('On Windows')
                
            fig.tight_layout()
            plt.show()
        
        
        '''
        return draw_img, hot_windows, heatmap
        '''
        return heatmap

import pickle

def train(extra_noncars):
    vClsfr = VehicleClassifier()
    car_files = [
                 'C:\\Users\\ahmed\\Downloads\\vehicles\\GTI_Far\\*.png',
                 'C:\\Users\\ahmed\\Downloads\\vehicles\\GTI_Left\\*.png',
                 'C:\\Users\\ahmed\\Downloads\\vehicles\\GTI_MiddleClose\\*.png',
                 'C:\\Users\\ahmed\\Downloads\\vehicles\\GTI_Right\\*.png',
                 'C:\\Users\\ahmed\\Downloads\\vehicles\\KITTI_extracted\\*.png',
                 #'C:\\Users\\ahmed\\Downloads\\vehicles_smallset\\cars1\\*.jpeg',
                 #'C:\\Users\\ahmed\\Downloads\\vehicles_smallset\\cars2\\*.jpeg',
                 #'C:\\Users\\ahmed\\Downloads\\vehicles_smallset\\cars3\\*.jpeg'
                 ]

    noncar_files = [
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\GTI\\*.png',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\*.png',
                    #'C:\\Users\\ahmed\\Downloads\\non-vehicles_smallset\\notcars1\\*.jpeg',
                    #'C:\\Users\\ahmed\\Downloads\\non-vehicles_smallset\\notcars2\\*.jpeg',
                    #'C:\\Users\\ahmed\\Downloads\\non-vehicles_smallset\\notcars3\\*.jpeg'
                    ]
    
    
    extra_noncars2 = [
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra189.png',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra190.png',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra191.png',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra202.png',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra26.png',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra27.png',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra215.png',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra216.png',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra225.png',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra212.png'
                    ]
    
    extra_noncars2 = list(np.concatenate((extra_noncars2, extra_noncars))) #glob.glob("C:\\Yaser\\Udacity\\CarND-Term1\\vehicle-detection\\test_images\\project_test25_2_*.png"))))
    
    
    vClsfr.train(car_files, noncar_files, None, extra_noncars2)
    
    outfile = open("./VehicleDetection/classifier4.pkl", "wb")
    
    d = {'clsfr':vClsfr.classifier, 'scaler':vClsfr.X_scaler}
    pickle.dump(d, outfile)
    outfile.close()
    '''
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2771.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2790.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2792.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2793.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2794.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2802.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2803.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2804.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2808.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2826.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2829.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2830.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2832.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2833.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2834.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2836.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2837.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2838.png',
    'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra2846.png'
    '''

def getSavedVehicleClassifier():
    infile = open("./VehicleDetection/classifier4.pkl", "rb")
    objects_dict = pickle.load(infile, encoding='bytes')
    clsfr = objects_dict['clsfr']
    scaler = objects_dict['scaler']
    infile.close()
    vClsfr = VehicleClassifier(clsfr, scaler)
    return vClsfr

def testScanImage(filename='../test_images/project_test01.jpg', search_scales=[2.0,2.5], heat_threshold=1, sframe_id=""):
    vClsfr = getSavedVehicleClassifier()
    frame = cv2.imread(filename) #bbox-example-image.jpg')
    heatmap = np.zeros_like(frame[:,:,0])
    vClsfr.identifyVehicles(frame, heatmap=heatmap, heat_threshold=heat_threshold, sframe_id=sframe_id, visualize=1)
    #testIndividual(vClsfr, '../test_images/project_test01_car_0013_2.jpg')
    #testIndividual()
            
def testIndividual(vClsfr=None, imgFile=None):
    if(vClsfr is None):
        vClsfr = getSavedVehicleClassifier()
        
    frame = None
    if(imgFile is None):
        #filename = 'C:\\Users\\ahmed\\Downloads\\non-vehicles\\Extras\\extra202.png'
        filename = '../test_images/project_test01_1_217.png'
        
        frame = cv2.imread(filename)
    else:
        frame = cv2.imread(imgFile)
        
    print("frame shape, min, max = ", frame.shape, np.min(frame), np.max(frame))
    
    print("params: ")
    print(vClsfr.color_space)
    print(vClsfr.orient)
    print(vClsfr.pix_per_cell)
    print(vClsfr.cell_per_block)
    print(vClsfr.hog_channel)
    print(vClsfr.spatial_size)
    
    print(vClsfr.hist_bins)
    print(vClsfr.spatial_feat)
    print(vClsfr.hist_feat)
    print(vClsfr.hog_feat)
    print(vClsfr.y_start_stop)
    print(vClsfr.svm_C)
    print(vClsfr.test_size)
    
    features = vClsfr.single_img_features(frame)
    features = vClsfr.X_scaler.transform(np.array(features).reshape(1, -1))
    #featuresFile = open("./featuresFile_test.pkl", "wb")
    #pickle.dump(features, featuresFile)
    #featuresFile.close()
    pred = vClsfr.classifier.predict(features)
    print("prediction = ", pred)
    
def compareFeatureVectors():
    infile1 = open("./featuresFile_test.pkl", "rb")
    test_features1 = pickle.load(infile1, encoding='bytes')
    infile1.close()
    infile2 = open("./featuresFile_0215_3.pkl", "rb")
    test_features2 = pickle.load(infile2, encoding='bytes')
    infile2.close()
    
    if((test_features1 == test_features2).all()):
        print("Equal !??")
    else:
        print("Unequal... phewh!")
        print("shape (test_features1) = ", test_features1.shape)
        print("shape (test_features2) = ", test_features2.shape)
        print("Unequal elements = ", np.where(test_features1!=test_features2))
        print(test_features1[0,0])
        print(test_features2[0,0])

#train()
#testScanImage()
#testIndividual()
#compareFeatureVectors()