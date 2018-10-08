# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 14:59:30 2018

@author: ahmed
"""
import matplotlib.pyplot as plt
import Camera
from VehicleDetection import VehicleClassifier
from VehicleDetection.lesson_functions import draw_boxes
from VehicleDetection.VehicleClassifier import train, testScanImage, testIndividual, compareFeatureVectors
import cv2
import glob
import numpy as np


def drawBoundingBoxes(frame, hot_windows):
    window_img = draw_boxes(frame, hot_windows, color=(0, 0, 255), thick=6)                    
        
    plt.figure(figsize=(10,10))
    plt.imshow(window_img)

def __main__():
    visualize = 1
    camera = Camera.Camera()
    camera.calibrate('camera_cal', 'calibration*.jpg', 6, 9, visualize=0)
    #camera.turnOn()
    camera.setOpMode(1)
    #camera.setStaticTestImages("test_images", "test*.jpg")
    camera.setStaticTestImages("test_images", "project_test0*.jpg") #"challenge_test0*.jpg")
    
    vClsfr = VehicleClassifier.VehicleClassifier()
    car_files = ['C:\\Users\\ahmed\\Downloads\\vehicles_smallset\\cars1\\*.jpeg',
                 'C:\\Users\\ahmed\\Downloads\\vehicles_smallset\\cars2\\*.jpeg'] #,
                 #'C:\\Users\\ahmed\\Downloads\\vehicles_smallset\\cars3\\*.jpeg']
    
    noncar_files = ['C:\\Users\\ahmed\\Downloads\\non-vehicles_smallset\\notcars1\\*.jpeg',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles_smallset\\notcars2\\*.jpeg'] #,
                    #'C:\\Users\\ahmed\\Downloads\\non-vehicles_smallset\\notcars3\\*.jpeg']
    
    vClsfr.train(car_files, noncar_files)
    
    while(True):
        _ , roadFrame = camera.getNextFrame()
        if(roadFrame == None):
            break
        draw_img, hot_windows = vClsfr.identifyVehicles(roadFrame.frame)
        if(visualize==1):
            drawBoundingBoxes(roadFrame.frame, hot_windows)

def saveExtras(ims, path, prefix, scale):
    i=0
    for im in ims:
        cv2.imwrite(path + prefix + str(i) + ".png", cv2.resize(im, (64,64)))
        i += 1
        
        
def extractNonCar(img, yrange=[400,680], xrange=[0,800], scale=1):
    noncar_imgs = []
    for y in range(yrange[0],yrange[1]-int(64*scale),16):
        for x in range(xrange[0],xrange[1]-int(64*scale),16):
            noncar_imgs.append(img[y:y+int(64*scale),x:x+int(64*scale)])
    return noncar_imgs


def getExtras(imgfile, yrange=[400,600], xrange=[0,800], scale=2):
    path = "C:\\Yaser\\Udacity\\CarND-Term1\\vehicle-detection\\test_images\\"
    img = cv2.imread('test_images/' + imgfile + ".png")
    extra_noncars = extractNonCar(img, yrange, xrange, scale=scale)
    filenamePattern = imgfile + "_" + str(scale) + "_"
    saveExtras(extra_noncars, path, filenamePattern, scale)
    files = glob.glob(path + filenamePattern + "_*.png")
    return files

if __name__ == '__main__':

    train()
    #imgfile = "project_test20"
    #files1 = getExtras(imgfile) 
    
    #imgfile = "project_test21"
    #files2 = getExtras(imgfile)
    '''
    files2 = glob.glob("test_images/project_test21_*.png")
    
    #imgfile = "project_test22"
    #files3 = getExtras(imgfile)
    
    #imgfile = "project_test24"
    #files4 = getExtras(imgfile)
    
    files4 = glob.glob("test_images/project_test23_*.png")
    
    imgfile = "project_test25"
    files5 = getExtras(imgfile)
    
    files6 = getExtras(imgfile, yrange=[550,680], xrange=[0,1000])
    
    files7 = glob.glob("test_images/project_test25_*.png")
    
    files8 = glob.glob("test_images/project_test28_*.png")
    
    files9 = glob.glob("test_images/noncar_*.png")
    
    files = list(np.concatenate((files2, files4, files5, files6, files7, files8, files9)))
    train(files)
    '''
    
    '''
    testScanImage('test_images/project_test21.png', heat_threshold=2, sframe_id="21_r6")
    testScanImage('test_images/project_test23.png', heat_threshold=2, sframe_id="23_r6")
    testScanImage('test_images/project_test24.png', heat_threshold=2, sframe_id="24_r6")
    testScanImage('test_images/project_test25.png', heat_threshold=2, sframe_id="25_r6")
    testScanImage('test_images/project_test28.png', heat_threshold=2, sframe_id="28_r7")
    
    
    #testScanImage('test_images/project_test21.png', heat_threshold=2, sframe_id="28_r7")
    testScanImage('test_images/project_trim_test30.png', heat_threshold=2, sframe_id="28_r7")
    testScanImage('test_images/project_trim1_test45.png', heat_threshold=2, sframe_id="28_r7")
    '''
    #testIndividual(imgFile='C:\\Users\\ahmed\\Downloads\\vehicles\\GTI_Right\\image0628.png') #'test_images/project_test01_2.0_82.png')
    #compareFeatureVectors()