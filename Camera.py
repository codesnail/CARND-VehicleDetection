# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 18:54:30 2018

@author: ahmed
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import RoadFrame
import matplotlib.image as mpimg

class Camera:
    def __init__(self):
        self.calibMatrix = None
        self.distCoeff = None
        self.test_images = None
        self.test_image_idx = 0
        self.opMode = False
        self.perspectiveM = None
        self.perspectiveMInv = None
    
    def calibrate(self, image_path, filename_pattern, nx, ny, visualize=0):
        objp = np.zeros((ny*nx, 3), np.float32)
        objp[:,:2] = np.mgrid[0:ny,0:nx].T.reshape(-1,2)
        print(objp.shape)
        images = glob.glob(image_path+ "/" + filename_pattern)
        
        print("Calibrating... # images = ", len(images))
        objPoints = [] #np.zeros([len(images), objp.shape[0], objp.shape[1]])
        #print(objPoints.shape)
        imgPoints = [] #np.zeros([objPoints.shape[0], objPoints.shape[1], 1, 2])
        #print(imgPoints.shape)
        img_size = None
        
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if(img_size==None):
                img_size = gray.shape[::-1]
                
            ret, corners = cv2.findChessboardCorners(gray, (ny,nx), None)
            
            if(ret == True):
                print("Image # ", idx)
                #objPoints[idx] = objp
                objPoints.append(objp)
                #imgPoints[idx] = corners
                imgPoints.append(corners)
                
                if(visualize == True):
                    # Draw and display corners
                    cv2.drawChessboardCorners(img, (ny, nx), corners, ret)
                    plt.imshow(img)
                    
        
        # Perform camera calibration and set matrix and distortion coeff in class
        ret, self.calibMatrix, self.distCoeff, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, img_size, None, None)
        
        if(visualize > 0):
            # test an image
            test_img = cv2.imread('camera_cal/calibration1.jpg')
            undist = self.undistort(test_img)
            
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
            ax1.imshow(test_img)
            ax1.set_title('Original Image', fontsize=20)
            ax2.imshow(undist)
            ax2.set_title('Undistorted Image', fontsize=20)
            plt.show()
            
    def setPerspectiveTransform(self, img, src, dst, visualize=0):
        self.perspectiveImage = img
        self.perspectiveM = cv2.getPerspectiveTransform(src, dst)
        self.perspectiveMInv = cv2.getPerspectiveTransform(dst, src)
        if(visualize > 0):
            img_size = (img.shape[1], img.shape[0])
            warped = cv2.warpPerspective(img, self.perspectiveM, img_size, flags=cv2.INTER_LINEAR)
            
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,7))
            fig.tight_layout()
            ax1.imshow(img, interpolation='nearest', aspect='auto')
            ax1.plot(src[:,0],src[:,1], 'o', markersize=12)
            ax1.set_title('Original', fontsize=16)
            ax2.imshow(warped, interpolation='nearest', aspect='auto')
            ax2.plot(dst[:,0],dst[:,1], 'o', markersize=12)
            ax2.set_title('Warped', fontsize=16)
            plt.show()
        
    def undistort(self, img):
        return cv2.undistort(img, self.calibMatrix, self.distCoeff, None, self.calibMatrix)
    
    def turnOn(self):
        pass
    
    def setStaticTestImages(self, path, filename_pattern):
        self.test_images = glob.glob(path+ "/" + filename_pattern)
        
    def setOpMode(self, mode):
        self.opMode = mode
    
    def resetTestIdx(self):
        self.test_image_idx = 0
        
    def getNextFrame(self):
        frame = None
        ret = True
        roadFrame = None
        if(self.opMode==1): # test mode with static images
            if(self.test_image_idx > len(self.test_images)-1):
                return None, None
            else:
                frame = mpimg.imread(self.test_images[self.test_image_idx]) # "test_images/straight_lines1.jpg")
                self.test_image_idx += 1
        elif(self.opMode==2): # test mode with video
            ret, frame = self.getNextVideoFrame()
        
        if(ret == True):
            roadFrame = None
            if(self.calibMatrix is not None):
                undist = self.undistort(frame)
                roadFrame = RoadFrame.RoadFrame(undist)
            else:
                roadFrame = RoadFrame.RoadFrame(frame)
            
            roadFrame.setPerspectiveTransform(self.perspectiveM, self.perspectiveMInv)
        return ret, roadFrame
    
    def setVideoCapture(self, videoFile):
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass camera port number (e.g. 0) instead of the video file name
        self.cap = cv2.VideoCapture(videoFile)
         
        # Check if camera opened successfully
        if (self.cap.isOpened()== False): 
            print("Error opening video stream or file")
          
    def getNextVideoFrame(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
          
        return ret, frame
         
    
    def isVideoComplete(self):
        self.cap.isOpened()
        
    def relaseVideo(self):
        # When everything done, release the video capture object
        self.cap.release()
        