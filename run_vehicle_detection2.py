# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 19:34:49 2018

@author: ahmed
"""

import cv2
import numpy as np
import Camera
from VehicleDetection import VehicleClassifier, lesson_functions, heat_map
from VehicleDetection.VehicleClassifier import getSavedVehicleClassifier
from VehicleDetection.lesson_functions import draw_boxes
import traceback
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt

def drawBoundingBoxes(frame, hot_windows):
    window_img = draw_boxes(frame, hot_windows, color=(0, 0, 255), thick=6)                    
    return window_img

def main():
    car_files = [#'C:\\Users\\ahmed\\Downloads\\vehicles\\GTI_Far\\*.png',
                 #'C:\\Users\\ahmed\\Downloads\\vehicles\\GTI_Left\\*.png',
                 #'C:\\Users\\ahmed\\Downloads\\vehicles\\GTI_MiddleClose\\*.png',
                 #'C:\\Users\\ahmed\\Downloads\\vehicles\\GTI_Right\\*.png',
                 #'C:\\Users\\ahmed\\Downloads\\vehicles\\KITTI_extracted\\*.png']
    
                'C:\\Users\\ahmed\\Downloads\\vehicles_smallset\\cars1\\*.jpeg',
                'C:\\Users\\ahmed\\Downloads\\vehicles_smallset\\cars2\\*.jpeg',
                'C:\\Users\\ahmed\\Downloads\\vehicles_smallset\\cars3\\*.jpeg']
    
    noncar_files = [#'C:\\Users\\ahmed\\Downloads\\non-vehicles\\GTI\\*.png',
    
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles_smallset\\notcars1\\*.jpeg',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles_smallset\\notcars2\\*.jpeg',
                    'C:\\Users\\ahmed\\Downloads\\non-vehicles_smallset\\notcars3\\*.jpeg']
    
           
    #vClsfr = VehicleClassifier.VehicleClassifier()         
    #vClsfr.train(car_files, noncar_files)
    
    vClsfr = getSavedVehicleClassifier()
    
    videoFile = "C:\\Yaser\\Udacity\\CarND-Term1\\vehicle-detection\\project_video_Trim2.mp4" #test_video.mp4" #
    outpath = "test_images"
    camera = Camera.Camera()
    camera.calibrate('camera_cal', 'calibration*.jpg', 6, 9, visualize=0)
    camera.setOpMode(2)
    camera.setVideoCapture(videoFile) #'challenge_video.mp4') #
    
    # Controls whether to just extract static images from video for development and testing,
    # or to actually process the video through the pipeline.
    mode = None #'extract' #
    
    #cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Original', 600,400)
    cv2.namedWindow('Cars', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cars', 600,400)
    
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # Convert the resolutions from float to integer.
    frame_width = int(camera.cap.get(3))
    frame_height = int(camera.cap.get(4))
     
    
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    #cv2.VideoWriter_fourcc('*divx')
    #out = cv2.VideoWriter('C:/Yaser/Udacity/CarND-Term1/project_video_processed.mp4', -1, 10, (frame_width,frame_height))
    out = cv2.VideoWriter("C:/Yaser/Udacity/CarND-Term1/vehicle-detection/project_video_out.avi", -1, 20, (frame_width,frame_height), isColor=True)

    frame_count = 0
    heatmap = None
    single_heat_threshold = 1 #2
    agg_heat_threshold = 12 #3
    heatmap_list = []
    frames_to_keep = 5
    try:
        while(True):
            frame_count += 1
            ret, roadFrame = camera.getNextFrame()
            if(ret == True):
                if(mode=='extract'): # Extract static images from video for development and testing
                    cv2.imshow('Cars', roadFrame.frame)
                    cv2.imwrite(outpath + "/project_trim1_test{:02d}".format(frame_count) + ".png", roadFrame.frame)
                    
                else: # Process video
                    #cv2.imshow('Original', roadFrame.frame)
                    
                    heatmap = vClsfr.identifyVehicles(roadFrame.frame, heat_threshold=single_heat_threshold)
                    
                    heatmap_list.append(heatmap)
                    
                    if(frame_count>frames_to_keep):
                       heatmap_list = heatmap_list[1:]
                    
                    heatmap = np.sum(heatmap_list, axis=0)
                    # Apply threshold to help remove false positives
                    heatmap = heat_map.apply_threshold(heatmap, agg_heat_threshold)
                    
                    # Visualize the heatmap when displaying    
                    heatmap = np.clip(heatmap, 0, 255)
                    
                    # Find final boxes from heatmap using label function
                    labels = label(heatmap)
                    vehicles_frame, hot_windows = heat_map.draw_labeled_bboxes(np.copy(roadFrame.frame), labels)
                    '''
                    fig = plt.figure(figsize=(15,15))
                    plt.subplot(121)
                    plt.imshow(np.flip(vehicles_frame, axis=2))
                    plt.title('Car Positions')
                    plt.subplot(122)
                    plt.imshow(heatmap, cmap='hot')
                    plt.title('Heat Map')
                    '''
                    
                    '''
                    for i in range(len(on_windows_img_list)):
                        plt.subplot(153+i)
                        plt.imshow(np.flip(on_windows_img_list[i], axis=2))
                        plt.title('On Windows')
                    '''
                    #fig.tight_layout()
                    #plt.show()
                    
                    #cars = drawBoundingBoxes(roadFrame.frame, hot_windows)
                    cv2.imshow('Cars', vehicles_frame)
                    out.write(vehicles_frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
    except: # Exception as e:
        print(traceback.format_exc())
        cv2.waitKey(0)
    
    camera.cap.release()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()