# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 02:35:32 2018

@author: ahmed
"""
import cv2
from thresholds import abs_sobel_thresh, mag_thresh, dir_threshold, hls_select, thresh_pipeline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import Lanes


class RoadFrame:
    
    def __init__(self, frameImg):
        self.frame = frameImg
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.frame_ctr = frameImg.shape[1]//2
        self.frame_ctr_xm = self.frame_ctr * self.xm_per_pix
        self.gray = cv2.cvtColor(frameImg, cv2.COLOR_BGR2GRAY)
        self.binary_warped = None
        self.regionOfInterest = np.array([[(100,frameImg.shape[0]), (500, 400), (frameImg.shape[1]-500, 400), (frameImg.shape[1]-100,frameImg.shape[0])]], dtype=np.int32)
        self.perspectiveM = None
        self.perspectiveMInv = None
        self.warped = None
        self.lane = None
        self.vehicle_offset = None
        
    def getRegionOfInterest(self, img):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (1,) * channel_count
        else:
            ignore_mask_color = 1
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, self.regionOfInterest, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def threshold(self, visualize=2):
        # Choose a Sobel kernel size
        ksize = 3 # Choose a larger odd number to smooth gradient measurements
        img = self.warped
        img = cv2.GaussianBlur(img, (5,5), 2)
        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(50, 255))
        #grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(50, 150))
        #mag_bin = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 255))
        dir_bin = dir_threshold(img, sobel_kernel=ksize, thresh=(-np.pi/6, np.pi/6))
        #hls_bin = hls_select(img, h_thresh=(10,50), l_thresh=(255, 255), s_thresh=(255,255))
        
        hls_bin = thresh_pipeline(img, h_thresh=(10,150), s_thresh=(150, 255), sx_thresh=(15, 200))
        
        #hls_bin = thresh_pipeline(img, h_thresh=(10,50), s_thresh=(30, 80), sx_thresh=(15, 200))

        hls_binary = hls_select(img, h_thresh=(50, 100), l_thresh=(100,120), s_thresh=(30,120))

        combined = np.zeros_like(dir_bin)
        combined[(hls_bin==1) | (gradx==1)] = 1 
        
        if(visualize==2):
            # Plot the result
            f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 15))
            f.tight_layout()
            ax1.imshow(self.frame)
            ax1.set_title('Original Image', fontsize=16)
            ax2.imshow(img, cmap='gray')
            ax2.set_title('Warped', fontsize=16)
            ax3.imshow(gradx, cmap='gray')
            ax3.set_title('gradx', fontsize=16)
            ax4.imshow(hls_bin, cmap='gray')
            ax4.set_title('HLS Bin', fontsize=16)
            ax5.imshow(combined, cmap='gray')
            ax5.set_title('combined', fontsize=16)
            #ax4.imshow(hls_binary, cmap='gray')
            #ax4.set_title('HLS Binary', fontsize=16)
            
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            
        return combined
    
    
    def sliding_window(self, binary_warped):
        # This method is mostly copied from Udacity quiz #
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
    
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
    
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        lane = Lanes.Lane(leftx, lefty, rightx, righty, out_img, histogram)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        lane.fitPoly(ploty)
        
        ## Visualization ##
        # Colors in the left and right lane regions
        lane.lane_img[lane.lefty, lane.leftx] = [255, 0, 0]
        lane.lane_img[lane.righty, lane.rightx] = [0, 0, 255]
        #print("leftx = ", lane.leftx)
        #print("left_fitx = ", np.int32(lane.left_fitx))
        #print("lane.lane_img.shape = ", lane.lane_img.shape)
        #print("ploty = ", ploty)
        lane.lane_img[np.int32(ploty), np.int32(lane.left_fitx)] = [0,255,255]
        lane.lane_img[np.int32(ploty), np.int32(lane.right_fitx)] = [0,255,255]
        
        return lane
    
    
    def search_around_poly(self, binary_warped, prev_left_fit, prev_right_fit):
        # This method is mostly copied from Udacity quiz #
        
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 60
    
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        left_line = prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2]
        right_line = prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2]
        left_lane_inds = ((nonzerox > (left_line - margin)) & (nonzerox < (left_line + margin)))
        right_lane_inds = ((nonzerox > (right_line - margin)) & (nonzerox < (right_line + margin)))
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        # Initialize Lane object with identified pixels
        lane = Lanes.Lane(leftx, lefty, rightx, righty, binary_warped)
        # Generate y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        # Fit new polynomials
        lane.fitPoly(ploty)
        
        # Modified the following 2 lines to create search window around the previous polynomial rather than current
        prev_left_fitx = prev_left_fit[0]*ploty**2 + prev_left_fit[1]*ploty + prev_left_fit[2]
        prev_right_fitx = prev_right_fit[0]*ploty**2 + prev_right_fit[1]*ploty + prev_right_fit[2]
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)).astype(np.uint8)*255
        window_img = np.zeros_like(out_img).astype(np.uint8)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([prev_left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([prev_left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([prev_right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([prev_right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1.0, window_img, 0.3, 0)
                        
        ## End visualization steps ##
        
        lane.lane_img = result
        
        lane.lane_img[np.int32(ploty), np.int32(lane.left_fitx)] = [0,255,255]
        lane.lane_img[np.int32(ploty), np.int32(lane.right_fitx)] = [0,255,255]
        
        return lane
    
    def getLane(self, prev_left_fitx=None, prev_right_fitx=None, visualize=2):
        img_size = (self.frame.shape[1], self.frame.shape[0])
        self.warped = cv2.warpPerspective(self.frame, self.perspectiveM, img_size, flags=cv2.INTER_LINEAR)
        self.binary_warped = self.threshold(visualize=visualize)
                
        if(prev_left_fitx is None):
            self.lane = self.sliding_window(self.binary_warped)
        else:
            self.lane = self.search_around_poly(self.binary_warped, prev_left_fitx, prev_right_fitx)
        
        lane_center_pix, lane_center_xm = self.lane.getCenter()
        self.vehicle_offset = self.frame_ctr_xm - lane_center_xm
        
        return self.lane
    
    def highlightLane(self, left_fitx, right_fitx, ploty):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty[:left_fitx.shape[0]]]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty[:right_fitx.shape[0]]])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.perspectiveMInv, (self.frame.shape[1], self.frame.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(self.frame, 1, newwarp, 0.3, 0)
        
        return result
    
    def getVehicleOffset(self):
        return self.vehicle_offset
    
    
    def setPerspectiveTransform(self, M, MInv):
        self.perspectiveM = M
        self.perspectiveMInv = MInv
        
    def getVehicles(self):
        '''
        This method returns bounding boxes around identified vehicles in a single frame
        '''
        vehicleDet = VehicleDetector()
        bboxes = vehicleDet.findVehicles()
        return bboxes
    
        