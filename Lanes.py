# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:15:22 2018

@author: ahmed
"""
import numpy as np

class Lane:
    
    def __init__(self, leftx, lefty, rightx, righty, lane_img=None, histogram=None):
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.curverad_tol = 1200
        self.lane_sep_std_tol = 60
        self.is_good_fit = True
        
        self.leftx = leftx
        self.lefty = lefty
        self.rightx = rightx
        self.righty = righty
        self.lane_img = lane_img
        
        self.histogram = histogram
        self.ploty = None
        self.left_coeff = None
        self.right_coeff = None
        self.left_fitx = None
        self.right_fitx = None
        self.left_curverad = None
        self.right_curverad = None
        
            
    def fitPoly(self, ploty):
        # Fit a second order polynomial to each using `np.polyfit`
        self.left_coeff = np.polyfit(self.lefty, self.leftx, 2)
        self.right_coeff = np.polyfit(self.righty, self.rightx, 2)
        
        self.ploty = ploty
        self.calculateByCoeffs(self.left_coeff, self.right_coeff)
    
    def calculateByCoeffs(self, left_coeff, right_coeff):
        '''
        This method is separated out from fitPoly so that it can be called separately
        from LaneSmoother to update the lane by coefficients
        '''
        
        self.left_coeff = left_coeff
        self.right_coeff = right_coeff
        try:
            self.left_fitx = left_coeff[0]*self.ploty**2 + left_coeff[1]*self.ploty + left_coeff[2]
            self.right_fitx = right_coeff[0]*self.ploty**2 + right_coeff[1]*self.ploty + right_coeff[2]
            
            #self.left_fitx = self.left_fitx[self.left_fitx<self.lane_img.shape[1]]
            #self.right_fitx = self.right_fitx[self.right_fitx<self.lane_img.shape[1]]
            
            self.calculateByFit(self.left_fitx, self.right_fitx)
            
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            self.left_fitx = self.ploty**2 + self.ploty + 1
            self.right_fitx = self.ploty**2 + self.ploty + 1
            
            #self.left_fitx = self.left_fitx[self.left_fitx<self.lane_img.shape[1]]
            #self.right_fitx = self.right_fitx[self.right_fitx<self.lane_img.shape[1]]

            self.is_good_fit = False
    
        return self
    
    def calculateByFit(self, left_fitx, right_fitx):
        '''
        This method is separated out so it can also be called externally by LaneSmoother
        with an average left_fitx and right_fitx
        '''
        
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        
        # Get left_fit and right_fit polynomials in meters
        left_fit_xm = np.polyfit(self.ploty*self.ym_per_pix, left_fitx*self.xm_per_pix, 2)
        right_fit_xm = np.polyfit(self.ploty*self.ym_per_pix, right_fitx*self.xm_per_pix, 2)
    
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty*self.ym_per_pix)
        
        # calculate R_curve (radius of curvature)
        self.left_curverad = ((1 + (2*left_fit_xm[0]*y_eval + left_fit_xm[1])**2)**(3/2)) / np.abs(2*left_fit_xm[0])  ## Implement the calculation of the left line here
        self.right_curverad = ((1 + (2*right_fit_xm[0]*y_eval + right_fit_xm[1])**2)**(3/2)) / np.abs(2*right_fit_xm[0])  ## Implement the calculation of the right line here
        
        if(self.isCurveRadGood() and self.isParallel() and self.isEquidistant()):
            self.is_good_fit = True
        else:
            self.is_good_fit = False
            
        return self
    
    def getRadiusOfCurvature(self):
        return self.left_curverad, self.right_curverad
    
    def getCenter(self):
        self.lane_center = (self.right_fitx[-1] - self.left_fitx[-1])
        self.lane_center_xm = self.lane_center*self.xm_per_pix
        return self.lane_center, self.lane_center_xm
    
    def isCurveRadGood(self):
        curve_rad_diff = abs(self.left_curverad-self.right_curverad)
        #print("Radius of Curvature Diff: ", curve_rad_diff)
        return curve_rad_diff<=self.curverad_tol
    
    def isEquidistant(self):
        point_diffs_std = np.std(self.right_fitx - self.left_fitx)
        print("Std Dev of distance between lane points", point_diffs_std)
        return point_diffs_std <= self.lane_sep_std_tol
    
    def isParallel(self):
        return True

class LaneSmoother:
    def __init__(self):
        # x values of the last n fits of the line
        self.n = 10
        
        self.recent_leftx = []
        self.recent_rightx = []
        
        self.recent_left_fitx = []
        self.recent_right_fitx = []
        
        #average x values of the fitted line over the last n iterations
        self.avg_left_fitx = None 
        self.avg_right_fitx = None
        
        self.avg_leftx = None 
        self.avg_rightx = None
        
        self.recent_left_coeff = []
        self.recent_right_coeff = []
        
        #polynomial coefficients averaged over the last n iterations
        self.avg_left_coeff = None
        self.avg_right_coeff = None
        
        #prev good lane
        self.prev_lane = None
    
    def addLanes(self, lane):
        self.recent_leftx.append(lane.leftx)
        self.recent_rightx.append(lane.rightx)
        self.recent_left_coeff.append(lane.left_coeff)
        self.recent_right_coeff.append(lane.right_coeff)
        
    def smooth(self, lane):
        if(lane.is_good_fit):
            print("Good fit...")
            self.prev_lane = lane
            self.recent_leftx.append(lane.leftx)
            self.recent_rightx.append(lane.rightx)
            self.recent_left_coeff.append(lane.left_coeff)
            self.recent_right_coeff.append(lane.right_coeff)
            
        if(len(self.recent_left_fitx)>self.n): # purge more than n entries
            self.recent_leftx = self.recent_leftx[1:]
            self.recent_rightx = self.recent_rightx[1:]
            self.recent_left_coeff = self.recent_left_coeff[1:]
            self.recent_right_coeff = self.recent_right_coeff[1:]
            
        if(len(self.recent_left_fitx)>0):
            self.avg_leftx = np.mean(self.recent_leftx, axis=0)
            self.avg_rightx = np.mean(self.recent_rightx, axis=0)
            self.avg_left_coeff = np.mean(self.recent_left_coeff, axis=0)
            self.avg_right_coeff = np.mean(self.recent_right_coeff, axis=0)
            #print("smooting 1...", self.prev_lane)
            #smoothLane = lane.calculateByCoeffs(self.avg_left_coeff, self.avg_right_coeff)
            smoothLane = Lane(self.avg_leftx, lane.lefty, self.avg_rightx, lane.righty, lane.lane_img, lane.histogram)
            smoothLane = lane.fitPoly(lane.ploty)
            
            #smoothLane.lane_img = self.prev_lane.lane_img
            #print("smooting 2...")
            return smoothLane
        else:
            return lane