# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 19:29:01 2018

@author: ahmed
"""
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Perspective:
    def __init__(self):
        pass
    
    def getTransformParameters(self):
        img = mpimg.imread('test_images/straight_lines1.jpg')
        src = np.float32([[520,500],
                          [200,720],
                          [1100,720],
                          [765,500]])
        dst = np.float32([[300,450], #400],
                          [300,720],#300,720], 
                          [1000,720],#1000,720],
                          [1000,450]]) #400]])
        
        return img, src, dst
    
    def getTransformParameters2(self):
        ''' Get perspective transform parameters for regular project video
        '''
        img = mpimg.imread('test_images/challenge_test01.jpg')
        src = np.float32([[570,500],
                          [280,720],
                          [1120,720],
                          [775,500]])
        dst = np.float32([[300,450], #400],
                          [300,720], 
                          [1000,720],
                          [1000,450]]) #400]])
        
        plt.imshow(img)
        plt.plot(src[:,0],src[:,1], 'o')
        plt.show()
        return img, src, dst
    
    def getTransformParametersLong(self):
        img = mpimg.imread('test_images/straight_lines1.jpg')
        src = np.float32([[610,440], #515,500],
                          [200,720],
                          [1100,720],
                          [670,440]]) #[765,500]])
        dst = np.float32([[400,10], #300,450], #400],
                          [400,720],#300,720], 
                          [900,720],#1000,720],
                          [900,10]]) #1000,450]]) #400]])
        '''
        fig = plt.figure(figsize=(20,15))
        fig.tight_layout()
        plt.imshow(img, interpolation='nearest', aspect='auto')
        plt.plot(src[:,0],src[:,1], 'o')
        plt.show()
        '''
        return img, src, dst